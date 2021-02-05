import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import networkx as nx
import scipy as sp

from .utils import spectral_radius
from .abstractenv import AbstractEnv
import controller

class DistributedLQREnv(AbstractEnv):
    def __init__(self, S, A, B, Q, R, QT):
        """ Initialize the environment. Here we represent the joint system
        state internally by concatenating the state of each agent.
        """
        self.S, self.A, self.B, self.Q, self.R, self.QT = \
                S, A, B, Q, R, QT
        self.N = self.S.size(0)
        self.p = int(self.A.size(0) / self.S.size(0))
        self.q = int(self.B.size(1) / self.S.size(0))
        self.device = self.S.device
        self.P, self.K, self.X = _lqr_solve(A, B, Q, R)
        self.QT = self.P

    def step(self, xi, ui):
        """ Takes in a state and control input, output the next state
        Note that the internal representation has the dimension of the system
        as a 1D array (size N*p), but the conventional representation of the
        system state in the GNN literature (and the representation we in this
        package but outside this environment is 2D of size N x p.
        """
        xi_, ui_ = xi.flatten(), ui.flatten()
        x_next = self.A @ xi_ + self.B @ ui_
        return x_next.view(self.N, -1)

    def sim_forward(self, controller, T, x0=None):
        """ Take an intial condition and controller, outputs the state and control
        trajecories of the system.
        Parameters:
            x0:         torch.tensor(p), initial state
            controller: function/method, takes in a state and outputs the control
            A:          torch.tensor(p, p), state transition matrix
            B:          torch.tensor(p, q), control transition matrix
            T:          integer, time horizon
            device:     device, specifies where to carry out the simulation
        Returns:
            x:          torch.tensor(T, p), state trajectory
            u:          torch.tensor(T, q), control trajectory
        """
        # TODO: enable batch processing
        x = torch.zeros((T+1, self.N, self.p), dtype=torch.double,
                device=self.device)
        u = torch.zeros((T, self.N, self.q), dtype=torch.double,
                device=self.device)
        if x0 is None:
            x0 = self.random_x0()
        x[0] = x0
        for t in range(T):
            ut = controller.control(x[t]).to(self.device)
            x[t+1] = self.step(x[t], ut)
            u[t] = ut
        return x, u

    def random_x0(self):
        """ Generate a random initial condition """
        return torch.randn((self.N, self.p), dtype=torch.double,
                device=self.device)

    def cost(self, x, u):
        """ Compute the cost of a given (x, u) trajectory """
        # note that * has higher precedence than @
        T = u.size(0)
        x_, u_ = x.flatten(1), u.flatten(1)
        state_cost = torch.sum(x_[:T] * (x_[:T] @ self.Q)) + \
                x_[T].dot(self.QT @ x_[T])
        control_cost = torch.sum(u_ * (u_ @ self.R))
        return state_cost + control_cost

    def get_optimal_controller(self):
        return LQRSSController(self.K, self.N)

    def instability_cost(self, x, rho=.9):
        T = x.size(0) - 1
        x_ = x.flatten(1)
        lyap_values = (x_[:T+1] * (x_[:T+1] @ self.X)).sum(1)
        violations = torch.maximum(lyap_values[1:] - rho * lyap_values[:T],
                torch.zeros(T))
        return torch.sum(violations)

    def lyapunov_function(X, state):
        return np.dot(state, X.dot(state))

def _sample_adjacency_matrix(N, degree):
    """ Samples a random adjacency matrix; See below for details """
    rand_vec = np.random.random(N)
    S = np.zeros((N,N))
    for i in range(N):
      inds = np.abs(rand_vec-rand_vec[i]).argpartition(degree)[:degree]
      S[np.ones(degree, dtype=int)*i, inds] = 1
      S[inds, np.ones(degree, dtype=int)*i] = 1
    return S

def _generate_graph(N, degree):
    """ Creates a graph based on the parameters given
    Parameters:
        - N:        integer, number of agents
        - degree:   desired degree of the communication graph
    Returns:
        - G:        nx.Graph, the graph
        - S:        np.array, the adjacency matrix of G
    """
    S = _sample_adjacency_matrix(N, degree)
    G = nx.Graph(S)
    while(not nx.is_connected(G)):
        S = _sample_adjacency_matrix(N, degree)
        G = nx.Graph(S)
    # Normalize graph shift operator S
    S = S / np.linalg.norm(S, ord=2)
    return G, S

def _generate_dynamics(G, S, A_norm=0.995, B_norm=1, AB_hops=3):
    """ Generates dynamics matrices (A,B) based on graph G
    Parameters:
        - G:        nx.Graph, the graph
        - S:        np.array, the adjacency matrix of G
        - A_norm:   float, operator norm of matrix A
        - B_norm:   float, operator norm of matrix B
        - AB_hops:  integer, this is a knob for the "sparsity" of the dynamics.
                    Setting this variable to a value k means that the dynamics
                    of a node can only be affected by its <=k-hop neighbors.
    Returns:
        - A:        np.array, state transition matrix
        - B:        np.array, control transition matrix
    """
    ## TODO:    - Generate sparse A and B based on G
    ##          - make it work for general p and q
    N = S.shape[0]
    eig_vecs = np.linalg.eig(S)[1]
    A = eig_vecs @ np.diag(np.random.randn(N)) @ eig_vecs.T
    B = eig_vecs @ np.diag(np.random.randn(N)) @ eig_vecs.T
    # Zero out the >k-hop neighbors
    distances = nx.floyd_warshall_numpy(G)
    distance_mask = (distances > AB_hops)
    A[distance_mask] = 0
    B[distance_mask] = 0
    # Normalize A and B
    A = A / np.linalg.norm(A, ord=2) * A_norm
    B = B / np.linalg.norm(B, ord=2) * B_norm
    return A, B

def generate_lq_env(N, degree, device, A_norm=0.995, B_norm=1, AB_hops=3):
    """ Generate all environment parameters using given parameters
    Parameters:
        - N:        integer, number of agents
        - degree:   integer, desired degree of the communication graph
        - device:   None/string, specifies where to move the parameters
                    if is None, then return the np.array directy without
                    casting to tensor
    Returns:
        - G:        nx.Graph, the graph
        - S:        torch.tensor(N, N), the adjacency matrix of G
        - A:        torch.tensor(Np, Np), state transition matrix
        - B:        torch.tensor(Np, Nq), control transition matrix
        - Q, R:     torch.tensor(Np, Np), cost matrices. Here just returns identity
    """
    G, S = _generate_graph(N, degree)
    A, B = _generate_dynamics(G, S, A_norm, B_norm, AB_hops)
    Q = np.eye(N)
    R = np.eye(N)
    # Cast into tensor and move to device if needed
    if device is not None:
        S = torch.tensor(S).to(device)
        A = torch.tensor(A, dtype=torch.double).to(device)
        B = torch.tensor(B, dtype=torch.double).to(device)
        Q = torch.tensor(Q).to(device)
        R = torch.tensor(R).to(device)
    env = DistributedLQREnv(S, A, B, Q, R, Q)
    return env, G

def _lqr_solve(A, B, Q=None, R=None, rho=.9, eps=.01):
    """Solve the discrete time lqr controller. Also solve the lyapunov equation
    to find a Lyapunov function. System given as (Ref Bertsekas, p.151)
                    x[k+1] = A x[k] + B u[k]
                    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    Parameters:
        - A:        torch.tensor(p, p), state transition matrix
        - B:        torch.tensor(p, q), control transition matrix
        - Q, R:     torch.tensor(p, p), cost matrices. Default to identity
        - rho:      float, exponential decay rate of the Lyapunov function
        - eps:      float, robustness parameter
    Returns:
        - P:        torch.tensor(p, p), steady-state cost matrix
        - K:        torch.tensor(q, p), optimal feedback controller
        - X:        torch.tensor(p, p), Lyapunov function is given by f(x)=x'Xx
    """
    # Cast everything to array to use scipy methods
    if Q is None:
        Q_ = np.eye(A.shape[0])
    else:
        Q_ = Q.cpu().numpy().copy()
    if R is None:
        R_ = np.eye(B.shape[1])
    else:
        R_ = R.cpu().numpy().copy()
    A_, B_ = A.cpu().numpy().copy(), B.cpu().numpy().copy()
    # Solve for P and K
    P = sp.linalg.solve_discrete_are(A_, B_, Q_, R_)
    K = -sp.linalg.solve(
            B_.T.dot(P).dot(B_) + R_, B_.T.dot(P).dot(A_), sym_pos=True)
    # Check for spectral radius of closed-loop dynamics
    A_c = A_ + B_.dot(K)
    TOL = 1e-5
    if spectral_radius(A_c) >= 1 + TOL:
        print("WARNING: spectral radius of closed loop is:", spectral_radius(A_c))
    # Solve for solution to
    #               (A + B K_lqr)' X (A + B K_lqr) - X + Q = 0
    #                   with Q = (1-\rho)X + eps I s.t.
    #               Acl ' X Acl = \rho X - eps I <= \rho X
    # (note transpose on Acl in DLYAP because of scipy implementation)
    # V(x) = x'Xx, s.t. V(x(t+1)) <= rho V(x(t)), (Acl x)'X(Acl x) <= x' X x' 
    X = sp.linalg.solve_discrete_lyapunov(
            (A_ + B_ @ K).T, (1-rho) * P + eps * np.eye(P.shape[0]))
    return torch.tensor(P, device=A.device), \
            torch.tensor(K, device=A.device), \
            torch.tensor(X, device=A.device)

def decrease_soft_constraint(X, x, controller):
    def f_closed_loop(x):
        return A.dot(x) + B.dot(controller(x))
    V_plus = lyapunov_function(X, f_closed_loop(x))
    V_cur = lyapunov_function(X, x)
    return np.maximum(V_plus - rho * V_cur, 0.0)

class LQRSSController(controller.AbstractController):
    def __init__(self, K, N):
        self.K = K
        self.N = N
    def control(self, x):
        return (self.K @ x.flatten()).view(self.N, -1)
