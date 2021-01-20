import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import networkx as nx
import scipy as sp

from utils import spectral_radius
import controller

def _generate_graph(N, degree):
    """ Creates a graph based on the parameters given
    Parameters:
        - N:        integer, number of agents
        - degree:   desired degree of the communication graph
    Returns:
        - G:        nx.Graph, the graph
        - S:        np.array, the adjacency matrix of G
    """
    rand_vec = np.random.random(N)
    S = np.zeros((N,N))
    for i in range(N):
      inds = np.abs(rand_vec-rand_vec[i]).argpartition(degree)[:degree]
      S[np.ones(degree, dtype=int)*i, inds] = 1
      S[inds, np.ones(degree, dtype=int)*i] = 1
    G = nx.Graph(S)
    # Normalize graph shift operator S
    S = S / np.linalg.norm(S, ord=2)
    return G, S

def _generate_dynamics(G, S):
    """ Generates dynamics matrices (A,B) based on graph G
    Parameters:
        - G:        nx.Graph, the graph
        - S:        np.array, the adjacency matrix of G
    Returns:
        - A:        np.array, state transition matrix
        - B:        np.array, control transition matrix
    """
    ## TODO: Generate sparse A and B based on G
    N = S.shape[0]
    eig_vecs = np.linalg.eig(S)[1]
    A = eig_vecs @ np.diag(np.random.randn(N)) @ eig_vecs.T
    B = eig_vecs @ np.diag(np.random.randn(N)) @ eig_vecs.T
    A = A / np.linalg.norm(A, ord=2) * 0.995
    B = B / np.linalg.norm(B, ord=2)
    return A, B

def generate_lq_env(N, degree, device):
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
        - A:        torch.tensor(p, p), state transition matrix
        - B:        torch.tensor(p, q), control transition matrix
        - Q, R:     torch.tensor(p, p), cost matrices. Here just returns identity
    """
    G, S = _generate_graph(N, degree)
    A, B = _generate_dynamics(G, S)
    Q = np.eye(N)
    R = np.eye(N)
    # Cast into tensor and move to device if needed
    if device is not None:
        S = torch.tensor(S).to(device)
        A = torch.tensor(A, dtype=torch.double).to(device)
        B = torch.tensor(B, dtype=torch.double).to(device)
        Q = torch.tensor(Q).to(device)
        R = torch.tensor(R).to(device)
    return G, S, A, B, Q, R

def sim_forward(x0, controller, A, B, T, device):
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
    x = torch.zeros((T+1, A.size(0)), dtype=torch.double, device=device)
    u = torch.zeros((T, A.size(0)), dtype=torch.double, device=device)
    x[0,:] = x0
    for i in range(T):
        ui = controller(x[i,:]).to(device)
        x[i+1,:] = A @ x[i,:] + B @ ui
        u[i,:] = ui
    return x, u

def lqr_solve(A, B, Q=None, R=None, rho=.9, eps=.01):
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
            (A + B @ K).T, (1-rho) * P + eps * np.eye(P.shape[0]))
    return torch.tensor(P, device=A.device), \
            torch.tensor(K, device=A.device), \
            torch.tensor(X, device=A.device)

def lyapunov_function(X, state):
    return np.dot(state, X.dot(state))

def decrease_soft_constraint(X, x, controller):
    def f_closed_loop(x):
        return A.dot(x) + B.dot(controller(x))
    V_plus = lyapunov_function(X, f_closed_loop(x))
    V_cur = lyapunov_function(X, x)
    return np.maximum(V_plus - rho * V_cur, 0.0)

def LQ_cost(x, u, Q, R, QT):
    """ Compute the LQR cost of a given state and control trajectory
    Parameters:
        - x:        torch.tensor(T+1, p), state trajectory
        - u:        torch.tensor(T, q), control trajectory
        - Q:        torch.tensor(p, p), state cost matrix
        - R:        torch.tensor(q, q), control cost matrix
        - QT:       torch.tensor(p, p), terminal state cost matrix
    Return:
        - cost:     float
    """
    # note that * has higher precedence than @
    T = u.size(0)
    state_cost = torch.sum(x[:T] * (x[:T] @ Q)) + x[T].dot(QT @ x[T])
    control_cost = torch.sum(u * (u @ R))
    return state_cost + control_cost

class LQRSSController(controller.AbstractController):
    def __init__(self, A, B, Q, R, T):
        self.P, self.K, _ = lqr_solve(A, B, Q, R)
        self.T = T
    def control(self, x):
        return self.K @ x
