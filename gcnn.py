import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import networkx as nx
from controller import GCNNController

class GCNN(nn.Module):
    """ A simplified version of the Graph Recurrent Neural Network
    """
    def __init__(self, S, N, T, kgnn=3, fgnn=32, p=1, q=1):
        """ Constructor
        Parameters:
            - S:            None/torch.tensor(N,N), the communication topolgy
                            if is None, then the topology will be treated as a
                            parameter to be optimized
            - N:            integer, number of agents
            - T:            integer, time horizon
            - p:            integer, state dimension of each agent
            - q:            integer, control dimension of each agent
            - h:            integer, hidden state dimension of each agent
        """
        super().__init__()
        self.N, self.T, self.p, self.q = N, T, p, q
        self.kgnn = kgnn
        self.fgnn = fgnn
        # Initialize S
        self.S = S
        H1 = torch.randn((kgnn+1, p, fgnn), dtype=torch.double)
        H2 = torch.randn((1, fgnn, p), dtype=torch.double)
        self.register_parameter(name='H1',
                param=torch.nn.Parameter(H1))
        self.register_parameter(name='H2',
                param=torch.nn.Parameter(H2))

    def _graph_conv(self, X):
        Z = X @ self.H1[0]
        for i in range(1, self.kgnn+1):
            Z = Z + torch.matrix_power(self.S, i) @ X @ self.H1[i]
        Z = torch.tanh(Z)
        U = self.S @ Z @ self.H2[0]
        return U, Z

    def forward(self, x0, step):
        batch_size = x0.size(0)
        x_traj = self.S.new_empty((batch_size, self.T+1, self.N, self.p))
        u_traj = self.S.new_empty((batch_size, self.T, self.N, self.q))
        x_traj[:,0,:,:] = x0
        for t in range(self.T):
          xt = x_traj[:,t,:,:].clone()
          ut, zt = self._graph_conv(xt)
          x_traj[:,t+1,:,:] = step(xt, ut)
          u_traj[:,t,:,:] = ut
        return x_traj, u_traj

    def get_params(self):
        return self.S.detach().clone(),\
                self.H1.detach().clone(),\
                self.H2.detach().clone()

    def get_controller(self, batch_size):
        return GCNNController(self, batch_size)

