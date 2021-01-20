import numpy as np
import torch

class AbstractController:
    """ Abstract class of controllers, to be inherited by the other classes
    """
    def __init__(self):
        self.counter = 0
    def reset(self):
        self.counter = 0
    def control(self, x):
        raise NotImplementedError

class RandomController(AbstractController):
    """ Generates random (normal) control actions
    """
    def __init__(self, q, mu=0, std=1):
        super().__init__()
        self.q = q
        self.mu = mu
        self.std = std
    def control(self, x):
        unit_rand = torch.randn(self.q, dtype=torch.double, device=x.device)
        return unit_rand * self.std + self.mu

class ZeroController(AbstractController):
    """ Generates zeros control actions. (So we get the autonomous dynamics)
    """
    def __init__(self, q):
        super().__init__()
        self.q = q
    def control(self, x):
        return torch.zeros(self.q, dtype=torch.double, device=x.device)

class GRNNController(AbstractController):
    """ A Controller wrapper for the parameters generated by our GRNN model
    """
    def __init__(self, model):
        super().__init__()
        self.S, self.A, self.B = model.get_params()
        self.Z = self.A.new_zeros((self.S.size(0), self.A.size(1)))
    def control(self, x):
        # TODO: stardardize the representation of states
        with torch.no_grad():
            self.Z = torch.tanh(self.S @ self.Z + x.unsqueeze(1) @ self.A)
            u = self.Z @ self.B
        return u[:,0]
    def reset(self):
        # In addition to resetting the counter, also reset hidden states
        super().reset()
        self.Z = Z.new_zeros(Z.size())

class LQRController:
    def __init__(self, A, B, Q, R, T):
        self.Ks = lqr_recursion(A, B, Q, R, T)
        self.counter = 0

    def control(self, x):
        u = self.Ks[self.counter] @ x
        self.counter += 1
        return u