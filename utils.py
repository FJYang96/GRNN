import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

def spectral_radius(A):
    return np.max(np.abs(np.linalg.eigvals(A)))

def train_it(model, criterion, N, A, B, Q, R, QT, beta=0, batch_size=20):
    error = 0
    for i in range(batch_size):
        x0 = torch.randn(N)
        xtraj, utraj = model.forward(x0, A, B)
        error += criterion(xtraj, utraj, Q, R, QT)
    regularization = torch.norm(model.S_(),1)
    loss = error / batch_size + beta * regularization
    return loss

def train_model(model, criterion, S, N, T, A, B, Q, R, QT,
        beta=0, num_epoch=100, verbose=False):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(num_epoch):
        model.zero_grad()
        loss = train_it(model, criterion, N, A, B, Q, R, QT, beta=beta)
        if(verbose and epoch % 10 == 0):
            print('Epoch: {} \t Loss: {}'.format(epoch+1, loss.item()))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()
    return model

def sim_controllers(sim_func, cost_func, x0, controllers, device):
    """ Simulates a list of controllers for one episode
    """
    xs, us, costs = [], [], []
    for c in controllers:
        x, u = sim_func(x0, c, device)
        cost = cost_func(x, u)
        xs.append(x); us.append(u)
        costs.append(cost.detach().cpu().item())
    xs = torch.stack(xs, dim=0).detach().cpu().numpy()
    us = torch.stack(us, dim=0).detach().cpu().numpy()
    costs = np.array(costs)
    return xs, us, costs

def relative_costs(costs, rel_cost_wrt):
    """ Normalize the cost by dividing all costs by one single cost
    Parameters:
        - costs:            np.array, costs for different controllers
        - rel_cost_wrt:     int, the index w.r.t. which we normalize the cost
    Return:
        - rel_costs:        the normalized costs
    """
    return costs / costs[rel_cost_wrt]

def plot_controllers(xs, names, costs, rel_cost_wrt=None):
    """ Plot trajectories
    Parameters:
        - rel_cost_wrt:     if is None, print absolute cost, o.w. normalize
    Return:
        - None
    """
    num_controllers, T, N = xs.shape
    costs_to_print = costs.copy()
    if rel_cost_wrt is not None:
        costs_to_print = relative_costs(costs_to_print, rel_cost_wrt)
    plt.figure(figsize=(num_controllers * 5, 5))
    for i in range(num_controllers):
        plt.subplot(1, num_controllers, i+1)
        for j in range(N):
          plt.plot(np.arange(T), xs[i, :, j])
          plt.title(names[i]+'\nCost={:.3f}'.format(costs_to_print[i]))
