import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import grnn, env.dlqr

def train_it(model, env, criterion, batch_size):
    error = 0
    for i in range(batch_size):
        x0 = env.random_x0()
        xtraj, utraj = model.forward(x0, env.step)
        error += criterion(xtraj, utraj, env, model)
    loss = error / batch_size
    return loss

def train_model(model, env, criterion,
        batch_size=20, num_epoch=100, verbose=False):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(num_epoch):
        model.zero_grad()
        loss = train_it(model, env, criterion, batch_size=batch_size)
        if(verbose and epoch % 10 == 0):
            print('Epoch: {} \t Loss: {}'.format(epoch+1, loss.item()))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()
    return model

def grnn_topology(env, criterion, model_params, threshold, device, batch_size=20,
        num_epoch=100, verbose=False):
    model = grnn.GRNN(None, **model_params).to(device)
    model = train_model(model, env, criterion, num_epoch=num_epoch)
    return model.S_().detach()

def sim_controllers(env, x0, controllers, T, device):
    """ Simulates a list of controllers for one episode
    """
    xs, us, costs = [], [], []
    for c in controllers:
        x, u = env.sim_forward(c, T, x0=x0)
        cost = env.cost(x, u)
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

def plot_controllers(xs, names, costs, rel_cost_wrt=None, dim2plot=0):
    """ Plot trajectories
    Parameters:
        - rel_cost_wrt:     if is None, print absolute cost, o.w. normalize
        - dim2plot:         which dimension of states to plot
    Return:
        - None
    """
    num_controllers, T, N, p = xs.shape
    costs_to_print = costs.copy()
    if rel_cost_wrt is not None:
        costs_to_print = relative_costs(costs_to_print, rel_cost_wrt)
    plt.figure(figsize=(num_controllers * 5, 5))
    for i in range(num_controllers):
        plt.subplot(1, num_controllers, i+1)
        for j in range(N):
          plt.plot(np.arange(T), xs[i, :, j, dim2plot])
          plt.title(names[i]+'\nCost={:.3f}'.format(costs_to_print[i]))

def estimate_grnn_cost(env, model, T, additional_controllers=[], num_x0s=100):
    """ Estimate the cost of GRNN on the given environment and compare to other
    controllers.
    """
    optctrl = env.get_optimal_controller()
    grnnctrl = model.get_controller()
    controllers = [optctrl, grnnctrl] + additional_controllers
    sum_costs = np.zeros(len(controllers))
    for _ in range(num_x0s):
        x0 = env.random_x0()
        _, _, costs = sim_controllers(env, x0, controllers, T, x0.device)
        sum_costs = sum_costs + relative_costs(costs, 0)
    return sum_costs / num_x0s

