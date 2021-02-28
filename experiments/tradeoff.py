import torch
import numpy as np
import time
import json

# Source Files
import sys
sys.path.append("..")
import grnn
import exp_utils
import controller
import env.dlqr

# Savedir
filename = 'tradeoff.data'

# Environment Parameters
N = 20
degree = 5 + 1
T = 50
p = 1
q = 1
h = 5
A_norm = 0.995
B_norm = 1

# Training Parameters
num_epoch = 10
batch_size = 20
ensemble_size = 2
val_size = 50
grnn_hidden_dim = 5
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Experiment Parameters
betas = np.logspace(-4, 5, 8)
threshold = 4e-3
num_topologies = 2
num_x0s = 100
verbose = True
num_controllers = 6

# Training losses for different setups
def grnn_criterion(x_traj, u_traj, env, model):
    norms = torch.norm(model.S_()) + torch.norm(model.A)
    return env.cost(x_traj, u_traj) + 0.5 * norms

def sparse_criterion(x_traj, u_traj, env, model, beta):
    return env.cost(x_traj, u_traj) + beta * torch.sum(torch.abs(model.S_()))

#####################  SCRIPT  ##############################
def run():

    # Group parameters that are reused
    model_params = {
            'N':N,
            'T':T,
            'p':p,
            'q':p,
            'h':grnn_hidden_dim
    }
    training_params = {
            'T': T,
            'device': device,
            'num_epoch': num_epoch,
            'batch_size': batch_size,
            'ensemble_size': ensemble_size,
            'val_size': val_size,
    }

    # Create empty arrays to store results
    num_edges = np.zeros((num_topologies, len(betas)))
    reg_costs = np.zeros((num_topologies, len(betas)))
    retrain_costs = np.zeros((num_topologies, len(betas)))
    num_env_edges = np.zeros(num_topologies)

    for j in range(num_topologies):
        print(j, end=', ')

        dlqrenv, G = env.dlqr.generate_lq_env(
                N, degree, device, A_norm=A_norm, B_norm=B_norm)
        num_env_edges[j] = torch.sum(dlqrenv.S > 0).item()
        for i, beta in enumerate(betas):
            # Train on full support to get a model and topology
            model = exp_utils.generate_model(model_params, dlqrenv,
                    use_given_support=False, S=None,
                    criterion=lambda x,u,e,m: sparse_criterion(x,u,e,m,beta),
                    **training_params)
            grnn_S = model.S_().detach()
            reg_costs[j, i] = exp_utils.estimate_controller_cost(
                    dlqrenv, T, [model.get_controller(100)])[1].item()

            # Retrain the model
            new_S = grnn_S.clone()
            new_S[new_S < threshold] = 0
            num_edges[j, i] = torch.sum(new_S > threshold).item()
            model = exp_utils.generate_model(model_params, dlqrenv,
                    use_given_support=True, S=new_S,
                    criterion=grnn_criterion,**training_params)
            retrain_costs[j, i] = exp_utils.estimate_controller_cost(
                    dlqrenv, T, [model.get_controller(100)])[1].item()

    avg_num_edges = num_edges.mean(0)
    avg_reg_costs = reg_costs.mean(0)
    avg_retrain_costs = retrain_costs.mean(0)
    average_edges = num_env_edges.mean()

    print(avg_num_edges)
    print(avg_reg_costs)
    print(avg_retrain_costs)

    with open(filename, 'w') as f:
        result_dict = {
                'num_edges': num_edges.tolist(),
                'reg_costs': reg_costs.tolist(),
                'retrain_cost': retrain_costs.tolist(),
                'env_edges': num_env_edges.tolist(),
                'betas': betas.tolist()
        }
        f.write(json.dumps(result_dict))
#########################################################
