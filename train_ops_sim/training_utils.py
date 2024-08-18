import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from numba import njit, prange
from numba.typed import List as nblist
import os

from .sim import pos_tp, reward_tp, TrainItinerarySim

sim_tp = TrainItinerarySim.class_type.instance_type

@njit(parallel=True)
def par_init_env(rngs_sim, rngs_act, state_dim : int, sim_args, env_stride : float = 1.0):
    '''
    Initializes a list of simulation environments in parallel, each with its own random seed.
    Executes the first step of each environment, and then a number of steps that increasess with the environment index, in the amount of env_stride.
    This allows the RL algorithm to see in the same gradient environments that are at different stages of the simulation.
    Returns the list of environments and the initial states.
    '''
    n_envs = len(rngs_sim)
    envs = nblist.empty_list(sim_tp, allocated= n_envs)
    states = np.empty((n_envs, state_dim), dtype= pos_tp)
    for k in range(n_envs):
        envs.append(TrainItinerarySim(rngs_sim[k], False, np.inf, *sim_args))
    for k in prange(n_envs):
        rng = rngs_act[k]
        steps = round(1.0 + env_stride * k)
        action_prob = rng.random(dtype=np.float32)
        actions = rng.random(steps, dtype=np.float32) < action_prob
        envs[k].first_step(False)
        for i in range(steps-1):
            envs[k].step(actions[i], False)
        _, state = envs[k].step(actions[steps-1])
        states[k,:] = state
    return envs, states

@njit(parallel=True, cache= True)
def par_step(sims, actions, n_threads : int, state_dim : int):
    '''
    Executes a single step per environment in parallel, applying the action given by the optimization algorithm.
    Returns the rewards and new states per environment.
    '''
    ns = len(sims)
    thread_samps = ns // n_threads
    idxs = list(range(0, ns+1, thread_samps))
    states = np.empty((ns,state_dim), dtype= pos_tp)
    rewards = np.empty(ns, dtype= reward_tp)
    for i in prange(n_threads):
        local_states = np.empty((thread_samps,state_dim), dtype= pos_tp)
        local_rewards = np.empty(thread_samps, dtype= reward_tp)
        for k in range(thread_samps):
            loc = idxs[i]+k
            rw, st = sims[loc].step(actions[loc])
            local_states[k,:] = st.reshape(1,-1)
            local_rewards[k] = rw
        states[idxs[i] : idxs[i+1], :] = local_states
        rewards[idxs[i] : idxs[i+1]] = local_rewards
    return rewards, states

def compute_gradient_norm(model):
    '''
    Computes the norm of the gradient for the combined actor-critic loss. This is used for gradient clipping.
    '''
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return np.sqrt(total_norm)

def pre_proc_states(device, spline_t, states):
    ''''
    Receives the states directly from the simulation and adds spline features, while removing the last component (the end state flag).
    '''
    n_trains = spline_t.n_features_in_
    return torch.FloatTensor(
        np.concatenate((
            states[:,:-1],
            spline_t.transform(states[:,11:11+n_trains])
        ), axis= 1)
        ).to(device)

def plot_training_stats(act_losses, crit_losses, entropies, action_avgs, action_stds,
                        rw_hist, val_hist, grad_norms, raw_rw_avg, grads_per_epi,
                        critic_w, entropy_w, steps_per_grad, avg_rw, std_rw,
                        model_suffix : str = '', show_plots : bool = True
                        ):
    '''
    Plots the training losses, action probabilities, rewards, gradients and target vs prediction for the actor-critic network.
    Plots are saved to the training_plots folder, and optionally shown in output.
    '''
    if not os.path.exists('training_plots'):
        os.makedirs('training_plots')
    # Unscaled loss plot
    loss_df = pd.DataFrame({'actor_loss' : act_losses, 'critic_loss' : crit_losses, 'entropy_loss' : entropies}).rolling(grads_per_epi, min_periods= 1).mean().iloc[::max(1,grads_per_epi//2),:]
    loss_df.plot(figsize= [18,5])
    plt.title('Unscaled losses')
    plt.xlabel('Gradient step')
    plt.ylabel('Loss')
    plt.savefig(f'training_plots/unscaled_losses{model_suffix}.svg', dpi=200, bbox_inches='tight')
    plt.savefig(f'training_plots/unscaled_losses{model_suffix}.png', dpi=200, bbox_inches='tight')
    if show_plots:
        plt.show()
    else:
        plt.close('all')
    
    # Scaled loss plot
    loss_df['critic_loss'] *= critic_w
    loss_df['entropy_loss'] *= entropy_w[1]
    loss_df.plot(figsize= [18,5])
    plt.title('Algorithm losses')
    plt.xlabel('Gradient step')
    plt.ylabel('Loss')
    plt.savefig(f'training_plots/real_losses{model_suffix}.svg', dpi=200, bbox_inches='tight')
    plt.savefig(f'training_plots/real_losses{model_suffix}.png', dpi=200, bbox_inches='tight')
    if show_plots:
        plt.show()
    else:
        plt.close('all')
    
    # Action probabilities plot
    pd.DataFrame({'action_avg' : action_avgs, 'action_std' : action_stds}).rolling(steps_per_grad*grads_per_epi, min_periods= 1).mean().iloc[::steps_per_grad,:].plot(figsize= [18,5])
    plt.title('Action proba')
    plt.xlabel('Environment step')
    plt.ylabel('Probability')
    plt.savefig(f'training_plots/action_proba{model_suffix}.svg', dpi=200, bbox_inches='tight')
    plt.savefig(f'training_plots/action_proba{model_suffix}.png', dpi=200, bbox_inches='tight')
    if show_plots:
        plt.show()
    else:
        plt.close('all')
    print('Last action proba avg: ', action_avgs[-1])

    # Rewards, average per step
    pd.Series(raw_rw_avg, name= 'Average step reward').plot()
    plt.title('Average reward (negative waiting time) per step')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig(f'training_plots/reward_per_step{model_suffix}.svg', dpi=200, bbox_inches='tight')
    plt.savefig(f'training_plots/reward_per_step{model_suffix}.png', dpi=200, bbox_inches='tight')
    if show_plots:
        plt.show()
    else:
        plt.close('all')
    
    # Rewards at each episode progress
    avg_rw_df = pd.DataFrame(rw_hist.mean(axis= 1).reshape(-1, grads_per_epi))
    avg_rw_df.columns = [f'Grad {k}: {int(k*steps_per_grad)} env steps' for k in range(1,grads_per_epi+1)]
    avg_rw_df.iloc[-100:,::int(grads_per_epi//5.5)].plot(figsize= [18,5])
    plt.title('Target (N-step reward + next state value) per episode stage')
    plt.xlabel('Episode (last 100)')
    plt.ylabel('Avg batch target')
    plt.savefig(f'training_plots/target_at_step{model_suffix}.svg', dpi=200, bbox_inches='tight')
    plt.savefig(f'training_plots/target_at_step{model_suffix}.png', dpi=200, bbox_inches='tight')
    if show_plots:
        plt.show()
    else:
        plt.close('all')
    
    # Target vs prediction scatter
    rw_val_df = pd.DataFrame({'prediction' : val_hist[-20:,:20].reshape(-1), 'target' : rw_hist[-20:,:20].reshape(-1)})
    rw_val_df.plot(x= 'target', y= 'prediction', kind= 'scatter', figsize= [6,6])
    axis_range = [
        min(rw_val_df['target'].min(), rw_val_df['prediction'].min()),
        max(rw_val_df['target'].max(), rw_val_df['prediction'].max())
    ]
    plt.xlim(axis_range)
    plt.ylim(axis_range)
    plt.plot(axis_range, axis_range, 'r--', label='y=x', linewidth= 1.0)
    plt.title('Target vs Prediction (last 20 episodes, batch sample 20)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Prediction')
    plt.ylabel('Target')
    plt.savefig(f'training_plots/prediction_scatter{model_suffix}.svg', dpi=200, bbox_inches='tight')
    plt.savefig(f'training_plots/prediction_scatter{model_suffix}.png', dpi=200, bbox_inches='tight')
    if show_plots:
        plt.show()
    else:
        plt.close('all')
    
    # Target and prediction histogram
    plt.figure(figsize=(10, 6))
    plt.hist((rw_hist[-100:,:] * std_rw + avg_rw).reshape(-1), bins= 50, color= (0,0,1,0.5), density= True, label= 'RL target')
    plt.hist((val_hist[-100:,:] * std_rw + avg_rw).reshape(-1), bins= 50, color= (1,0,0,0.5), density= True, label= 'Critic output')
    plt.legend()
    plt.title('Target vs prediction histogram (last 100 episodes)')
    plt.xlabel('Unscaled values')
    plt.ylabel('Density')
    plt.savefig(f'training_plots/prediction_histogram{model_suffix}.svg', dpi=200, bbox_inches='tight')
    plt.savefig(f'training_plots/prediction_histogram{model_suffix}.png', dpi=200, bbox_inches='tight')
    if show_plots:
        plt.show()
    else:
        plt.close('all')
    
    # Gradient norm histogram
    plt.figure(figsize=(10, 6))
    plt.hist(grad_norms[-grads_per_epi*100:], bins=30, density= True)
    plt.title('Histogram of Gradient Norms (last 100 episodes)')
    plt.xlabel('Gradient Norm')
    plt.ylabel('Density')
    plt.savefig(f'training_plots/gradient_norm{model_suffix}.svg', dpi=200, bbox_inches='tight')
    plt.savefig(f'training_plots/gradient_norm{model_suffix}.png', dpi=200, bbox_inches='tight')
    if show_plots:
        plt.show()
    else:
        plt.close('all')
    print('Gradient norm 95% percentile: ', np.quantile(grad_norms[-grads_per_epi*100:], 0.95))