import numpy as np
import torch
import torch.optim as optim
from torchinfo import summary as model_summary
from tqdm import tqdm
from sklearn.preprocessing import SplineTransformer
import os
from typing import Tuple

from .input_preproc import preproc_sim_params
from .training_nn import ACTrainNN
from .training_utils import par_init_env, par_step, compute_gradient_norm, pre_proc_states, plot_training_stats

def train_a2c(filename : str
              , model_suffix : str = ''
              , num_episodes: int = 1500
              , grads_per_epi: int = 44
              , steps_per_grad: int = 5
              , gamma: Tuple[float,float] = (0.95,0.998)
              , lr: float = 0.0005
              , n_threads : int = -1
              , replicas_per_thread : int = 32
              , entropy_w: Tuple[float,float] = (0.035, 0.015)
              , critic_w : float = 1.0
              , NN_args: Tuple[int,int] = (8,32)
              , spline_knots : int = 10
              , spline_degree : int = 1
              , show_plots : bool = True
             ):
    '''
    Trains the actor-critic network using a combined loss with entropy regularization.
    The gradient algorithm is RMSprop.
    The batch size is n_threads * replicas_per_thread.
    The simulation steps are done in parallel through independent environments.
    The reward is defined as the negative of the waiting time observed in the simulation.
    The reward accumulates over discounted N-step returns, where N = steps_per_grad. After the N-step return, a gradient step is performed.
    After grads_per_epi gradient steps, all the environments are discarded and replaced with new random starting points.
    This is done so that the RL algorithm has exposure to both chaotic starting points and regular operations.
    Gradient clipping is used to prevent large updates. Also, reward scaling is used to have similar scale of values in all losses.

    After training, the model is saved to disk with the name a2c{model_suffix}.pth, and several plots with training diagnostics are also saved.
    '''
    # Determine CPU threads and CUDA availability
    if n_threads == -1:
        n_threads = os.cpu_count()
        print('Simulation threads: ', n_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Torch device: ', device)
    
    # Read circuit spreadsheet
    sim_args, _, _, _ = preproc_sim_params(filename)
    
    # Sequences of discount factor and entropy weight per episode
    gammas = np.concatenate((1.0-1.0/np.linspace(1.0/(1.0-gamma[0]), 1.0/(1.0-gamma[1]), int(np.ceil(0.75*num_episodes))), np.full(num_episodes-int(np.ceil(0.75*num_episodes)), gamma[1])))
    entropy_ws = np.logspace(np.log10(entropy_w[0]), np.log10(entropy_w[1]), num_episodes)
    
    # Spline transformer for feature pre-processing. It transforms each train position into n_knots + degree - 1 features
    spline_t = SplineTransformer(n_knots= spline_knots, degree= spline_degree, extrapolation='constant')
    n_trains = sum(sim_args[2])
    spline_t.fit(np.repeat(np.linspace(0,1,3, dtype= np.float32).reshape(-1,1), n_trains, axis= 1))
    
    # Initialize the actor-critic net
    n_segments, ntr_per_circ, spline_dim, batch_size = len(sim_args[4]), sim_args[2], \
                                                        int(spline_t.n_features_out_ / n_trains), n_threads*replicas_per_thread
    model_base = ACTrainNN(n_segments, ntr_per_circ, spline_dim, device, batch_size, *NN_args).to(device)
    
    # Print info and compile
    sim_state_dim = 12 + 2*n_trains
    nn_input_feats = int(sim_state_dim - 1 + n_trains*spline_dim)
    print(model_summary(model_base, (batch_size, nn_input_feats), verbose= 0))
    #model = torch.compile(model_base) # Compilation didn't give better results
    model = model_base
    
    # Initialize optimizer
    optimizer = optim.RMSprop(model.parameters(), lr=lr)

    # Arrays to store losses, entropy, action prob stats, rewards, values and grand norms for diagnostics
    act_losses = np.zeros(num_episodes * grads_per_epi)
    crit_losses = np.zeros(num_episodes * grads_per_epi)
    entropies = np.zeros(num_episodes * grads_per_epi)
    action_avgs = np.zeros(num_episodes * grads_per_epi * steps_per_grad)
    action_stds = np.zeros(num_episodes * grads_per_epi * steps_per_grad)
    rw_hist = np.zeros((num_episodes * grads_per_epi, batch_size))
    val_hist = np.zeros((num_episodes * grads_per_epi, batch_size))
    grad_norms = np.zeros(num_episodes * grads_per_epi)
    raw_rw_avg = np.zeros(num_episodes)

    # Episodes loop
    for episode in tqdm(range(num_episodes)):
        gamma_ep = gammas[episode]
        entropy_w_ep = entropy_ws[episode]
        
        # Initialize environments for the episode, and obtain initial state
        seeds = [s.spawn(2) for s in np.random.SeedSequence(episode).spawn(batch_size)]
        rngs_sim = [np.random.Generator(np.random.SFC64(s[0])) for s in seeds]
        rngs_act = [np.random.Generator(np.random.SFC64(s[1])) for s in seeds]
        envs, states = par_init_env(rngs_sim, rngs_act, sim_state_dim, sim_args, env_stride= 400.0/batch_size)

        # Gradient steps loop
        for step_g in range(grads_per_epi):
            state_tensor = pre_proc_states(device, spline_t, states)
            action_probs, values = model(state_tensor) # The values computed here will be used as prediction in the TD error for the gradient update
            
            # Get log_probs for actor gradient computation, and entropy for loss
            dist = torch.distributions.Bernoulli(action_probs)
            entropy = -dist.entropy().mean()
            actions = dist.sample()
            log_probs = dist.log_prob(actions)
            
            gamma_step = 1.0
            rewards_np = np.zeros(batch_size, dtype= np.float64)
            
            # N-step returns loop
            model.eval() # During the N-step return, the model is used only for evaluation
            with torch.no_grad():
                for step_m in range(steps_per_grad):
                    if step_m > 0: # On the first step, skip the forward pass since it's already been performed
                        state_tensor = pre_proc_states(device, spline_t, states)
                        action_probs, _ = model(state_tensor) # Only keep the action probs, since the values are only needed for first and last step
                        actions = torch.distributions.Bernoulli(action_probs).sample()
                    
                    actions_np = actions.cpu().numpy().astype(bool)
                    act_prob_np = action_probs.detach().cpu().numpy()
                    # Store action averages and stds for analysis
                    act_avg = act_prob_np.mean()
                    act_std = act_prob_np.std()
                    action_avgs[step_m + step_g*steps_per_grad + episode*grads_per_epi*steps_per_grad] = act_avg
                    action_stds[step_m + step_g*steps_per_grad + episode*grads_per_epi*steps_per_grad] = act_std
                    # Get rewards and states for the next iteration
                    rewards_step, states = par_step(envs, actions_np, n_threads, states.shape[1])
                    # Update rewards and increase discount rate
                    rewards_np += rewards_step * gamma_step
                    gamma_step *= gamma_ep
                    raw_rw_avg[episode] += rewards_step.mean() / (grads_per_epi*steps_per_grad)

                # Store reward avg and std of first batch every 10 episodes for scaling
                if not(episode%10) and step_g == 0:
                    avg_rw = rewards_np.mean()
                    std_rw = rewards_np.std()

                rewards_np = (rewards_np - avg_rw) / std_rw # Scale rewards to have similar scale in all losses

                # Get values of next state
                next_state_tensor = pre_proc_states(device, spline_t, states)
                _, next_values = model(next_state_tensor)
            
            model.train() # Set the model back to training mode
            # Compute TD error
            target = torch.FloatTensor(rewards_np.reshape(-1,1)).requires_grad_(False).to(device)
            target += gamma_step * next_values.detach() # Detach the next_values to consider them as constant in the target
            td_error = target - values.detach()
            
            # Calculate losses
            actor_loss = -(log_probs * td_error).mean() 
            critic_loss = 0.5 * torch.nn.functional.mse_loss(values, target)
            loss = actor_loss + critic_loss * critic_w + entropy * entropy_w_ep

            # Store losses, rewards, values for diagnostics
            grad_idx = step_g + episode*grads_per_epi
            entropies[grad_idx] = entropy.item()
            act_losses[grad_idx] = actor_loss.item()
            crit_losses[grad_idx] = critic_loss.item()
            rw_hist[grad_idx, :] = target.cpu().numpy().reshape(1,-1)
            val_hist[grad_idx, :] = values.detach().cpu().numpy().reshape(1,-1)

            # Update model (gradient step)
            optimizer.zero_grad()
            loss.backward()
            # Clip gradient to prevent large updates
            grad_norm = compute_gradient_norm(model)
            grad_norms[grad_idx] = grad_norm
            if grad_idx >= 19:
                max_norm = np.quantile(grad_norms[max(0, grad_idx-299) : grad_idx+1], 0.95) # Max allowed norm will be 95% percentile of last 300 gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm= max_norm)
            optimizer.step()

        # Print status
        if not((episode+1)%10):
            roll_critic_loss = crit_losses[(episode-9)*grads_per_epi : episode*grads_per_epi+1].mean()
            roll_act_prob = action_avgs[(episode-9)*grads_per_epi*steps_per_grad : episode*grads_per_epi*steps_per_grad+1].mean()
            roll_act_std = action_stds[(episode-9)*grads_per_epi*steps_per_grad : episode*grads_per_epi*steps_per_grad+1].mean()
            roll_reward = raw_rw_avg[episode-9:episode+1].mean()
            print(f'Critic loss: {roll_critic_loss : .3g}. Average action prob: {roll_act_prob : .3g}. Std Dev action prob: {roll_act_std : .3g}. ' + \
                  f'Gamma: {gamma_ep : .4g}. Entropy weight: {entropy_w_ep : .3g}. Avg reward: {roll_reward : .5g}'
                 )
    print('Training complete. Saving the model and plotting...')
    
    # Save model weights to disk
    torch.save(model_base.state_dict(), f'a2c{model_suffix}.pth')

    # Plot training diagnostics
    plot_training_stats(act_losses, crit_losses, entropies, action_avgs, action_stds,
                        rw_hist, val_hist, grad_norms, raw_rw_avg, grads_per_epi,
                        critic_w, entropy_w, steps_per_grad, avg_rw, std_rw,
                        model_suffix, show_plots
                       )

    return (act_losses, crit_losses, entropies, action_avgs, action_stds, rw_hist, val_hist, grad_norms, raw_rw_avg)