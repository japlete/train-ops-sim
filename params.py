from typing import Tuple

# Simulation parameters
first_crew_rotation_hr : int = 9 # Hour of the first crew rotation event in the simulation
crew_rotation_freq_hr : int = 8 # Hours between crew rotation events
crew_rotation_duration : float = 1/4 # Time it takes for the crew rotation (the trains stop) in hours

# Training parameters
num_episodes : int = 1500 # Number of episodes to simulate per parallel environment. Total simulated episodes = num_episodes * batch_size
grads_per_epi : int = 44 # Gradient steps per episode. After the final step, the environment is discarded and replaced with a new random starting point.
steps_per_grad : int = 5 # Number of environment steps for the N-step return (collect N discounted rewards for a single gradient step)
gamma : Tuple[float,float] = (0.95,0.998) # Initial and final discount factor for the rewards and future state values
lr : float = 0.0001 # Gradient method (RMSprop) learning rate
n_threads : int = -1 # Number of threads to use to take steps in parallel environments
replicas_per_thread : int = 32 # Number of environments per thread. Gradient batch_size = n_threads * replicas_per_thread
entropy_w : Tuple[float,float] = (0.1, 0.013) # Initial and final weight of the entropy loss. Higher values lead to more exploration.
critic_w : float = 1.0 # Weight of the critic loss
embedding_dim : int = 8 # Dimension of the embeddings used for train IDs and circuit IDs. Segment IDs use half this number.
hidden_dim : int = 64 # Dimension used in most hidden layer outputs. Some use half this number.
spline_knots : int = 10 # Number of knot points for the spline interpolation of the train position, as part of the feature extraction of the state representation.
spline_degree : int = 1 # Degree of the spline polynomials

# Prediction parameters
warmup_cycles : int = 0 # Cycles to discard when computing average cycle times. Doesn't apply to training.
random_seed : int = 0 # Seed for the train starting positions when doing a single run. During training, the episode number is the seed.

sim_params = (first_crew_rotation_hr, crew_rotation_freq_hr, crew_rotation_duration)
training_params = (num_episodes, grads_per_epi, steps_per_grad, gamma, lr, n_threads, replicas_per_thread, entropy_w, critic_w, (embedding_dim, hidden_dim), spline_knots, spline_degree)
pred_params = (warmup_cycles, random_seed, (embedding_dim, hidden_dim), spline_knots, spline_degree)