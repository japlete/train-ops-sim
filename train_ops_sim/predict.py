import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import SplineTransformer
from typing import Optional, Tuple
from time import perf_counter

from .input_preproc import preproc_sim_params
from .sim import TrainItinerarySim
from .predict_plots import SimResult
from .training_nn import ACTrainNN
from .training_utils import pre_proc_states

def sim_operation(filename : str, 
                  timespan_hours : int,
                  circuit_name : str,
                  model_state_filename : Optional[str] = None,
                  plot_suffix : str = '',
                  warmup_cycles : int = 0,
                  random_seed : int = 0,
                  NN_args: Tuple[int,int] = (8,32),
                  spline_knots : int = 10,
                  spline_degree : int = 1,
                  show_plots : bool = True,
                 ):
    '''
    Perform a single long simulation of the train operations, using the given actor-critic model if available, or a simple heuristic otherwise.
    The simulation is run for 100 times the train graph timespan parameter, to compute the cycle times with low error.
    The cycle times per circuit, segment occupation ratios and train movement graphs are plotted at the end.
    '''
    time_stats = {}
    # Read circuit spreadsheet
    acum_stats = True
    sim_end_tm = timespan_hours * 60.0 * 100.0
    input_tm = perf_counter()
    args, circuits, c_full, stations = preproc_sim_params(filename)
    time_stats['spreadsheet read'] = perf_counter() - input_tm
    
    # Run simulation
    sim_setup_tm = perf_counter()
    rng1 = np.random.Generator(np.random.SFC64(random_seed))
    sim = TrainItinerarySim(rng1, acum_stats, sim_end_tm, *args)
    state = sim.first_step()
    total_rw = 0.0
    step_count = 0
    if model_state_filename is None:
        # No model, use simple rule: first arrival has priority
        while(state[-1] == 0.0):
            rw, state = sim.step(True)
            total_rw -= rw
            if step_count == 0:
                normal_sim_st = perf_counter()
                time_stats['compilation'] = normal_sim_st - sim_setup_tm
            step_count += 1
    else:
        # Use loaded actor-critic model
        n_trains = sum(args[2])
        spline_t = SplineTransformer(n_knots= spline_knots, degree= spline_degree, extrapolation='constant')
        spline_t.fit(np.repeat(np.linspace(0,1,3).reshape(-1,1), n_trains, axis= 1))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_segments, ntr_per_circ, spline_dim, batch_size = len(args[4]), args[2], spline_t.n_features_out_ // n_trains, 1
        loaded_model = ACTrainNN(n_segments, ntr_per_circ, spline_dim, device, batch_size, *NN_args).to(device)
        loaded_model.load_state_dict(torch.load(model_state_filename, weights_only= True))
        loaded_model.eval()
        actions = []
        action_probs = []
        with torch.no_grad():
            while(state[-1] == 0.0):
                state_tensor = pre_proc_states(device, spline_t, state.reshape(1,-1))
                action_prob, _ = loaded_model(state_tensor)
                action = (action_prob > 0.5).item()
                actions.append(action)
                action_probs.append(action_prob.item())
                rw, state = sim.step(action)
                total_rw -= rw
                if step_count == 0:
                    normal_sim_st = perf_counter()
                    time_stats['compilation'] = normal_sim_st - sim_setup_tm
                step_count += 1
        #torch.save(loaded_model, 'full-'+model_state_filename) # Save full model if necessary

    # Show total reward (wait time) and if using a mode, plot the action probability histogram
    post_proc_st = perf_counter()
    time_stats['sim step'] = (post_proc_st - normal_sim_st) / (step_count - 1)
    print(f'Total wait time: {total_rw : .0f}')
    print(f'Average wait time: {total_rw/step_count : .5g}')
    if model_state_filename is not None:
        # Plot model action probability histogram
        print(f'Average action proba: {np.mean(actions) : .2g}', '. Actions=1: ', np.sum(actions), '. Total actions', len(actions))
        plt.figure(figsize=(10, 5))
        plt.hist(action_probs, bins= 30, density= True)
        plt.title('Action proba histogram')
        plt.xlabel('Probability (1= priority to first arrived train)')
        plt.ylabel('Density')
        plt.savefig(f'plots/inference_proba{plot_suffix}.svg', dpi=200, bbox_inches='tight')
        plt.savefig(f'plots/inference_proba{plot_suffix}.png', dpi=200, bbox_inches='tight')
        if show_plots:
            plt.show()
        else:
            plt.close('all')

    # Post-process statistics and plot
    op_stats = SimResult(sim, circuits, c_full, stations)
    op_stats.plot_cycle_tm(plot_suffix= plot_suffix, warmup_cycles= warmup_cycles, show_plots= show_plots)
    op_stats.plot_occupation(plot_suffix= plot_suffix, show_plots= show_plots)
    op_stats.plot_trains(circuit_name, timespan_hours, plot_suffix= plot_suffix, show_plots= show_plots)

    time_stats['post proc'] = perf_counter() - post_proc_st
    return op_stats, time_stats
