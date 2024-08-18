import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from .sim import TrainItinerarySim
from .sim_objs import idx_tp

class SimResult:
    '''
    Class to store the results of a single simulation, compute statistics (cycle times per circuit, segment occupation ratios), 
    train movement graphs and plot them.
    The constructor also receives the table of stations, circuit summaries and circuit itineraries (c_full).
    These inputs come from the input_preproc module, which reads the spreadsheet.
    The simulation is received after already performing the required number of steps.
    '''
    def __init__(self, sim : TrainItinerarySim
                 , circuits : pd.DataFrame
                 , c_full : pd.DataFrame
                 , stations : pd.DataFrame):
        self.sim = sim
        self.circuits = circuits
        self.c_full = c_full
        self.stations = stations.reset_index().set_index('st_id')
        self.stations.loc[-1] = -1 # Add dummy station with ID -1 to easily signal cycle end for open circuits
        
        # Extract arrival and departure times
        arrival_stats = sim.arrival_stats.copy()
        for cid, c in enumerate(sim.circuits):
            for tid in c.train_ids:
                arrival_stats.append((idx_tp(cid), tid, sim.trains[tid].est_arrival_tm[:sim.trains[tid].next_arrival_idx]))
        dep_stats = sim.departure_stats.copy()
        for cid, c in enumerate(sim.circuits):
            for tid in c.train_ids:
                arrival_stats.append((idx_tp(cid), tid, sim.trains[tid].est_depart_tm[:sim.trains[tid].next_depart_idx]))
        arr_times = (
            pd.DataFrame(arrival_stats, columns= ['circuit_id','train_id','time'])
            .explode('time')
            .reset_index().rename(columns= {'index' : 'global_cycle_idx'})
        )
        arr_times['circuit_segm_idx'] = arr_times.groupby(['global_cycle_idx','circuit_id','train_id']).cumcount()
        dep_times = (
            pd.DataFrame(dep_stats, columns= ['circuit_id','train_id','time'])
            .explode('time')
            .reset_index().rename(columns= {'index' : 'global_cycle_idx'})
        )
        dep_times['circuit_segm_idx'] = dep_times.groupby(['global_cycle_idx','circuit_id','train_id']).cumcount()
        
        # Concatenate arrival and departure times and join with station location information
        times = (
            pd.concat([
                arr_times, dep_times
            ])
            .join(c_full.set_index('circuit_segm_idx', append= True)
                          .drop(columns= ['priority_operator','double_track','acum_exit_tm','acum_entry_tm','waiter_switch_tm','passer_switch_tm','cooldown_tm'])
                  , on= ['circuit_id','circuit_segm_idx'])
        )
        graph_full = (
            times.set_index(['circuit_id','train_id'])
            .drop(columns= ['global_cycle_idx','circuit_segm_idx','origin_st_name','dest_st_name','dest_st_id','orig_st_id','segm_id'])
            .set_index('time', append= True).sort_index()
        )
        graph_full = graph_full[~graph_full.index.duplicated(keep='first')].unstack([0,1])
        
        # Extract spawn times for the open circuits and append to the arrival and departure times
        # (when there is a spawn in an open circuit, the train isn't departing or arriving, so its position is registered in a separate array)
        spawn_tms = (
            pd.DataFrame(sim.snapshot_stats, columns= ['circuit_id','train_id','st_id','time'])
            .join(self.stations.drop(columns= ['waiter_switch_tm','passer_switch_tm','st_cap','st_name']), on= 'st_id')
            .drop(columns= 'st_id')
            .set_index(['circuit_id','train_id','time']).sort_index()
        )
        spawn_tms = spawn_tms[~spawn_tms.index.duplicated(keep='first')].unstack([0,1]).sort_index(axis= 1)
        
        # Extract the start and end time of each stop during travel (due to a crew rotation) and append to the arrival and departure times
        stop_info = pd.DataFrame(self.sim.graph_stats, columns= ['circuit_id','train_id','origin_st_id','dest_st_id','progress','time'])
        st_pos_only = self.stations.drop(columns= ['waiter_switch_tm','passer_switch_tm','st_cap','st_name'])
        origin_pos = stop_info.join(st_pos_only, on= 'origin_st_id').drop(columns= stop_info.columns)
        dest_pos = stop_info.join(st_pos_only, on= 'dest_st_id').drop(columns= stop_info.columns)
        stop_locs = origin_pos + (dest_pos - origin_pos) * stop_info.progress.to_numpy().reshape(-1,1)
        stop_locs.index = pd.MultiIndex.from_frame(stop_info[['circuit_id','train_id','time']])
        stop_locs = stop_locs.unstack([0,1]).sort_index(axis= 1)

        # Remove the dummy station
        self.stations.drop(index= -1, inplace= True)

        # The final graph_full has a time index and columns with 3 levels (circuit ID, train ID, graph section).
        # The values in the table are the position of each train in each graph section at the given time (so it contains a lot of null values).
        self.graph_full = pd.concat([graph_full, spawn_tms, stop_locs]).sort_index()

    def plot_trains(self, circuit_name : str, timespan_hours : int, plot_suffix : str = '', show_plots : bool = True):
        '''
        Plot the train movement graph for each track section, highlighting with more identifiable colors the trains from the circuit in 'circuit_name'.
        The total graph width is timespan_hours.
        '''
        # Plot pre-processing
        assert timespan_hours > 0, 'Timespan must be greater than zero'
        assert circuit_name in self.circuits.circuit_name.values, f"Circuit name {circuit_name} doesn't match known names"
        
        sorted_secs = self.graph_full.count().groupby(level= 0).sum().sort_values(ascending= False).index
        sorted_trains = self.graph_full.columns.droplevel(0).unique().sort_values()
        max_tm = self.graph_full.T.stack().reset_index('time').time.groupby(['circuit_id','train_id']).max().min() # Max time is the minimum last reported position of all trains
        graph_length = timespan_hours*60
        min_tm = max_tm - graph_length
        min_tm = (min_tm // (24*60)) * (24*60) # Round to start the graph always at 00:00
        max_tm = min_tm + graph_length
        min_draw_tm = self.graph_full[self.graph_full.index < min_tm].T.stack().reset_index('time').time.groupby(['circuit_id','train_id']).max().min()
        graph_selec = self.graph_full.loc[min_draw_tm:].copy()
        x_range = (min_tm, max_tm)
        
        cid = self.circuits.index[self.circuits.circuit_name == circuit_name].to_numpy().squeeze()
        is_circuit = [t[0] == cid for t in sorted_trains]
        n_in_circuit = sum(is_circuit)
        n_oth = len(self.circuits) - 1
        circuit_colors , oth_colors = sns.color_palette("Set1", n_in_circuit) , sns.color_palette("blend:#5A9,#253,#888", n_oth)
        
        # Iterate over each track section
        for l in sorted_secs:
            st_tab = self.stations.set_index('st_name')[l].dropna().sort_values()
            fig_h = max(2, 3.2*np.log2(st_tab.count())-6)
            fig, ax = plt.subplots(figsize=(20, fig_h))
        
            # Plot 6h vertical marks
            y_range = (st_tab.values.min(), st_tab.values.max())
            xlabels = ['']
            for t in np.arange(0.0, x_range[1]-x_range[0]+1, 60.0):
                if not (int(t) % 1440) and t > 0.0:
                    xlabels.append(f'{int(t/1440)}D')
                    lw, c = 1.0, 'black'
                elif not ((int(t) % 1440) % 240) and t > 0.0:
                    xlabels.append(f'{int((int(t) % 1440)/60)}h')
                    lw, c = 0.75, (0.25,0.25,0.25)
                else:
                    lw, c = 0.5, (0.5,0.5,0.5)
                plt.vlines(x_range[0] + t, y_range[0], y_range[1], color= c, linestyles= '-', linewidth= lw)

            # Plot grey background on crew rotation periods
            crew_rot_tms = [t for t in np.arange(self.sim.crew_rot_first_tm, x_range[1], self.sim.crew_rot_freq_tm) if t >= x_range[0]]
            for cr_start in crew_rot_tms:
                ax.axvspan(cr_start, cr_start + self.sim.crew_rot_tm, facecolor='grey', alpha=0.2)
            
            # Plot horizontal lines at station locations
            for _, st_loc in st_tab.items():
                plt.hlines(st_loc, x_range[0], x_range[1], color= 'black', linestyles= '--', linewidth= 0.5)
            
            # Plot trains, selecting the color based on the circuit ID
            y_range = (y_range[0] - (y_range[1] - y_range[0])*0.005, y_range[1] + (y_range[1] - y_range[0])*0.005)
            circuit_added = 0
            legend_labels = []
            legend_mask = np.zeros(len(sorted_trains), dtype= bool)
            used_circuits = np.zeros(n_oth+1, dtype= bool)
            for i,c in enumerate(sorted_trains):
                if c[0] == cid:
                    color = circuit_colors[circuit_added]
                    circuit_added += 1
                    legend_labels.append(f'T{circuit_added}')
                    legend_mask[i] = True
                else:
                    color = oth_colors[c[0] - (0 if c[0] < cid else 1)]
                    if not used_circuits[c[0]]:
                        legend_labels.append(self.circuits.circuit_name.loc[c[0]])
                        legend_mask[i] = True
                        used_circuits[c[0]] = True
                (
                    graph_selec.loc[:,pd.IndexSlice[:,*c]]
                    .dropna(how= 'all')[l]
                    .sort_index()
                    .replace(-1, np.nan) # For open circuits, replace the dummy station location (cycle end) with NaN, to force a line break
                    .plot(ax=ax, xlim=x_range, ylim=y_range, linewidth= 2.0, color= color)
                )
            
            # Formatting
            plt.title(l)
            plt.xlabel('')
            leg_handles, _ = ax.get_legend_handles_labels()
            leg_handles = [lh for i, lh in enumerate(leg_handles) if legend_mask[i]]
            ncol = int(np.ceil(len(legend_labels) / 2))
            ax.legend(leg_handles, legend_labels, loc='lower center', ncol= ncol, bbox_to_anchor=(0.5, -1.5/fig_h))
            ax.set_yticks(st_tab.values)
            ax.set_yticklabels(st_tab.index, fontsize=7, fontstyle='italic')
            h4_ticks = np.arange(x_range[0], x_range[1]+1, 240.0)
            ax.set_xticks(h4_ticks)
            ax.set_xticklabels(xlabels, fontsize=10)
            # Add top x-axis
            ax_top = ax.twiny()
            ax_top.set_xlim(ax.get_xlim())
            ax_top.set_xticks(h4_ticks)
            ax_top.set_xticklabels(xlabels, fontsize=10)
            # Save plots
            if not os.path.exists('plots'):
                os.makedirs('plots')
            plt.savefig(f'plots/{l}{plot_suffix}.svg', dpi=200, bbox_inches='tight')
            plt.savefig(f'plots/{l}{plot_suffix}.png', dpi=200, bbox_inches='tight')
            if show_plots:
                plt.show()
            else:
                plt.close('all')

    def plot_cycle_tm(self, warmup_cycles : int = 0, plot_suffix : str = '', show_plots : bool = True):
        '''
        Plot the observed and minimal theoretical average cycle time per circuit in a bar plot.
        The minimal time is the sum of all mandatory travel times and processes at stations, adjusted by crew rotation estimates.
        Also, plots the cumulative average cycle time per circuit, with the warmup_cycles removed.
        '''
        # Gather cycle times from the train stats
        tr_cycle_l = []
        for i,tr in enumerate(self.sim.trains):
            for cs in tr.cycle_stats:
                if not np.isnan(cs[1]) and not np.isnan(cs[0]):
                    tr_cycle_l.append((tr.circuit_id, i, *cs))
        
        # Compute average time per circuit and cycle number (averaging across trains)
        cycle_tms = pd.DataFrame(tr_cycle_l, columns= ['circuit_id', 'train_id', 'time', 'duration']).sort_values('time').reset_index(drop=True)
        cycle_tms['Train cycle'] = cycle_tms.groupby(['circuit_id','train_id']).cumcount()
        self.cycle_tms = cycle_tms.groupby(['circuit_id', 'Train cycle']).duration.mean().unstack('circuit_id').sort_index(axis= 1).iloc[warmup_cycles:]
        self.cycle_tms.columns = self.circuits.sort_index().circuit_name.rename('Circuit')

        # Reverse the cycles to get the mean starting from the last (to see in the graph if the first cycles affect the total average)
        self.cycle_tms_cummean_inv = self.cycle_tms.iloc[::-1].expanding().mean().iloc[::-1] / 60.0
        self.cycle_tms_cummean_inv.plot(figsize= (15, 5))
        plt.ylabel('Hours')
        plt.title('Inverse cumulative average cycle time')
        
        # Save plots
        if not os.path.exists('plots'):
            os.makedirs('plots')
        plt.savefig(f'plots/inv_cum_avg_cycle_tm{plot_suffix}.svg', dpi= 200, bbox_inches='tight')
        plt.savefig(f'plots/inv_cum_avg_cycle_tm{plot_suffix}.png', dpi= 200, bbox_inches='tight')
        if show_plots:
            plt.show()
        else:
            plt.close('all')

        # Compare simulated cycle times vs base cycle times
        cycle_tm_comp = self.circuits.join(self.cycle_tms_cummean_inv.mean().rename('full_cycle_tm') * 60.0, on= 'circuit_name')
        cycle_tm_comp['base_cycle_tm'] = cycle_tm_comp.cycle_tm * (1.0 + self.sim.crew_rot_tm / self.sim.crew_rot_freq_tm)
        cycle_tm_comp['wait_tm'] = cycle_tm_comp.full_cycle_tm - cycle_tm_comp.base_cycle_tm
        cycle_tm_comp['wait_tm_prop'] = cycle_tm_comp.wait_tm / cycle_tm_comp.full_cycle_tm
        cycle_tm_comp.set_index('circuit_name', inplace= True)
        cycle_tm_comp = (
            cycle_tm_comp.rename(columns= {'base_cycle_tm' : 'base cycle time', 
                                           'full_cycle_tm' : 'simulated cycle time'})
        )
        # Plotting
        ax = (
            cycle_tm_comp[['base cycle time','simulated cycle time']]
            .map(lambda x : x / 60.0)
            .plot(kind= 'bar', width= 0.7, color=['#8b0000','#3e60a7'], figsize=(min(20, max(6, len(cycle_tm_comp))), 4))
        )
        # Formatting
        ax.set_yticks([])
        plt.title('Circuits cycle time')
        plt.ylabel('Hours')
        plt.xlabel('')
        plt.xticks(rotation=45, ha='right')
        y_delta = plt.ylim()
        y_delta = (y_delta[1] - y_delta[0]) * 0.015
        for i, v in enumerate(cycle_tm_comp['base cycle time']/60.0):
            ax.text(i-0.7/3, v+y_delta, str(np.round(v,1)), ha='center', fontsize=10)
        for i, v in enumerate(cycle_tm_comp['simulated cycle time']/60.0):
            ax.text(i+0.7/3, v+y_delta, str(np.round(v,1)), ha='center', fontsize=10)
        sns.despine(ax=ax, top=True, right=True, left=True, bottom=True)
        plt.tick_params(axis='x', which='both', tick1On=False, tick2On=False)
        
        # Save plots
        plt.savefig(f'plots/base_vs_sim_cycle{plot_suffix}.svg', dpi= 200, bbox_inches='tight')
        plt.savefig(f'plots/base_vs_sim_cycle{plot_suffix}.png', dpi= 200, bbox_inches='tight')
        if show_plots:
            plt.show()
        else:
            plt.close('all')

    def plot_occupation(self, plot_suffix : str = '', show_plots : bool = True):
        '''
        Bar plot of the segment occupation ratios (percentage of time that each segment is occupied).
        Only the top 20 segments are plotted.
        '''
        # Compute segment utilizations from simulation statistics
        seg_util_l = []
        for i,s in enumerate(self.sim.segments):
            for t in s.util_stats:
                if t[1] != -1.0:
                    seg_util_l.append((i, *t))
                else:
                    seg_util_l.append((i, t[0], self.sim.sim_tm - t[0]))
        self.seg_util = pd.DataFrame(seg_util_l, columns= ['segm_id','time','duration'])
        self.seg_util = (
            self.c_full[['origin_st_name','dest_st_name','segm_id']]
            .drop_duplicates(subset= ['segm_id'])
            .set_index('segm_id')
            .join(
                (self.seg_util.groupby('segm_id').duration.sum() / self.sim.sim_tm)
                .rename('Segment Utilization')
            )
            .sort_values('Segment Utilization', ascending= False)
        )
        self.seg_util['segm_name'] = self.seg_util.origin_st_name + ' - ' + self.seg_util.dest_st_name

        # Plotting
        ax = (
            (self.seg_util.set_index('segm_name')['Segment Utilization']*100.0)
            .iloc[:20]
            .plot(kind= 'bar', width= 0.7, figsize=(7, 4))
        )
        
        # Formatting
        ax.set_yticks([])
        plt.title('Segment occupation time')
        plt.ylabel('% utilization')
        plt.xlabel('')
        y_delta = plt.ylim()
        y_delta = (y_delta[1] - y_delta[0]) * 0.015
        for i, v in enumerate(self.seg_util['Segment Utilization'].iloc[:20]*100.0):
            ax.text(i, v+y_delta, str(int(np.round(v,0))), ha='center', fontsize=10)
        sns.despine(ax=ax, top=True, right=True, left=True, bottom=True)
        plt.tick_params(axis='x', which='both', tick1On=False, tick2On=False)
        # Save plots
        if not os.path.exists('plots'):
            os.makedirs('plots')
        plt.savefig(f'plots/segment_util{plot_suffix}.svg', dpi= 200, bbox_inches='tight')
        plt.savefig(f'plots/segment_util{plot_suffix}.png', dpi= 200, bbox_inches='tight')
        if show_plots:
            plt.show()
        else:
            plt.close('all')
