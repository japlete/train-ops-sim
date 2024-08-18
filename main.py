# INSTRUCTIONS: fill the commented parameters in the params.py file, then the commented parameters in this file.
# This script will run both training and prediction. For running only predictions, use the main.ipynb notebook.
# This script is meant for console (not Jupyter) use, so graphs are only saved to disk, in the training_plots and plots directories, but not rendered.

filename = 'Input example.xlsx' # Spreadsheet name with the circuit specification
model_suffix = '-example' # Will be added to the name of files written to disk, to track different scenarios

from params import training_params
from train_ops_sim.training_loop import train_a2c
_ = train_a2c(filename, *(model_suffix, *training_params, False))

print('Running prediction')

timespan_hours = 48 # Width of the train graphs (total simulation time is x100 this number to get precise cycle time estimates)
circuit_name = 'Circuit A' # Name of the main circuit (to highlight the train colors in the graph)
model_state_filename = 'a2c'+model_suffix+'.pth' # Filename with the saved model, as written automatically by the training process
plot_suffix = '-example' # Suffix for the plot filenames that will be written to disk, to keep track of different scenarios

from params import pred_params
from train_ops_sim.predict import sim_operation
_, ts = sim_operation(filename, timespan_hours, circuit_name, model_state_filename, plot_suffix, *(*pred_params, False))

print(ts)
print('Done')
