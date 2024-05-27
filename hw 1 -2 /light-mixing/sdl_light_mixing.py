from uuid import uuid4
import argparse, random, os
from self_driving_lab_demo import (
    SelfDrivingLabDemoLight,
    mqtt_observe_sensor_data,
    get_paho_client,
)
from self_driving_lab_demo.utils.search import (
    grid_search,
    random_search,
    ax_bayesian_optimization,
)
from plot import plot

import time
start = time.time()
# Parse arguments
parser = argparse.ArgumentParser(description='Run the self-driving lab demo')
parser.add_argument('--pico_id', required=True, help='The PICO ID of the device, or "test" for remote simulation')
parser.add_argument('--color', type=int, nargs=3, required=True, help='Target color as three integers "R G B"')
parser.add_argument('--save_dir', type=str, default='/', help='Directory to save results')
args = parser.parse_args()

# Set target color and random seed
R, G, B = args.color
print(f'Running on {args.pico_id} with target color R: {R}, G: {G}, B: {B}')

# Device settings
PICO_ID = args.pico_id
simulation = False
log_to_database = False
SESSION_ID = str(uuid4())
print(f'session ID: {SESSION_ID}')

# Number of search iterations
num_iter = 27

# Target color
target_inputs = {'R': R, 'G': G, 'B': B}

# SDL instance
sdl = SelfDrivingLabDemoLight(
    autoload=True,  # perform target data experiment automatically
    target_inputs=target_inputs, # if None, then defaults to random color using `target_seed` attribute
    simulation=simulation, # run simulation instead of physical experiment
    observe_sensor_data_fn=mqtt_observe_sensor_data,  # (default)
    observe_sensor_data_kwargs=dict(
        pico_id=PICO_ID, session_id=SESSION_ID, mongodb=log_to_database, 
        client=get_paho_client(f'sdl-demo/picow/{PICO_ID}/as7341/')
    ),
    target_seed=random.randint(0, 100000)
)

# Perform grid search, random search, and Bayesian optimization
# grid, grid_data = grid_search(sdl, num_iter)
# random_inputs, random_data = random_search(sdl, num_iter)
best_parameters, values, experiment, model = ax_bayesian_optimization(sdl, num_iter)

# Save plots
# save_dir = os.path.join(args.save_dir, f'R{R}_G{G}_B{B}')
# plot(save_dir, sdl, grid, grid_data, random_inputs, random_data, experiment)
time_elapsed = time.time() - start
print(f'Time elapsed: {time_elapsed} seconds')