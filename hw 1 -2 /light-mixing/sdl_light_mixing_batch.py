from uuid import uuid4 
import pandas as pd
import argparse, random, os
from self_driving_lab_demo import (
    SelfDrivingLabDemoLight,
    mqtt_observe_sensor_data,
)
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
import ray
import plotly.express as px
import plotly.io as pio


# Parse arguments
parser = argparse.ArgumentParser(description='Run the self-driving lab demo')
parser.add_argument('--pico_id', nargs='+', required=True, help='The PICO ID of the devices, can be "test" for remote simulation')
parser.add_argument('--color', type=int, nargs=3, required=True, help='Target color as three integers "R G B"')
parser.add_argument('--num_iter', type=int, default=27, help='Number of search iterations')
parser.add_argument('--save_dir', type=str, default='/', help='Directory to save results')
args = parser.parse_args()

# Set target color and random seed
R, G, B = args.color
print(f'Running on {args.pico_id} with target color R: {R}, G: {G}, B: {B}')

# Device settings
N = len(args.pico_id)
PICO_IDS = [pico_id for pico_id in args.pico_id]
simulation = False
log_to_database = False
SESSION_IDS = [str(uuid4()) for _ in range(N)]
print(f'session IDs: {SESSION_IDS}')

# Number of search iterations
num_iter = args.num_iter

# Target color
target_inputs = {'R': R, 'G': G, 'B': B}

# SDL instances
sdl_batch = [
    SelfDrivingLabDemoLight(
        autoload=True,  # perform target data experiment automatically
        target_inputs=target_inputs, # if None, then defaults to random color using `target_seed` attribute
        simulation=simulation, # run simulation instead of physical experiment
        observe_sensor_data_fn=mqtt_observe_sensor_data,  # (default)
        observe_sensor_data_kwargs=dict(
            pico_id=pico_id, session_id=session_id, mongodb=log_to_database
        ),
        target_seed=random.randint(0, 100000)
    )
    for pico_id, session_id in zip(PICO_IDS, SESSION_IDS)
]

# Get parameters for search space
sdl = sdl_batch[0]
bounds = dict(R=sdl.bounds['R'], G=sdl.bounds['G'], B=sdl.bounds['B'])
params = [dict(name=nm, type='range', bounds=bnd) for nm, bnd in bounds.items()]
batch_size = N
objectives = {'frechet': ObjectiveProperties(minimize=True)}

# Initialize Ray for parallel evaluation
ray.shutdown()
ray.init(num_cpus=batch_size)

# Define the evaluation function
@ray.remote
def evaluate(parameters, sdl_num):
    results = sdl_batch[sdl_num].evaluate(parameters)
    return {'frechet': (results['frechet'], 0.0)}

# Create the Ax client
ax_client = AxClient()
ax_client.create_experiment(
    name='sdl-demo-batch',
    parameters=params,
    objectives=objectives,
    overwrite_existing_experiment=True,
)

# Store results
R_trials = []
G_trials = []
B_trials = []
frechet_trials = []
best_so_far_trials = []

for i in range(num_iter // 3):
    trials_dict, optimization_complete = ax_client.get_next_trials(
        max_trials=batch_size
    )
    # Start running the batch of experiments in parallel
    futures = [
        evaluate.remote(parameters, sdl_num)
        for sdl_num, parameters in enumerate(trials_dict.values())
    ]
    # Block until the batch of experiments finishes running
    results = ray.get(futures)
    # Add results to the optimizer
    for trial_index, raw_data in zip(trials_dict.keys(), results):
        R_trials.append(trials_dict[trial_index]['R'])
        G_trials.append(trials_dict[trial_index]['G'])
        B_trials.append(trials_dict[trial_index]['B'])
        frechet_trials.append(raw_data['frechet'][0])
        best_so_far_trials.append(min(frechet_trials))
        ax_client.complete_trial(trial_index=trial_index, raw_data=raw_data)

# Save results
batch_df = pd.DataFrame({
        'R': R_trials,
        'G': G_trials,
        'B': B_trials,
        'frechet': frechet_trials,
        'best_so_far': best_so_far_trials
    }
)

# Plot and save
save_dir = os.path.join(args.save_dir, f'R{R}_G{G}_B{B}')
os.makedirs(save_dir, exist_ok=True)
batch_df.to_csv(os.path.join(save_dir, 'batch_optimization.csv'), index=False)
fig = px.scatter_3d(batch_df, x='R', y='G', z='B', color='frechet', title='Grid')
save_html = False
if save_html:
    pio.write_html(fig, os.path.join(save_dir, 'batch_optimization.html'))
pio.write_image(fig, os.path.join(save_dir, 'batch_optimization.png'))
