import os
import plotly.express as px
import plotly.io as pio
import pandas as pd


def plot(save_dir, sdl, grid, grid_data, random_inputs, 
         random_data, experiment, save_html=False):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Grid search
    grid_input_df = pd.DataFrame(grid)
    grid_output_df = pd.DataFrame(grid_data)[['frechet']]
    grid_df = pd.concat([grid_input_df, grid_output_df], axis=1)
    grid_df['best_so_far'] = grid_df['frechet'].cummin()

    # Random search
    random_input_df = pd.DataFrame(random_inputs, columns=['R', 'G', 'B'])
    random_output_df = pd.DataFrame(random_data)[['frechet']]
    random_df = pd.concat([random_input_df, random_output_df], axis=1)
    random_df['best_so_far'] = random_df['frechet'].cummin()

    # Bayesian optimization
    trials = list(experiment.trials.values())
    bayes_input_df = pd.DataFrame([t.arm.parameters for t in trials])
    bayes_output_df = pd.Series([t.objective_mean for t in trials], name='frechet').to_frame()
    bayes_df = pd.concat([bayes_input_df, bayes_output_df], axis=1)
    bayes_df['best_so_far'] = bayes_df['frechet'].cummin()

    # Save to CSV
    grid_df.to_csv(os.path.join(save_dir, 'grid_search.csv'), index=False)
    random_df.to_csv(os.path.join(save_dir, 'random_search.csv'), index=False)
    bayes_df.to_csv(os.path.join(save_dir, 'bayesian_optimization.csv'))

    # Iteration vs error plot
    grid_df['type'] = 'grid'
    random_df['type'] = 'random'
    bayes_df['type'] = 'bayesian'
    df = pd.concat([grid_df, random_df, bayes_df], axis=0)
    fig = px.line(df, x=df.index, y='best_so_far', color='type').update_layout(
        xaxis_title='Iteration',
        yaxis_title='Best error so far',
    )
    if save_html:
        pio.write_html(fig, os.path.join(save_dir, 'search_iterations.html'))
    pio.write_image(fig, os.path.join(save_dir, 'search_iterations.png'))

    # Observed search points plots
    fig = px.scatter_3d(grid_df, x='R', y='G', z='B', color='frechet', title='Grid')
    if save_html:
        pio.write_html(fig, os.path.join(save_dir, 'grid_search.html'))
    pio.write_image(fig, os.path.join(save_dir, 'grid_search.png'))
    fig = px.scatter_3d(random_df, x='R', y='G', z='B', color='frechet', title='Random')
    if save_html:
        pio.write_html(fig, os.path.join(save_dir, 'random_search.html'))
    pio.write_image(fig, os.path.join(save_dir, 'random_search.png'))
    fig = px.scatter_3d(bayes_df, x='R', y='G', z='B', color='frechet', title='Bayesian')
    if save_html:
        pio.write_html(fig, os.path.join(save_dir, 'bayesian_optimization.html'))
    pio.write_image(fig, os.path.join(save_dir, 'bayesian_optimization.png'))

    # Target plot
    target_inputs = sdl.get_target_inputs()
    true_inputs = pd.DataFrame(
        {key: target_inputs[key] for key in target_inputs}, index=[0]
    )
    true_inputs['type'] = 'true'
    best_grid_inputs = grid_df.iloc[grid_df['frechet'].idxmin()][['R', 'G', 'B', 'type']]
    best_random_inputs = random_df.iloc[random_df['frechet'].idxmin()][
        ['R', 'G', 'B', 'type']
    ]
    best_bayes_inputs = bayes_df.iloc[bayes_df['frechet'].idxmin()][['R', 'G', 'B', 'type']]
    best_df = pd.concat([best_grid_inputs, best_random_inputs, best_bayes_inputs], axis=1).T
    best_df['marker'] = 'observed'
    true_inputs['marker'] = 'target'
    best_df = pd.concat([best_df, true_inputs], axis=0)
    bnds = sdl.bounds
    fig = px.scatter_3d(
        best_df, x='R', y='G', z='B', color='type', symbol='marker', title='best'
    ).update_layout(
        scene=dict(
            xaxis=dict(
                nticks=4,
                range=[bnds['R'][0], bnds['R'][1]],
            ),
            yaxis=dict(
                nticks=4,
                range=[bnds['G'][0], bnds['G'][1]],
            ),
            zaxis=dict(
                nticks=4,
                range=[bnds['B'][0], bnds['B'][1]],
            ),
        ),
    )
    fig.update_traces(marker={'opacity': 0.75})
    fig.data[-1].marker.symbol = 'diamond-open'
    if save_html:
        pio.write_html(fig, os.path.join(save_dir, 'target.html'))
    pio.write_image(fig, os.path.join(save_dir, 'target.png'))
        