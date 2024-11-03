

import numpy as np
import pandas as pd


def euclidean_distance(p1, p2):
    # Calculate Euclidean distance between two points in 3D space
    return np.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2 + (p1['z'] - p2['z'])**2)


def merge_network(snakes, junctions):
    # Add a 'Network' column to the results DataFrame
    if 'Network' not in snakes.columns:
        snakes['Network'] = None
    if 'Junction' not in snakes.columns:
        snakes['Junction'] = None
    
    # Initialize a counter for network labels
    junction_label = 1
    network_label = 0
    actual_file = None
    
    # For each junction, find the closest points from every snake that is part of the same File
    for index, junction in junctions.iterrows():
        snakes_file = snakes[snakes['File'] == junction['File']]

        if actual_file != junction['File']:
            network_label = 0
            junction_label = 1
            actual_file = junction['File']

        # Initialize a list to track snakes that are part of this junction
        involved_snakes = []

        # Find the closest point from each snake to the junction
        for snake_id, group in snakes_file.groupby('Snake'):
            group['Distance'] = group.apply(lambda row: euclidean_distance(junction, row), axis=1)
            closest_point = group.loc[group['Distance'].idxmin()]
            
            if closest_point['Distance'] < 1:
                snakes.loc[(snakes['File'] == junction['File']) &
                           (snakes['Snake'] == snake_id) &
                           (snakes['Point'] == closest_point['Point']), 'Junction'] = junction_label
                involved_snakes.append(snake_id)

        # After identifying involved snakes, check if any of them already have a network label
        for snake_id in involved_snakes:
            snake_points = snakes[(snakes['File'] == junction['File']) & (snakes['Snake'] == snake_id)]
            if snake_points['Network'].notna().any():
                # Use the first existing network label found
                network_label = snake_points['Network'].dropna().iloc[0]
                break
        else:
            # If no existing network label, increment and use a new one
            network_label += 1

        # Assign the determined network label to all involved snake points
        for snake_id in involved_snakes:
            snakes.loc[(snakes['File'] == junction['File']) & (snakes['Snake'] == snake_id), 'Network'] = network_label

        # Increment the junction label for the next junction
        junction_label += 1


    return snakes


def calculate_metrics(snakes):
    # Ensure all required columns are present or create them
    if 'Sinuosity' not in snakes.columns:
        snakes['Sinuosity'] = np.nan
    if 'Length' not in snakes.columns:
        snakes['Length'] = np.nan
    if 'SNR' not in snakes.columns:
        snakes['SNR'] = np.nan
    if 'Gaps' not in snakes.columns:
        snakes['Gaps'] = 0
    
    # Group by 'File' and 'Snake' since IDs may repeat across files
    grouped = snakes.groupby(['File', 'Snake'])
    
    # Iterate over each group to calculate metrics
    for (file_id, snake_id), group in grouped:
        # Calculate the straight-line distance between the first and the last point of the snake
        first_point = group.iloc[0]
        last_point = group.iloc[-1]
        straight_distance = euclidean_distance(first_point, last_point)
        
        # Calculate the sum of the distances between consecutive points and sinuosity
        snake_distance = 0
        for i in range(len(group) - 1):
            snake_distance += euclidean_distance(group.iloc[i], group.iloc[i + 1])
        
        # Handle division by zero if straight distance is zero
        sinuosity = snake_distance / straight_distance if straight_distance > 0 else np.nan
        
        # Assign length and sinuosity to the group
        snakes.loc[group.index, 'Length'] = snake_distance
        snakes.loc[group.index, 'Sinuosity'] = sinuosity
        
        # Calculate SNR
        mean_intensity = group['fg_int'].mean()
        std_background = group['bg_int'].std()
        snr = mean_intensity / std_background if std_background > 0 else np.nan
        snakes.loc[group.index, 'SNR'] = snr
        
        # Count gaps
        gaps = 0
        in_gap = False
        for i, row in group.iterrows():
            if row['fg_int'] < row['bg_int']:
                if not in_gap:
                    in_gap = True
                    gaps += 1
            else:
                in_gap = False
        
        # Assign gap count to the group
        snakes.loc[group.index, 'Gaps'] = gaps
    
    return snakes


def fix_soac_df(snakes):
    # Rename columns to match the SOAC API output
    column_rename_map = {
        'fg_int': 'Intensity',
        'bg_int': 'Background'
    }
    snakes.rename(columns=column_rename_map, inplace=True)
    
    # Define the new order of the columns
    new_column_order = [
        'File', 'Network', 'Snake', 'Point', 'Junction',
        'x', 'y', 'z', 'Length', 'Sinuosity', 'Intensity', 
        'Background', 'SNR', 'Gaps'
    ]
    
    # Reorder the columns in the DataFrame
    # This assumes that all the columns are already present in the DataFrame.
    # If some columns are missing, you would need to add them with default values.
    snakes = snakes[new_column_order]
    
    return snakes


def soac_analytics_pipeline(snakes, junctions):
    # Merge the network and calculate length and sinuosity
    snakes = merge_network(snakes, junctions)
    snakes = calculate_metrics(snakes)
    
    # Change column names
    snakes = fix_soac_df(snakes)
    
    return snakes


def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.
    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)  # Fast and numerically precise
    return (average, np.sqrt(variance))

def obtain_cell_metrics(snakes):
    # Group by both 'File' and 'Snake', then apply the metrics calculation
    result = snakes.groupby(['File', 'Snake']).apply(calculate_snake_metrics)

    # Preparing to calculate weighted averages and std deviation
    weighted_metrics = {}
    metrics = ['Intensity', 'SNR', 'Contrast', 'Continuity', 'Sinuosity']
    
    for metric in metrics:
        weighted_avg, weighted_std = weighted_avg_and_std(
            result[metric].values, 
            result['Length'].values
        )
        weighted_metrics[f'{metric} (mean)'] = weighted_avg
        weighted_metrics[f'{metric} (std)'] = weighted_std

    # Calculate simple mean and std for Length and Junctions as these might not need weighting
    weighted_metrics['Length (mean)'] = result['Length'].mean()
    weighted_metrics['Length (std)'] = result['Length'].std()
    weighted_metrics['Junctions (mean)'] = result['Junctions'].mean()
    weighted_metrics['Junctions (std)'] = result['Junctions'].std()

    # Put in order so that the means are first and the stds are second
    weighted_metrics = {k: weighted_metrics[k] for k in weighted_metrics if 'mean' in k} | {k: weighted_metrics[k] for k in weighted_metrics if 'std' in k}

    # Creating DataFrame
    overall_metrics = pd.DataFrame(weighted_metrics, index=[0])

    return overall_metrics


def calculate_snake_metrics(group):
    std_background = group['Background'].std()
    mean_background = group['Background'].mean()
    mean_intensity = group['Intensity'].mean()

    # Calculate SNR for each row within the group
    SigNr = (mean_intensity - mean_background) / std_background if std_background > 0 else np.nan

    # Define individual metrics
    metrics = {
        'Sinuosity': group['Sinuosity'].mean(),
        # Calculate how many Junction different values are present
        'Junctions': group['Junction'].nunique(),
        'Length': group['Length'].mean(),
        'SNR': SigNr,
        'Intensity': group['Intensity'].mean(),
        'Contrast': group['Intensity'].mean() / group['Background'].mean(),
        'Continuity': ((group['Intensity'] > group['Background']).sum() / len(group['Intensity'])) * 100  # As a percentage
    }

    return pd.Series(metrics)