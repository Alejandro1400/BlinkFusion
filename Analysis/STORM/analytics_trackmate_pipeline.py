import numpy
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from scipy.spatial import KDTree


def load_data(file_path):
    """ Load data from a CSV file into a DataFrame. """
    return pd.read_csv(file_path)

def merge_tracking_events(df):
    """ Merge tracking events and calculate weighted positions and frame extents. """
    # Calculate weighted positions
    df['weighted_x'] = df['POSITION_X'] * df['QUALITY']
    df['weighted_y'] = df['POSITION_Y'] * df['QUALITY']
    df['weighted_z'] = df['POSITION_Z'] * df['QUALITY']
    
    # Group by TRACK_ID
    grouped = df.groupby('TRACK_ID')
    result = grouped.apply(lambda g: pd.Series({
        'mean_x': g['weighted_x'].sum() / g['QUALITY'].sum(),
        'mean_y': g['weighted_y'].sum() / g['QUALITY'].sum(),
        'mean_z': g['weighted_z'].sum() / g['QUALITY'].sum(),
        'min_frame': g['FRAME'].min(),
        'max_frame': g['FRAME'].max(),
        'gaps': list(set(range(g['FRAME'].min(), g['FRAME'].max() + 1)) - set(g['FRAME']))
    })).reset_index()
    
    return result


def prepare_localizations(df):
    """ Prepare the localizations dataset with relevant columns. """
    return df[['ID', 'TRACK_ID', 'POSITION_X', 'POSITION_Y', 'POSITION_Z', 'QUALITY', 'MEAN_INTENSITY_CH1', 'SNR_CH1', 'FRAME']]


def calculate_xy_distance(track, loc):
    """ Calculate the Euclidean distance in the XY plane between a track and a localization. """
    return numpy.sqrt((track['mean_x'] - loc['POSITION_X'])**2 + (track['mean_y'] - loc['POSITION_Y'])**2)

import numpy as np


def calculate_distance(event1, event2):
    return np.sqrt((event1['mean_x'] - event2['mean_x'])**2 + 
                   (event1['mean_y'] - event2['mean_y'])**2 + 
                   (event1['mean_z'] - event2['mean_z'])**2)


def recursive_frame_check(track_index, frame, tracking_events, localizations, max_distance):
    """Recursively check and merge localizations at a given frame, adjust min/max frames and gaps if needed.
       Returns updates for applying after recursion completes."""
    updates = []
    track = tracking_events.iloc[track_index]
    frame_localizations = localizations[(localizations['FRAME'] == frame) & (localizations['TRACK_ID'].isnull())]
    
    min_distance = float('inf')
    selected_loc = None

    # Find the closest localization in this specific frame
    for j, loc in frame_localizations.iterrows():
        distance = calculate_xy_distance(track, loc)
        if distance <= max_distance and distance < min_distance:
            min_distance = distance
            selected_loc = j

    if selected_loc is not None:
        updates.append((frame, selected_loc, 'assigned', track['TRACK_ID']))

        # Determine if the frame is a boundary or gap and update accordingly
        if frame == track['min_frame'] - 1:
            new_updates = recursive_frame_check(track_index, frame - 1, tracking_events, localizations, max_distance)
            updates.extend(new_updates)
            updates.append((frame, selected_loc, 'min', track['TRACK_ID']))
        elif frame == track['max_frame'] + 1:
            new_updates = recursive_frame_check(track_index, frame + 1, tracking_events, localizations, max_distance)
            updates.extend(new_updates)
            updates.append((frame, selected_loc, 'max', track['TRACK_ID']))
        if frame in track['gaps']:
            updates.append((frame, selected_loc, 'gap', track['TRACK_ID']))

    return updates


def merge_non_tracking_events(localizations, tracking_events, max_distance=0.1):
    """ Identify and merge closest localizations and tracking events within a maximum distance, using recursion. """
    all_updates = []

    for i, track in tracking_events.iterrows():
        potential_frames = set(track['gaps'] + [track['min_frame'] - 1, track['max_frame'] + 1])
        
        for frame in potential_frames:
            updates = recursive_frame_check(i, frame, tracking_events, localizations, max_distance)
            all_updates.extend(updates)

    # Apply updates to localizations and adjust tracking_events
    for update in all_updates:
        frame, loc_index, update_type, track_id = update
        if update_type == 'assigned':
            localizations.at[loc_index, 'TRACK_ID'] = track_id
        if update_type in ['min', 'max']:
            if update_type == 'min':
                tracking_events.at[track_id, 'min_frame'] = frame
            elif update_type == 'max':
                tracking_events.at[track_id, 'max_frame'] = frame
        if update_type == 'gap':
            tracking_events.at[track_id, 'gaps'].remove(frame)

    return tracking_events, localizations

                
def merge_molecules(tracking_events, max_distance=0.1):
    """ Merge tracking events that are close to each other within a maximum distance. """
    # Create a KD-tree for tracking events
    tree = tracking_events[['mean_x', 'mean_y']].values
    kd_tree = KDTree(tree)

    # Add a column for molecule_id and initialize it
    tracking_events['molecule_id'] = -1
    molecule_id = 0

    # Iterate through each tracking event
    for i, track in tracking_events.iterrows():
        # Skip if the tracking event has already been assigned
        if track['molecule_id'] != -1:
            continue

        # Assign molecule ID to the current tracking event
        molecule_id += 1
        tracking_events.at[i, 'molecule_id'] = molecule_id

        frames_set = set(range(track['min_frame'], track['max_frame']+1))
        frames_set = frames_set - set(track['gaps'])
        # Query the KD-tree for tracking events within a maximum distance
        indices = kd_tree.query_ball_point([track['mean_x'], track['mean_y']], r=max_distance)

        
        for j in indices:
            # Skip if the tracking event has already been assigned
            if tracking_events.at[j, 'molecule_id'] != -1:
                continue

            # Obtain frames 
            new_frames_set = set(range(tracking_events.at[j, 'min_frame'], tracking_events.at[j, 'max_frame']+1))
            new_frames_set = frames_set - set(tracking_events.at[j, 'gaps'])

            # Check frame overlap 
            if len(frames_set.intersection(new_frames_set)) > 0:
                tracking_events.at[j, 'molecule_id'] = molecule_id
                frames_set.update(new_frames_set)

    return tracking_events

def assign_molecule_id(tracking_events, localizations):
    """ Assign molecule ID to localizations based on tracking events. """
    # Match tracking events id and track_id to assign molecule_id to localizations
    updated_localizations = localizations.merge(tracking_events[['TRACK_ID', 'molecule_id']], 
                                                left_on='TRACK_ID', 
                                                right_on='TRACK_ID', 
                                                how='left')

    return updated_localizations


def process_file(file_path):
    """ Process the entire file and merge tracking events. """
    df = load_data(file_path)
    tracking_events = merge_tracking_events(df)
    localizations = prepare_localizations(df)
    tracking_events, localizations = merge_non_tracking_events(localizations, tracking_events)
    tracking_events = merge_molecules(tracking_events)
    localizations = assign_molecule_id(tracking_events, localizations)
    return localizations

# Usage of the function. Use file dialog
file_path = tk.filedialog.askopenfilename()
localizations = process_file(file_path)
# Save as a CSV file
localizations.to_csv(file_path.replace('.csv', '_processed.csv'), index=False)
print(localizations.head())

