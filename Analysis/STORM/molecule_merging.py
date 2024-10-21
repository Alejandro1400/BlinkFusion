import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from scipy.spatial import KDTree


def load_data(file_path):
    """ Load data from a CSV file into a DataFrame. """
    return pd.read_csv(file_path)

def create_tracking_events_df(df, file_type):
    """
    Calculate weighted positions for tracking events based on the specified weight column.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing tracking data for localizations.
    - file_type (str): Type of file which determines the weighting column ('thunderstorm' uses 'UNCERTAINTY [NM]', others use 'QUALITY').
    
    Returns:
    - pd.DataFrame: Summarized DataFrame of tracking events for weighted coordinates and statistical information.
    """
    weight_column = 'UNCERTAINTY [NM]' if file_type == 'thunderstorm' else 'QUALITY'

    if weight_column not in df.columns:
        df[weight_column] = 1

    df['weighted_x'] = df['X'] * df[weight_column]
    df['weighted_y'] = df['Y'] * df[weight_column]
    df['weighted_z'] = df['Z'] * df[weight_column] if 'Z' in df.columns else df[weight_column] 

    grouped = df.groupby('TRACK_ID')
    result = grouped.apply(lambda g: pd.Series({
        'mean_x': g['weighted_x'].sum() / g[weight_column].sum(),
        'mean_y': g['weighted_y'].sum() / g[weight_column].sum(),
        'mean_z': g['weighted_z'].sum() / g[weight_column].sum() if 'Z' in g.columns else 0,
        'min_frame': g['FRAME'].min(),
        'max_frame': g['FRAME'].max(),
        'gaps': list(set(range(g['FRAME'].min(), g['FRAME'].max() + 1)) - set(g['FRAME']))
    })).reset_index()

    # Remove the weighted columns
    df.drop(columns=['weighted_x', 'weighted_y', 'weighted_z'], inplace=True)

    return result


def extract_localizations_data(df):
    """
    Extract and prepare the localizations dataset with essential columns.
    
    Parameters:
    - df (pd.DataFrame): DataFrame from which to extract localization data.
    
    Returns:
    - pd.DataFrame: DataFrame with localization data.
    """
    if 'Z' not in df.columns:
        df['Z'] = 0

    return df[['ID', 'FRAME', 'TRACK_ID', 'X', 'Y', 'Z']]


def calculate_distance(track, localization):
    """
    Calculate the Euclidean distance in XYZ space between a track and a localization point.
    
    Parameters:
    - track (pd.Series): Data for the track.
    - localization (pd.Series): Data for the localization.
    
    Returns:
    - float: The calculated distance.
    """
    return np.sqrt((track['mean_x'] - localization['X'])**2 + 
                   (track['mean_y'] - localization['Y'])**2 + 
                   (track['mean_z'] - localization['Z'])**2)


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
        distance = calculate_distance(track, loc)
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
    """
    Merge tracking events with localizations based on proximity and update tracking details recursively.
    
    Parameters:
    - localizations (pd.DataFrame): DataFrame containing localization data.
    - tracking_events (pd.DataFrame): DataFrame containing tracking event data.
    - max_distance (float): Maximum distance to consider for merging events.
    
    Returns:
    - pd.DataFrame: Updated tracking events.
    - pd.DataFrame: Updated localizations.
    """
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
    """
    Cluster tracking events that are close to each other within a maximum distance.
    
    Parameters:
    - tracking_events (pd.DataFrame): DataFrame containing tracking event data.
    - max_distance (float): Maximum distance to consider for clustering.
    
    Returns:
    - pd.DataFrame: DataFrame with updated molecule IDs for each tracking event.
    """
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

def assign_molecule_ids(molecules, localizations):
    """
    Assign molecule IDs to localizations based on associated tracking events.
    
    Parameters:
    - tracking_events (pd.DataFrame): DataFrame with tracking events including molecule IDs.
    - localizations (pd.DataFrame): DataFrame of localizations to update with molecule IDs.
    
    Returns:
    - pd.DataFrame: Updated localizations with molecule IDs.
    """

    return localizations.merge(molecules[['TRACK_ID', 'molecule_id']], on='TRACK_ID', how='left')

        
def normalize_column_names(df, file_type):
    """
    Normalize column names based on the file type and convert to uppercase.
    
    Parameters:
    - df (pd.DataFrame): DataFrame whose columns are to be normalized.
    - file_type (str): Type of file ('trackmate' or 'thunderstorm') to determine the column mapping.
    
    Returns:
    - pd.DataFrame: DataFrame with normalized column names.
    """
    mappings = {
        'trackmate': {'ID': 'id', 'FRAME': 'frame', 'POSITION_X': 'x', 'POSITION_Y': 'y'},
        'thunderstorm': {'id': 'id', 'frame': 'frame', 'x [nm]': 'x', 'y [nm]': 'y'}
    }
    mapping = mappings.get(file_type.lower(), {})
    df.rename(columns=mapping, inplace=True)
    df.columns = df.columns.str.upper()

    return df


def process_file(file_path, file_type):
    """ Process the entire file and merge tracking events. """
    df = load_data(file_path)

    # Normalize column names
    raw_localizations = normalize_column_names(df, file_type)

    # Create tracking events DataFrame
    tracking_events = create_tracking_events_df(raw_localizations, file_type)

    # Extract localization data
    localizations = extract_localizations_data(raw_localizations)

    # Merge non-tracking events
    tracking_events, localizations = merge_non_tracking_events(localizations, tracking_events)

    # Merge molecules
    molecules = merge_molecules(tracking_events)

    # Assign molecule IDs to localizations
    localizations = assign_molecule_ids(molecules, raw_localizations)

    # Convert columns to uppercase
    localizations.columns = localizations.columns.str.upper()


    return localizations

# Usage of the function. Use file dialog
file_path = tk.filedialog.askopenfilename()
localizations = process_file(file_path, 'trackmate')
# Save as a CSV file
localizations.to_csv(file_path.replace('.csv', '_processed.csv'), index=False)
print(localizations.head())

