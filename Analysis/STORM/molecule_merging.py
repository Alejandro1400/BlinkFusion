import time
import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from Analysis.STORM.track_storm import merge_localizations

def create_tracking_events(locs, file_type):
    """
    Calculate weighted positions for tracking events based on the specified weight column.
    
    Parameters:
    - locs (pd.DataFrame): DataFrame containing tracking data for localizations.
    - file_type (str): Type of file which determines the weighting column ('thunderstorm' uses 'UNCERTAINTY [NM]', others use 'QUALITY').
    
    Returns:
    - pd.DataFrame: Summarized DataFrame of tracking events for weighted coordinates and statistical information.
    """
    weight_column = 'UNCERTAINTY' if file_type == 'thunderstorm' else 'QUALITY'

    if weight_column not in locs.columns:
        locs[weight_column] = 1

    # Filtrar el DataFrame para excluir los registros donde 'TRACK_ID' es 0
    filtered_locs = locs[locs['TRACK_ID'] != 0]

    filtered_locs['weighted_x'] = filtered_locs['X'] * filtered_locs[weight_column]
    filtered_locs['weighted_y'] = filtered_locs['Y'] * filtered_locs[weight_column]
    filtered_locs['weighted_z'] = filtered_locs['Z'] * filtered_locs[weight_column]

    # Agrupar por 'TRACK_ID' después de excluir los 0s
    grouped = filtered_locs.groupby('TRACK_ID')
    result = grouped.apply(lambda g: pd.Series({
        'X': g['weighted_x'].sum() / g[weight_column].sum(),
        'Y': g['weighted_y'].sum() / g[weight_column].sum(),
        'Z': g['weighted_z'].sum() / g[weight_column].sum(),
        'QUALITY': g['QUALITY'].sum(),
        'START_FRAME': g['FRAME'].min(),
        'END_FRAME': g['FRAME'].max(),
        'GAPS': list(set(range(g['FRAME'].min(), g['FRAME'].max() + 1)) - set(g['FRAME']))
    })).reset_index()

    return result


def calculate_distance(track, localization):
    """
    Calculate the Euclidean distance in XYZ space between a track and a localization point.
    
    Parameters:
    - track (pd.Series): Data for the track.
    - localization (pd.Series): Data for the localization.
    
    Returns:
    - float: The calculated distance.
    """
    return np.sqrt((track['X'] - localization['X'])**2 + 
                   (track['Y'] - localization['Y'])**2 + 
                   (track['Z'] - localization['Z'])**2)

                
def merge_tracking_events(tracking_events, max_distance=100):
    """
    Cluster tracking events that are close to each other within a maximum distance.
    
    Parameters:
    - tracking_events (pd.DataFrame): DataFrame containing tracking event data.
    - max_distance (float): Maximum distance to consider for clustering.
    
    Returns:
    - pd.DataFrame: DataFrame with updated molecule IDs for each tracking event.
    """
    # Create a KD-tree for tracking events
    tree = tracking_events[['X', 'Y']].values
    kd_tree = KDTree(tree)

    # Add a column for molecule_id and initialize it
    tracking_events['MOLECULE_ID'] = -1

    molecule_id = 0

    # Iterate through each tracking event
    for i, track in tracking_events.iterrows():
        # Skip if the tracking event has already been assigned
        if tracking_events.at[i, 'MOLECULE_ID'] != -1:
            continue

        # Assign molecule ID to the current tracking event
        molecule_id += 1
        tracking_events.at[i, 'MOLECULE_ID'] = molecule_id

        frames_set = set(range(track['START_FRAME'], track['END_FRAME']+1))
        frames_set = frames_set - set(track['GAPS'])
        # Query the KD-tree for tracking events within a maximum distance
        indices = kd_tree.query_ball_point([track['X'], track['Y']], r=max_distance)

        
        for j in indices:
            # Skip if the tracking event has already been assigned
            if tracking_events.at[j, 'MOLECULE_ID'] != -1:
                continue

            # Obtain frames 
            new_frames_set = set(range(tracking_events.at[j, 'START_FRAME'], tracking_events.at[j, 'END_FRAME']+1))
            new_frames_set = new_frames_set - set(tracking_events.at[j, 'GAPS'])

            # Check frame overlap 
            if len(frames_set.intersection(new_frames_set)) == 0:
                # Assign molecule ID to the current tracking event
                tracking_events.at[j, 'MOLECULE_ID'] = molecule_id
                frames_set.update(new_frames_set)

    return tracking_events


def update_merged_locs(tracking_events, localizations):
    """
    Update the localizations DataFrame to reflect merged tracking events and remove obsolete tracks from tracking_events.

    Parameters:
    - tracking_events (pd.DataFrame): DataFrame containing tracking event data.
    - localizations (pd.DataFrame): DataFrame containing localizations with TRACK_ID.
    - track_changes (dict): Dictionary mapping old TRACK_IDs to new TRACK_IDs due to merging.

    Returns:
    - pd.DataFrame: Updated localizations DataFrame.
    - pd.DataFrame: Updated tracking_events DataFrame with old tracks removed.
    """

    track_changes = {}

    # Group by MOLECULE_ID and process each group
    for molecule_id, group in tracking_events.groupby('MOLECULE_ID'):
        sorted_group = group.sort_values(by='START_FRAME').reset_index(drop=True)

        for i in range(len(sorted_group) - 1):
            # If they were already added to track_changes, skip
            if sorted_group.loc[i, 'TRACK_ID'] in track_changes:
                continue

            for j in range(i+1, len(sorted_group) - 1):
                # Access tracking event i and j
                track_ini = tracking_events[tracking_events['TRACK_ID'] == sorted_group.loc[i, 'TRACK_ID']]
                track_fin = tracking_events[tracking_events['TRACK_ID'] == sorted_group.loc[j, 'TRACK_ID']]

                # Check if they are consecutive
                if (track_fin['START_FRAME'].values[0] - track_ini['END_FRAME'].values[0] <= 2):
                    track_changes[sorted_group.loc[j, 'TRACK_ID']] = sorted_group.loc[i, 'TRACK_ID']

                    # Obtain frames for both tracking events
                    frames_ini = set(range(track_ini['START_FRAME'].values[0], track_ini['END_FRAME'].values[0]+1))
                    gaps_ini = set(track_ini['GAPS'].values[0])
                    frames_ini = frames_ini - gaps_ini

                    frames_fin = set(range(track_fin['START_FRAME'].values[0], track_fin['END_FRAME'].values[0]+1))
                    gaps_fin = set(track_fin['GAPS'].values[0])
                    frames_fin = frames_fin - gaps_fin

                    # Join frames
                    frames_union = frames_ini.union(frames_fin)

                    # Update gaps
                    gaps_ini = gaps_ini.union(gaps_fin).difference(frames_union)

                    # Update tracking events
                    tracking_events.at[track_ini.index[0], 'GAPS'] = list(gaps_ini)
                    tracking_events.at[track_ini.index[0], 'END_FRAME'] = max(track_ini['END_FRAME'].values[0], track_fin['END_FRAME'].values[0])
                    tracking_events.at[track_ini.index[0], 'START_FRAME'] = min(track_ini['START_FRAME'].values[0], track_fin['START_FRAME'].values[0])
                else:
                    break

    # Update the TRACK_ID column in the localizations DataFrame using the track_changes
    # This assumes track_changes maps from old to new, as shown.
    localizations['TRACK_ID'] = localizations['TRACK_ID'].replace(track_changes)

    # Eliminate from tracking_events the old tracks that were merged
    # Here we use the keys from track_changes because those are the old TRACK_IDs that have been replaced.
    tracking_events = tracking_events[~tracking_events['TRACK_ID'].isin(track_changes.keys())]

    return localizations, tracking_events



def track_blinking_times(tracking_events):
    """
    Adds ON_TIME and OFF_TIME for tracking events in the DataFrame, considering only tracks within the same molecule.

    Parameters:
    - tracking_events (pd.DataFrame): DataFrame containing tracking event data.

    Returns:
    - pd.DataFrame: DataFrame with ON_TIME and OFF_TIME columns added.
    """
    # Calculate ON_TIME for each tracking event
    tracking_events['ON_TIME'] = tracking_events['END_FRAME'] - tracking_events['START_FRAME'] + 1

    # Initialize OFF_TIME column
    tracking_events['OFF_TIME'] = 0  # Initialize with 0 which will remain for the last track in each group

    # Group by MOLECULE_ID and process each group
    for molecule_id, group in tracking_events.groupby('MOLECULE_ID'):
        sorted_group = group.sort_values(by='START_FRAME').reset_index(drop=True)

        # Calculate OFF_TIME for each tracking event in the group except the last one
        for i in range(len(sorted_group) - 1):
            track_id = sorted_group.loc[i, 'TRACK_ID']
            tracking_events.loc[tracking_events['TRACK_ID'] == track_id, 'OFF_TIME'] = sorted_group.loc[i+1, 'START_FRAME'] - sorted_group.loc[i, 'END_FRAME']

    return tracking_events


def create_molecules(tracking_events):
    """
    Create a DataFrame of molecules from tracking events.
    
    Parameters:
    - tracking_events (pd.DataFrame): DataFrame of tracking events.
    
    Returns:
    - pd.DataFrame: DataFrame of molecules with molecule IDs.
    """
    # Correctly perform aggregation on the grouped DataFrame
    molecules = tracking_events.groupby('MOLECULE_ID').agg({
        'TRACK_ID': ['min', 'max', 'count'],  # Aggregates for TRACK_ID to find min, max, and count
        'ON_TIME': 'sum'  # Sum of ON_TIME for each molecule
    }).reset_index()

    # Rename columns to match your original intentions
    molecules.columns = ['MOLECULE_ID', 'START_TRACK', 'END_TRACK', '#_TRACKS', 'TOTAL_ON_TIME']

    return molecules


def bleaching_identification(molecules, tracking_events):
    """
    Identify potential bleaching events based on the END_FRAME of the tracking events and maximum OFF_TIME.
    
    Parameters:
    - molecules (pd.DataFrame): DataFrame of molecules with END_TRACK indicating the last track ID of each molecule.
    - tracking_events (pd.DataFrame): DataFrame of tracking events with OFF_TIME and specific frame data.
    
    Returns:
    - pd.DataFrame: DataFrame of molecules with a BLEACHED column indicating potential bleaching events.
    """
    # Find the maximum OFF_TIME from all tracking events
    max_off_time = tracking_events['OFF_TIME'].max()

    # Map END_TRACK to END_FRAME from tracking events
    track_to_end_frame = tracking_events.set_index('TRACK_ID')['END_FRAME'].to_dict()

    # Compare each molecule's END_FRAME to the max_off_time criterion
    for index, molecule in molecules.iterrows():
        # Fetch the END_FRAME for the corresponding END_TRACK of the molecule
        end_frame = track_to_end_frame.get(molecule['END_TRACK'])

        # Check if END_FRAME exists and calculate bleaching condition
        if end_frame is not None and 10000 - end_frame > max_off_time:
            molecules.at[index, 'BLEACHED'] = True

    return molecules




def prepare_columns(df, file_type):
    """ Prepare the columns for the DataFrame. """
    
    if file_type == 'trackmate':
        # Encontrar el primer índice donde los valores de 'LABEL_ID' comienzan con 'ID'
        first_valid_index = df[df['LABEL'].astype(str).str.startswith('ID')].index.min()

        # Eliminar todas las filas anteriores a este índice
        df = df.loc[first_valid_index:]

        # Change to upper case all column names
        df.columns = df.columns.str.upper()

        # Keep the columns we need
        df = df[['ID', 'FRAME', 'POSITION_X', 'POSITION_Y', 'POSITION_Z', 'QUALITY', 'TOTAL_INTENSITY_CH1', 'SNR_CH1', 'TRACK_ID']]
        # Change the column names
        df.columns = ['ID', 'FRAME', 'X', 'Y', 'Z', 'QUALITY', 'INTENSITY', 'SNR', 'TRACK_ID']
        # change type 
        df['ID'] = df['ID'].astype(int)
        # Convertir NaN a 0 y luego cambiar el tipo a int
        df['TRACK_ID'] = df['TRACK_ID'].fillna(0).astype(int)
        df['X'] = df['X'].astype(float)
        df['Y'] = df['Y'].astype(float)
        df['Z'] = df['Z'].astype(float)
        df['FRAME'] = df['FRAME'].astype(int)
        df['QUALITY'] = df['QUALITY'].astype(float)
        df['INTENSITY'] = df['INTENSITY'].astype(float)
        df['SNR'] = df['SNR'].astype(float)

    return df


def process_tracks(df, file_type):
    """ Process the entire file and merge tracking events. """

    raw_localizations = None
    tracking_events = None

    if file_type == 'trackmate':
        # Prepare the columns
        raw_localizations = prepare_columns(df, file_type)
        # Create tracking events DataFrame
        tracking_events = create_tracking_events(raw_localizations, file_type)
    elif file_type == 'thunderstorm':
        raw_localizations, tracking_events = merge_localizations(df, time.time())

    # Merge molecules
    merged_tracking_events = merge_tracking_events(tracking_events)

    # Update localizations
    localizations, merged_tracking_events = update_merged_locs(merged_tracking_events, raw_localizations)

    # Calculate blinking times
    blink_tracking_events = track_blinking_times(merged_tracking_events)

    # Create molecules
    molecules = create_molecules(blink_tracking_events)
    
    # Determine photobleaching
    #molecules = bleaching_identification(molecules, blink_tracking_events)

    return localizations, blink_tracking_events, molecules

