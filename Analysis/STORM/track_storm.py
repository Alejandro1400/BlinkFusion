import numpy as np
import pandas as pd
from scipy.spatial import KDTree
import time


def merge_localizations(df, start_time, min_frames=3, max_gaps=2, max_distance=200):

    # Add id column if it doesn't exist
    if 'id' not in df.columns:
        df['id'] = range(len(df))

    # Assign frame as int
    df['id'] = df['id'].astype(int)
    df['frame'] = df['frame'].astype(int)

    # If uncertainty_xy column is uncertainty [nm], rename it
    if 'uncertainty_xy [nm]' in df.columns:
        df.rename(columns={'uncertainty_xy [nm]': 'uncertainty [nm]'}, inplace=True)

    # Initialize track ID column
    df['TRACK_ID'] = 0

    # Create a tracks list to store the merged localizations
    active_tracks = []
    tracking_events = pd.DataFrame(columns=['TRACK_ID', 'X', 'Y', 'Z', 'START_FRAME', 'END_FRAME', 'INTENSITY', 'OFFSET', 'BKGSTD', 'UNCERTAINTY', 'GAPS'])

    # Dictionary to store elapsed time for each frame range
    time_dict = {}

    actual_frame = 0
    last_frame = 0
    locs_merged = 0
    assigned_tracks = 0
    track_id = 0

    df = df.sort_values('frame')

    # Iterate through each frame's localizations
    for frame, frame_data in df.groupby('frame'):
        # Check if the frame is mod 500
        if frame != actual_frame and frame % 500 == 0:
            elapsed_time = time.time() - start_time
            last_frame = actual_frame
            actual_frame = frame
            frame_range = f"{last_frame} to {actual_frame}"
            if frame != 0:
                print(f"Frame {frame_range} - Localizations Merged: {locs_merged} - Assigned tracks: {assigned_tracks} - Elapsed time: {elapsed_time:.2f} seconds")
            time_dict[frame_range] = elapsed_time
            locs_merged = 0
            assigned_tracks = 0

        # Add frame localizations to kd-tree
        frame_localizations = frame_data[['x [nm]', 'y [nm]']].values
        kd_tree = KDTree(frame_localizations)

        index_tracked = []
        tracks_to_remove = []

        # Iterate through active tracks
        for track in active_tracks:
            # Query the kd-tree for the nearest localization
            dist_last, nearest_index_last = kd_tree.query(track['localizations'][-1], distance_upper_bound=max_distance)
            dist_weighted, nearest_index_weighted = kd_tree.query([track['x'], track['y']], distance_upper_bound=max_distance)

            dist, nearest_index = (dist_last, nearest_index_last) if dist_last < dist_weighted else (dist_weighted, nearest_index_weighted)

            if dist < max_distance and nearest_index not in index_tracked:
                index_tracked.append(nearest_index)
                coord_loc = frame_localizations[nearest_index]
                localization = frame_data.iloc[nearest_index]
                # Add the localization to the track
                track['ids'].append(localization['id'])
                track['localizations'].append(coord_loc)
                track['uncertainties'].append(localization['uncertainty [nm]'])

                # Obtain the weighted average of the localizations
                localizations = np.array(track['localizations'])
                uncertainties = np.array(track['uncertainties'])
                weights = 1 / uncertainties**2
                new_x = np.average(localizations[:, 0], weights=weights)
                new_y = np.average(localizations[:, 1], weights=weights)
                new_uncertainty = np.sqrt(1 / weights.sum())

                track['x'] = new_x
                track['y'] = new_y
                track['uncertainty'] = new_uncertainty
                track['intensity'] += localization['intensity [photon]']
                track['offset'] += localization['offset [photon]']
                track['bkgstd'] += localization['bkgstd [photon]']

                track['gap'] = 0

            else:
                track['gap'] += 1

            
            # Check if the track has reached the maximum number of gaps
            if track['gap'] >= max_gaps:
                # Check if the track has more than one localization
                if len(track['localizations']) >= min_frames:
                    filtered_df = df[df['id'].isin(track['ids'])]

                    # Get a set of frames from the filtered DataFrame
                    frame_set = set(filtered_df['frame'])

                    # Generate a set of all frames from start_frame to end_frame
                    all_frames = set(range(track['start_frame'], frame - track['gap'] + 1))

                    # Determine the frames where there is a gap
                    gap_frames = all_frames - frame_set

                    track_id += 1
                    # Add the track to the tracking events DataFrame
                    new_track_df = pd.DataFrame([{
                        'TRACK_ID': track_id,
                        'X': track['x'],
                        'Y': track['y'],
                        'Z': 0,
                        'START_FRAME': track['start_frame'],
                        'END_FRAME': frame - track['gap'],
                        'INTENSITY': track['intensity'],
                        'OFFSET': track['offset'] / len(track['ids']),
                        'BKGSTD': track['bkgstd'],
                        'UNCERTAINTY': track['uncertainty'],
                        'GAPS': list(gap_frames),
                    }])
                    tracking_events = pd.concat([tracking_events, new_track_df], ignore_index=True)
                    # Update the track ID for each localization
                    for loc_id in track['ids']:
                        df.loc[df['id'] == loc_id, 'TRACK_ID'] = track_id

                    assigned_tracks += 1
                    locs_merged += len(track['ids'])

                # Remove the track from the active tracks
                tracks_to_remove.append(track)

        # Remove the tracks that have been assigned to tracking events
        for track in tracks_to_remove:
            active_tracks.remove(track)

        # Add new tracks for unassigned localizations
        for i in range(len(frame_data)):
            if i not in index_tracked:
                coord_loc = frame_localizations[i]
                localization = frame_data.iloc[i]
                active_tracks.append({
                    'x': coord_loc[0],
                    'y': coord_loc[1],
                    'uncertainty': localization['uncertainty [nm]'],
                    'start_frame': frame,
                    'gap': 0,
                    'intensity': localization['intensity [photon]'],
                    'offset': localization['offset [photon]'],
                    'bkgstd': localization['bkgstd [photon]'],
                    'ids': [localization['id']],
                    'localizations': [coord_loc],
                    'uncertainties': [localization['uncertainty [nm]']]
                })

    # Change localizations columns to uppercase
    df.columns = [col.upper() for col in df.columns]

    return df, tracking_events



