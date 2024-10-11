import numpy as np
import pandas as pd
from scipy.spatial import KDTree
import time

def analyze_data_storm(file_path):
    # Load CSV file into DataFrame
    df = pd.read_csv(file_path)
    print("Data loaded.")

    # Start the timer
    start_time = time.time()
    
    # Process the DataFrame to merge localizations using a KD-tree
    id_to_molecule, time_dict = merge_localizations(df, start_time)

    # Create a new DataFrame with the molecules
    df['molecule_id'] = df['id'].map(id_to_molecule)

    # Save the processed data to a new CSV file
    output_file = file_path.replace('.csv', '_processed.csv')
    df.to_csv(output_file, index=False)
    print("Data processing complete and saved.")

    # Save the timing dictionary as a CSV file
    time_df = pd.DataFrame(list(time_dict.items()), columns=['Frame Range', 'Elapsed Time (s)'])
    time_output_file = file_path.replace('.csv', '_timing.csv')
    time_df.to_csv(time_output_file, index=False)
    print(f"Timing data saved to {time_output_file}.")

def merge_localizations(df, start_time):
    # Add id column if it doesn't exist
    if 'id' not in df.columns:
        df['id'] = range(len(df))

    # If uncertainty_xy column is uncertainty [nm], rename it
    if 'uncertainty [nm]' in df.columns:
        df.rename(columns={'uncertainty [nm]': 'uncertainty_xy [nm]'}, inplace=True)

    # Initialize molecule ID column
    df['molecule_id'] = np.nan

    # Create a molecules list to store the merged localizations
    molecules = []
    id_to_molecule = {}

    # Dictionary to store elapsed time for each frame range
    time_dict = {}

    # Store the coordinates and uncertainties for all localizations
    coords = df[['x [nm]', 'y [nm]']].values
    uncertainties = df['uncertainty_xy [nm]'].values

    # Create a KD-tree for storing and querying molecule positions
    kd_tree = None  # Initially no molecules are present
    actual_frame = 0
    last_frame = 0
    new_molecules = 0
    assigned_molecules = 0

    # Iterate through each row
    for index, (loc_id, loc_frame, loc_x, loc_y, loc_uncertainty) in df[['id', 'frame', 'x [nm]', 'y [nm]', 'uncertainty_xy [nm]']].iterrows():
        loc_coord = np.array([loc_x, loc_y])

        # Check if the frame has changed and if it is mod 500
        if loc_frame != actual_frame and loc_frame % 500 == 0:
            if actual_frame != 0:
                elapsed_time = time.time() - start_time
                frame_range = f"{last_frame} to {actual_frame}"
                print(f"Frame {frame_range} - New: {new_molecules} - Assigned: {assigned_molecules} - Elapsed time: {elapsed_time:.2f} seconds")
                time_dict[frame_range] = elapsed_time
            last_frame = actual_frame
            actual_frame = loc_frame
            new_molecules = 0
            assigned_molecules = 0

        # KD-tree querying and molecule assignment logic
        if kd_tree and len(molecules) > 0:
            dist, nearest_index = kd_tree.query(loc_coord, distance_upper_bound=loc_uncertainty)
            if dist < loc_uncertainty and dist < molecules[nearest_index]['uncertainty']:
                assigned_molecule = nearest_index
                molecules[assigned_molecule]['ids'].append(loc_id)
                molecules[assigned_molecule]['localizations'].append(loc_coord)
                molecules[assigned_molecule]['uncertainties'].append(loc_uncertainty)

                # Update molecule properties (average position)
                localizations = np.array(molecules[assigned_molecule]['localizations'])
                uncertainties = np.array(molecules[assigned_molecule]['uncertainties'])

                # Calculate the weighted average of the localizations
                weights = 1 / uncertainties**2
                new_x = np.average(localizations[:, 0], weights=weights)
                new_y = np.average(localizations[:, 1], weights=weights)
                new_uncertainty = np.sqrt(1 / weights.sum())

                molecules[assigned_molecule]['x'] = new_x
                molecules[assigned_molecule]['y'] = new_y
                molecules[assigned_molecule]['uncertainty'] = new_uncertainty

                id_to_molecule[loc_id] = assigned_molecule
                df.at[index, 'molecule_id'] = assigned_molecule
                assigned_molecules += 1
            else:
                # Create a new molecule if no suitable match is found
                new_molecule_id = len(molecules)
                molecules.append({
                    'x': loc_x, 'y': loc_y, 'uncertainty': loc_uncertainty,
                    'ids': [loc_id],
                    'localizations': [loc_coord],
                    'uncertainties': [loc_uncertainty]
                })
                id_to_molecule[loc_id] = new_molecule_id
                df.at[index, 'molecule_id'] = new_molecule_id
                new_molecules += 1
                kd_tree = KDTree([mol['localizations'][0] for mol in molecules])
        else:
            # If there are no molecules yet, add the first molecule directly
            molecules.append({
                'x': loc_x, 'y': loc_y, 'uncertainty': loc_uncertainty,
                'ids': [loc_id],
                'localizations': [loc_coord],
                'uncertainties': [loc_uncertainty]
            })
            id_to_molecule[loc_id] = len(molecules) - 1
            df.at[index, 'molecule_id'] = len(molecules) - 1
            new_molecules += 1
            kd_tree = KDTree([mol['localizations'][0] for mol in molecules])

    # Print information for the last frame range
    elapsed_time = time.time() - start_time
    frame_range = f"{last_frame} to {actual_frame}"
    print(f"Frame {frame_range} - New: {new_molecules} - Assigned: {assigned_molecules} - Elapsed time: {elapsed_time:.2f} seconds")
    time_dict[frame_range] = elapsed_time

    # Print the total number of molecules found
    print(f"Found {len(molecules)} molecules. Total time: {elapsed_time:.2f} seconds")

    return id_to_molecule, time_dict
