import numpy as np
import pandas as pd


def analyze_data_storm(file_path):

    # Load CSV file into DataFrame
    df = pd.read_csv(file_path)

    # Print the dataframe was loaded
    print("Data loaded.")
    
    # Process the DataFrame to merge localizations
    molecules, id_to_molecule = merge_localizations(df)

    # Create a new DataFrame with the molecules
    df['molecule_id'] = df['id'].map(id_to_molecule)
    

    print("Data processing complete and saved.")
    #return processed_df

def merge_localizations(df):
    
    # Add id column if it doesn't exist
    if 'id' not in df.columns:
        df['id'] = range(len(df))

    # Initialize molecule ID column
    df['molecule_id'] = np.nan

    # Create a molecules dictionary to store the merged localizations
    molecules = {}
    id_to_molecule = {}
    actual_frame = 0
    new_molecules = 0
    assigned_molecules = 0

    # Iterate through each row
    for index, row in df.iterrows():
        loc_id = row['id']
        loc_frame = row['frame']
        loc_x = row['x [nm]']
        loc_y = row['y [nm]']
        loc_uncertainty = row['uncertainty_xy [nm]']

        # Check if the frame has changed and say how many new molecules were found in the last frame as well as how many where assigned to a previous one
        if loc_frame != actual_frame:
            if actual_frame != 0:
                print(f"Frame {loc_frame} - New: {new_molecules} - Assigned: {assigned_molecules}")
            actual_frame = loc_frame
            new_molecules = 0
            assigned_molecules = 0

        mol_id = find_closest_molecule(loc_x, loc_y, molecules, loc_uncertainty)

        if mol_id:
            assigned_molecules += 1
            # Update existing molecule
            mol_data = molecules[mol_id]
            mol_data['ids'].append(loc_id)
            mol_data['localizations'].append({'x': loc_x, 'y': loc_y, 'uncertainty': loc_uncertainty})

            # Recalculate the molecule properties
            new_x = np.mean([loc['x'] for loc in mol_data['localizations']])
            new_y = np.mean([loc['y'] for loc in mol_data['localizations']])
            new_uncertainty = max(loc['uncertainty'] for loc in mol_data['localizations'])

            # Update molecule dictionary
            mol_data['x'] = new_x
            mol_data['y'] = new_y
            mol_data['uncertainty'] = new_uncertainty
        else:
            new_molecules += 1
            mol_id = len(molecules)
            # Create new molecule
            molecules[len(molecules)] = {
                'x': loc_x, 'y': loc_y, 'uncertainty': loc_uncertainty,
                'ids': [loc_id],
                'localizations': [{'x': loc_x, 'y': loc_y, 'uncertainty': loc_uncertainty}]
            }

        # Assign molecule ID directly to the DataFrame
        df.at[index, 'molecule_id'] = mol_id  
            
            
    # Print the number of molecules found
    print(f"Found {len(molecules)} molecules.")

    return molecules, id_to_molecule



def find_closest_molecule(x,y, molecules, threshold):
    # It finds the closest molecule to a given point (x,y) within a given threshold
    closest_molecule = None
    min_distance = threshold
    for mol_id, mol_data in molecules.items():
        distance = ((x - mol_data['x'])**2 + (y - mol_data['y'])**2)**0.5
        # Here I should still try to change the way it is decided. Especially because the uncertainty is biased by number of frame 
        if distance < min_distance and distance <= mol_data['uncertainty']:
            min_distance = distance
            closest_molecule = mol_id

    return closest_molecule
        