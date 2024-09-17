import os
import numpy as np
import pandas as pd
from skimage import io

# Results csv content
#   - unique identifier for results file, ignored (int)
# Frame - This will be ignored as we are analyzing after doing a stack projection (int)
# Contour ID - Id of the contour (int)
# Pos. - Position of point in regards to contour (int). Starts at 1 and goes up to the number of points in the contour
# X - X coordinate of the point (float)
# Y - Y coordinate of the point (float)
# Length - Length of the contour (float). This value is the length of the contour so it is the same for all points in the contour
# Contrast - Contrast of the point (float). 
# Asymmetry - Asymmetry of the point (float).
# Line width - Line width of the point (float).
# Angle of normal - Angle of the point (float).
# Class - Defines if it has junctions. Possible values are: 'start_junc', 'end_junc', 'both_junc', 'no_junc' (str)

# Junctions csv content
# Frame - This will be ignored as we are analyzing after doing a stack projection (int)
# Contour ID 1 - Id of the contour 1 (int)
# Contour ID 2 - Id of the contour 2 (int)
# X - X coordinate of the junction (int)
# Y - Y coordinate of the junction (int)


# Contour Network function
# Input: results (DataFrame) - Pandas DataFrame containing the results data
#       junctions (DataFrame) - Pandas DataFrame containing the junction data
# Output: network (DataFrame) - Pandas DataFrame containing the network data
# It creates a network DataFrame from the results and junctions DataFrames
# It starts labelling from the ones 'start_junc' and goes through the junctions attached to them, assigning the same label until it reaches all 'end_junc'
# It returns a dataframe  that adds a column 'Network' to the results DataFrame
# The 'Network' column contains the label of the network that the point belongs to
# The network is defined by the junctions that are connected to each other
def contour_network(results, junctions): 
    # Add a 'Network' column to the results DataFrame
    # Put the column at the beginning of the DataFrame
    results.insert(0, 'Network', 0)
    
    # Initialize a counter for network labels
    network_label = 0
    # Contour ID counter
    contour_id = 0
    
    # Process each 'start_junc' in the results DataFrame
    for index, start_point in results[results['Class'] == 'start_junc'].iterrows():
        if (results.loc[results['Contour ID'] == start_point['Contour ID'], 'Network'] != 0).any():
            continue
        # Increment the network label counter
        if contour_id != start_point['Contour ID']:
            contour_id = start_point['Contour ID']
            network_label += 1
        
        # Start a queue with the current start junction point
        queue = [start_point['Contour ID']]
        
        # While there are items in the queue, process each one
        while queue:
            current_id = queue.pop(0)
            
            # Label the current contour ID in the results DataFrame
            # Network label should be an integer
            network_label = int(network_label)
            results.loc[results['Contour ID'] == current_id, 'Network'] = network_label
            
            # Find junctions where the current contour ID is involved
            linked_junctions = junctions[(junctions['Contour ID 1'] == current_id) | (junctions['Contour ID 2'] == current_id)]
            
            # Iterate through linked junctions to enqueue linked contour IDs
            for _, junction in linked_junctions.iterrows():
                next_id = junction['Contour ID 2'] if junction['Contour ID 1'] == current_id else junction['Contour ID 1']
                if (results.loc[results['Contour ID'] == next_id, 'Network'] == 0).any():
                    queue.append(next_id)
    
    return results


# Sinuosities function
# Input: results (DataFrame) - Pandas DataFrame containing the results data
# Output: results (DataFrame) - Pandas DataFrame containing the results data with the 'Sinuosity' column added
# It calculates the sinuosity of each contour and adds a 'Sinuosity' column to the results DataFrame
# The sinuosity is calculated as the ratio of the contour length to the position 1 to the position n of the contour (last point)
def calculate_sinuosity(results):
    # Check if 'Sinuosity' column exists; if not, create it
    if 'Sinuosity' not in results.columns:
        results['Sinuosity'] = np.nan
    
    # Group by 'Contour ID' since each contour can have multiple points
    grouped = results.groupby('Contour ID')
    
    # Calculate sinuosity for each contour
    for contour_id, group in grouped:
        # Calculate the straight-line distance between the first and the last point of the contour
        first_point = group.iloc[0]
        last_point = group.iloc[-1]
        straight_line_distance = np.sqrt((last_point['X'] - first_point['X']) ** 2 + (last_point['Y'] - first_point['Y']) ** 2)
        
        # Get the contour length from any row (since it's the same for all points in the same contour)
        contour_length = group['Length'].iloc[0]
        
        # Calculate sinuosity
        if straight_line_distance > 0 and contour_length > 0:  # Prevent division by zero
            sinuosity = contour_length / straight_line_distance
            if sinuosity > 1:
                results.loc[results['Contour ID'] == contour_id, 'Sinuosity'] = sinuosity
            else:
                results.loc[results['Contour ID'] == contour_id, 'Sinuosity'] = 1  # Handle potential zero distance case
        else:
            results.loc[results['Contour ID'] == contour_id, 'Sinuosity'] = np.nan  # Handle potential zero distance case
    
    return results


# Gaps functions
# Input: results (DataFrame) - Pandas DataFrame containing the results data
# Output: results (DataFrame) - Pandas DataFrame containing the results data with the 'Gaps' column added
# It calculates the number of gaps in each contour and adds a 'Gaps' column to the results DataFrame
# A gap is defined as a point where the contrast is below 1 and or the line width is below 0.5
# It should count as 1 gap if there are multiple joined points that are below the required values
# The first and last points of the contour should not be considered as gaps
def calculate_gaps(results):
    # Initialize the 'Gaps' column with zeros
    results['Gaps'] = 0
    
    # Group the data by 'Contour ID' to process each contour individually
    for contour_id, group in results.groupby('Contour ID'):
        # Ensure the data is sorted by 'Pos.' to correctly identify gaps
        group = group.sort_values('Pos.')
        
        # Initialize gap counting
        in_gap = False
        gap_count = 0
        
        # Iterate over the rows in the group
        for index, row in group.iterrows():
            # Check if the current point should be considered as part of a gap
            is_gap = (row['Contrast'] < 1 or row['Line width'] < 0.5)
            
            # If the first or last point, ignore the gap rule
            if row['Pos.'] == 1 or row['Pos.'] == group['Pos.'].max():
                continue
            
            # If entering a gap
            if is_gap and not in_gap:
                in_gap = True
            
            # If exiting a gap
            elif not is_gap and in_gap:
                gap_count += 1
                in_gap = False
        
        # If ended in a gap, increment the gap count
        if in_gap:
            gap_count += 1
        
        # Assign the gap count to all points in this contour
        results.loc[results['Contour ID'] == contour_id, 'Gaps'] = gap_count
    
    return results


# Intensity function
# Input: results (DataFrame) - Pandas DataFrame containing the results data
#       image (ndarray) - NumPy array containing the image data
# Output: results (DataFrame) - Pandas DataFrame containing the results data with the 'Intensity' column added
# It calculates the intensity of each point in the results DataFrame
# The intensity is calculated as the pixel value in the image at the X and Y coordinates of the point
# It should convert X and Y to integers before indexing the image, it makes sure it stays inside the image boundaries
# If the point is outside the image boundaries, it should approximate the position to the nearest pixel inside the image
def calculate_intensity(results, image):
    # Initialize the 'Intensity' column
    results['Intensity'] = 0
    
    # Get image dimensions
    max_y, max_x = image.shape[:2]  # Assuming a 2D image or 2D layer of a multi-channel image
    
    # Calculate intensity for each point
    for index, row in results.iterrows():
        # Convert X and Y to integer and ensure they are within the image boundaries
        x = int(min(max(round(row['X']), 0), max_x - 1))
        y = int(min(max(round(row['Y']), 0), max_y - 1))
        
        # Assign the pixel value at (y, x) to the 'Intensity' column
        results.at[index, 'Intensity'] = image[y, x]
    
    return results

# Dataframe agrouping function
# Input: results (DataFrame) - Pandas DataFrame containing the results data
# Output: grouped (DataFrame) - Pandas DataFrame containing the grouped data
# It groups the data by 'Contour ID' and calculates the mean of the 'Contrast', 'Line Width', and 'Intensity' columns
def dataframe_agrouping(df):
    # Define custom aggregation rules: mean for most columns, 'first' for the 'Class' column
    aggregation_rules = {
        'Pos.': 'mean',
        'X': 'mean',
        'Y': 'mean',
        'Length': 'mean',
        'Contrast': 'mean',
        'Line width': 'mean',
        'Sinuosity': 'mean',
        'Gaps': 'mean',
        'Intensity': 'mean',
        'Class': 'first'  # Use 'first' for non-numeric data
    }

    # Apply groupby and aggregate functions
    grouped_df = df.groupby(['Network', 'Contour ID']).agg(aggregation_rules).reset_index()

    grouped_df = grouped_df.sort_values(by=['Network', 'Contour ID'])
    
    return grouped_df


# Analyze data function
# Input: results (DataFrame) - Pandas DataFrame containing the results data
#    junctions (DataFrame) - Pandas DataFrame containing the junction data
#    image (ndarray) - NumPy array containing the image data
# Output: results (DataFrame) - Pandas DataFrame containing the results data with the analyzed columns added
# It calculates the sinuosity, gaps, and intensity of the points in the results DataFrame
# It also creates a network DataFrame from the results and junctions DataFrames
def analyze_data(results, junctions, image):
    # Remove ' ', Frame, assymetry and angle of normal columns
    results = results.drop(columns=[' ', 'Frame', 'Asymmetry', 'Angle of normal'])

    # Calculate sinuosity
    results = calculate_sinuosity(results)
    
    # Calculate gaps
    results = calculate_gaps(results)
    
    # Calculate intensity
    results = calculate_intensity(results, image)
    
    # Create network
    results = contour_network(results, junctions)

    # Group dataframe
    results = dataframe_agrouping(results)

    # Change the name and order of the columns
    results = results.rename(columns={'Contour ID': 'Contour'})
    results = results[['Network', 'Contour', 'Length', 'Line width', 'Intensity', 'Contrast', 'Sinuosity', 'Gaps', 'Class']]
    
    return results