import os
import subprocess
import tkinter as tk
from tkinter import filedialog

import pandas as pd

def run_soax_analysis(image_path, parameter_file, executable_path, output_dir):
    """
    Run the SOAX analysis on the given input image.
    """
    
    # Prepare the command to run the batch mode of SOAX
    command = [executable_path, '-i', image_path, '-p', parameter_file, '-s', output_dir]

    # Run the command and wait for it to complete, capture output
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Print outputs for debugging
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

    # Check if the process was successful
    if result.returncode != 0:
        raise Exception(f"Error running SOAX analysis: {result.stderr}")
    else:
        print("SOAX analysis completed successfully.")



def obtain_df_result_snakes(snakes_folder):
    snake_data = []
    junctions_data = []
    
    # Loop through all txt files in the folder
    for filename in os.listdir(snakes_folder):
        if filename.endswith('.txt'):
            file_path = os.path.join(snakes_folder, filename)
            base_filename = filename.split('.')[0]
            with open(file_path, 'r') as file:
                content = file.readlines()
                
                # Find the start of the data
                start_index = -1
                for i, line in enumerate(content):
                    if 's p x y z fg_int bg_int'.replace(' ', '') in line.replace(' ', '').lower():
                        start_index = i + 1
                        break
                
                # Process snake data
                for line in content[start_index:]:
                    clean_line = line.strip()
                    if clean_line.startswith('['):
                        # Extract and process junction data
                        # Remove brackets and split by comma
                        clean_line = clean_line[1:-1].split(', ')
                        junction = [float(val) for val in clean_line]
                        
                        junction.insert(0, base_filename)

                        junctions_data.append(junction)
                    elif clean_line and not clean_line.startswith('#'):
                        # Split the line by whitespace and convert to floats as needed
                        snake = clean_line.split()
                        snake = [float(val) if i > 1 else int(val) for i, val in enumerate(snake)]
                        
                        snake.insert(0, base_filename)

                        snake_data.append(snake)

            # Delete txt file after processing
            os.remove(file_path)
    
    snakes_df = pd.DataFrame(snake_data, columns=['File', 'Snake', 'Point', 'x', 'y', 'z', 'fg_int', 'bg_int'])
    junctions_df = pd.DataFrame(junctions_data, columns=['File', 'x', 'y', 'z'])

    snakes_df['Snake'] = snakes_df['Snake'].astype(int)
    snakes_df['Point'] = snakes_df['Point'].astype(int)

    return snakes_df, junctions_df





def soac_api(image_path, parameter_file, executable_path, output_dir):
    """
    Run the SOAC analysis on the given input image and return the results as DataFrames.
    """
    roi_path = os.path.join(output_dir, 'ROIs')
    # Run the SOAC analysis
    run_soax_analysis(image_path, parameter_file, executable_path, roi_path)
    
    # Obtain the results from the output directory
    snakes_df, junctions_df = obtain_df_result_snakes(image_path)
    
    return snakes_df, junctions_df


