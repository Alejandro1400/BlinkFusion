# soac_filament_analysis.py
import os
import pandas as pd
import streamlit as st

from Analysis.SOAC.analytics_soac_filaments import obtain_cell_metrics
from Data_access import box_connection
from Data_access.file_explorer import assign_structure_folders, find_items, find_valid_folders


@st.cache_data
def load_soac_data(soac_folder):
    # List to store dataframes from all CSVs
    cells = []
    metrics = []

    try:
        # Find the folder structure file (folder_structure.txt)
        structure_path = find_items(base_directory=soac_folder, item='folder_structure.txt', is_folder=False)

        if structure_path:
            # Find all valid folders (folders that contain the required files)
            valid_folders = find_valid_folders(
                soac_folder,
                required_files={'soac_results.csv'}
            )
            
            # Get the folder structure as a DataFrame
            image_df = assign_structure_folders(soac_folder, structure_path, valid_folders)

            # Iterate through each valid folder
            for folder in valid_folders:
                cell = find_items(
                    base_directory=folder, 
                    item='soac_results.csv', 
                    is_folder=False, 
                    search_by_extension=True
                )

                if cell:
                    try:
                        # Load the dataframes
                        cell_df = pd.read_csv(cell)
                        
                        # Calculate the relative path and assign it
                        relative_path = os.path.relpath(folder, soac_folder)
                        cell_df['IDENTIFIER'] = relative_path

                        cell_metrics = obtain_cell_metrics(cell_df)
                        cell_metrics['IDENTIFIER'] = relative_path  # Use IDENTIFIER for joining later

                        # Append the metrics to the list to be processed later
                        metrics.append(cell_metrics)

                        # Append the dataframes to the lists
                        cells.append(cell_df)

                    except Exception as e:
                        print(f"Failed to process files in {folder}. Error: {e}")

        # Combine all dataframes into a single dataframe for each type
        cells_df = pd.concat(cells, ignore_index=True)
        print(cells_df.head())

        # Combine all mol_metrics into a single dataframe
        metrics_df = pd.concat(metrics)

        # Merge mol_metrics with image_df using IDENTIFIER
        merged_metrics = pd.merge(metrics_df, image_df, on='IDENTIFIER', how='left')
        # Reorder columns to have image_df columns first, then metrics_df columns, excluding duplicates
        merged_metrics = merged_metrics[image_df.columns.tolist() + [col for col in merged_metrics.columns if col not in image_df.columns]]

        return cells_df, image_df, merged_metrics

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None


def run_soac_ui(soac_folder):

    cells, images, metrics = load_soac_data(soac_folder)

    if cells.empty or images.empty or metrics.empty:
        st.error("Failed to load data. Please check the folder structure and contents.")
        return

    # Display Images loaded
    st.write("Images loaded:")
    # Reset the index and drop the old one to ensure it does not appear in the display
    metrics_no_id = metrics.drop(columns=['IDENTIFIER'])
    st.dataframe(metrics_no_id)

    # Average by Date, sample. Drop cell column
    metrics_no_id.drop(columns=['CELL'], inplace=True)
    metrics_avg = metrics_no_id.groupby(['DATE', 'SAMPLE']).mean()
    st.write("Averages by Date, Sample, and Cell:")
    st.dataframe(metrics_avg)
