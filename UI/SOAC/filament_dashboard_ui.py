# soac_filament_analysis.py
import os
import time
import pandas as pd
import streamlit as st

from Analysis.SOAC.analytics_soac_filaments import obtain_cell_metrics
from Data_access import box_connection
from Data_access.file_explorer import find_items, find_valid_folders
from Data_access.metadata_manager import read_tiff_metadata


@st.cache_data
def load_filament_data(soac_folder):
    # List to store dataframes from all CSVs
    metadata = []
    cells = []
    metrics = []

    # Progress bar and status text
    status_text = st.text("Loading filaments data...")
    progress_bar = st.progress(0, text="Loading filaments data...")

    try:
        # Find all valid folders (folders that contain the required files)
        valid_folders = find_valid_folders(
            soac_folder,
            required_files={'soac_results.csv'}
        )
        total_folders = len(valid_folders)

        # Iterate through each valid folder
        for index, folder in enumerate(valid_folders):
            time.sleep(0.1)  # Add a delay to update the progress bar

            cell = find_items(
                base_directory=folder, 
                item='soac_results.csv', 
                is_folder=False, 
                search_by_extension=True
            )

            tif_file = find_items(
                base_directory=folder, 
                item='.tif', 
                is_folder=False, 
                search_by_extension=True
            )

            if cell and tif_file:
                try:
                    # Calculate the relative path and assign it
                    relative_path = os.path.relpath(folder, soac_folder)
                    progress_bar.progress((index + 1) / total_folders, text=f"Loading snakes file for {relative_path} ({index + 1} of {total_folders})")

                    pulsestorm_metadata = read_tiff_metadata(tif_file, root_tag=['pulsestorm', 'tif-pulsestorm'])

                    metadata_dict = {}

                    # Loop through each item in the metadata list
                    for item in pulsestorm_metadata:
                        # Check if the item's 'id' is not already in the dictionary
                        if item['id'] not in metadata_dict:
                            # If not present, add the item's 'id' and 'value' to the dictionary
                            metadata_dict[item['id']] = item['value']

                    meta_df = pd.DataFrame(metadata_dict)
                    meta_df.columns = meta_df.columns
                    meta_df['IDENTIFIER'] = relative_path
                    meta_df['IMAGE'] = os.path.basename(tif_file)

                    # Load the dataframes
                    cell_df = pd.read_csv(cell)
                    cell_df['IDENTIFIER'] = relative_path

                    progress_bar.progress((index + 1) / total_folders, text=f"Obtaining metrics for {relative_path} ({index + 1} of {total_folders})")

                    cell_metrics = obtain_cell_metrics(cell_df)
                    cell_metrics['IDENTIFIER'] = relative_path  # Use IDENTIFIER for joining later

                    # Append the metrics to the list to be processed later
                    metadata.append(meta_df)
                    metrics.append(cell_metrics)
                    cells.append(cell_df)

                except Exception as e:
                    print(f"Failed to process files in {folder}. Error: {e}")

            status_text.empty()
            progress_bar.empty()

        # Combine all dataframes into a single dataframe for each type
        cells_df = pd.concat(cells)
        metadata_df = pd.concat(metadata)
        metrics_df = pd.concat(metrics)

        return cells_df, metadata_df, metrics_df

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None


def run_filament_dashboard_ui(soac_folder):

    cells, metadata, metrics = load_filament_data(soac_folder)

    st.write(f" **{len(metadata)}** images loaded in the Dataset.")

    if cells.empty or metadata.empty or metrics.empty:
        st.error("Failed to load data. Please check the folder structure and contents.")
        return
    
    # Initialize copies of all datasets for use after filtering
    cells_analysis = cells.copy()
    metadata_analysis = metadata.copy()
    metrics_analysis = metrics.copy()

    desc_columns = metadata_analysis.columns

    with st.expander("Filter Options", expanded=True):

        if not metadata.index.is_unique:
            metadata = metadata.reset_index(drop=True)

        # Section for selecting filters
        st.write("Apply filters based on the selected columns (all values are selected by default).")
        selected_filter_columns = st.multiselect(
            'Select columns to filter by:',
            list(desc_columns),
            key='filter_select'
        )
        
        # Apply filters if any columns are selected for filtering
        if selected_filter_columns:
            filters = {}
            filter_mask = pd.Series([True] * len(metadata))
            for col in selected_filter_columns:
                unique_values = metadata[col].unique()
                selected_values = st.multiselect(
                    f"Filter {col}:",
                    unique_values,
                    default=unique_values,
                    key=f'filter_{col}'
                )
                filters[col] = selected_values
                filter_mask &= metadata[col].isin(selected_values)

            metadata_analysis = metadata[filter_mask]
        else:
            metadata_analysis = metadata.copy()

        # Display grouped data, ensure IDENTIFIER is included if not already in the group by list
        if 'IDENTIFIER' not in selected_filter_columns and selected_filter_columns:
            selected_filter_columns.append('IDENTIFIER')

        # If no columns are selected for grouping, use all available columns from filtered metadata
        display_columns = selected_filter_columns if selected_filter_columns else metadata_analysis.columns.tolist()
        filtered_metadata = metadata_analysis[display_columns]

        st.markdown("___")

        st.write("Data after Filtering:")
        if not metadata_analysis.empty:
            st.dataframe(filtered_metadata)
        else:
            st.write("No data found for the selected groups.")
