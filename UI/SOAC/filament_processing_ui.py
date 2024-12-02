import os
import pandas as pd
import streamlit as st

from Analysis.SOAC.analytics_soac_filaments import soac_analytics_pipeline
from Analysis.SOAC.preprocessing_image_selection import preprocessing_image_selection
from Analysis.SOAC.soac_api import soac_api
from Data_access.file_explorer import find_items, find_valid_folders, process_and_save_rois, save_csv_file


def run_filament_processing_ui(soac_folder, config_path, parameter_file, executable_path):
    """
    Streamlit UI for running SOAC analysis on selected folders and files.

    Args:
        soac_folder (str): Path to the folder containing data for SOAC analysis.
        config_path (str): Path to the configuration file ('ridge_detector_param.json').
        parameter_file (str): Path to the parameters file ('batch_parameters.txt').
        executable_path (str): Path to the SOAC executable file ('batch_soax_v3.7.0.exe').
    """
    if soac_folder is None or soac_folder == "":
        st.error("SOAC database folder not set. Please configure it in the sidebar.")
        return
    
    st.info("This interface allows you to process SOAC analysis for all valid folders and `.tif` files.")

    # Find all valid folders
    with st.spinner("Searching for valid folders..."):
        valid_folders = find_valid_folders(
            soac_folder,
            required_files={'.tif'},
            exclude_files={'ridge_metrics.csv', 'soac_results.csv'},
            exclude_folders={'ROIs'}
        )
    
    st.write(f"Found **{len(valid_folders)}** valid folders for analysis.")

    # Check if no valid folders were found
    if not valid_folders:
        st.warning("No valid folders found. Please check the folder structure and contents.")
        return

    # Collect all `.tif` files across valid folders
    all_tif_files = []
    for folder in valid_folders:
        tif_files = find_items(
            base_directory=folder,
            item='.tif',
            is_folder=False,
            check_multiple=True,
            search_by_extension=True
        )
        all_tif_files.extend([(folder, tif_file) for tif_file in tif_files])

    st.write(f"Total `.tif` files found: **{len(all_tif_files)}**")

    # If no `.tif` files are found, stop further processing
    if not all_tif_files:
        st.warning("No `.tif` files found for processing.")
        return

    # Multi-select for file selection
    default_selected = [f"{folder}/{os.path.basename(tif_file)}" for folder, tif_file in all_tif_files]
    selected_files = st.multiselect(
        "Select `.tif` files to process:",
        options=default_selected,
        default=default_selected,
        help="Choose the `.tif` files you want to process. All files are selected by default."
    )

    # Extract selected file paths
    files_to_process = [
        (folder, tif_file) for folder, tif_file in all_tif_files
        if f"{folder}/{os.path.basename(tif_file)}" in selected_files
    ]

    # Display table of selected files
    st.write("### Files to be Processed")
    if files_to_process:
        selected_files_df = pd.DataFrame(
            {"Folder": [folder for folder, _ in files_to_process],
             "File Name": [os.path.basename(tif_file) for _, tif_file in files_to_process]}
        )
        st.dataframe(selected_files_df)

        # Confirmation button to start processing
        if st.button("Start Processing"):
            st.success(f"Starting processing of {len(files_to_process)} file(s).")

            # Progress bars
            folder_progress = st.progress(0, text="Folder Progress")
            file_progress = st.progress(0, text="File Progress")

            # Create a placeholder for file-specific logs
            file_log = st.empty()

            # Process each selected file
            for folder_idx, (folder, tif_file) in enumerate(files_to_process):
                # Update file log placeholder with the current file being processed
                file_log.info(f"Processing file: **{os.path.basename(tif_file)}** in folder: **{folder}**")

                # Update file progress bar
                file_progress.progress((folder_idx + 1) / len(files_to_process), text=f"File {folder_idx + 1} of {len(files_to_process)}")

                # Preprocessing: Image and ROI selection
                st.write("Step 1: Preprocessing image and selecting ROIs...")
                ROIs, metrics = preprocessing_image_selection(tif_file, config_path, num_ROIs=16)
                save_csv_file(folder, metrics, f'{os.path.basename(tif_file).split(".")[0]}_ridge_metrics.csv')

                # Processing ROIs
                st.write("Step 2: Processing and saving ROIs...")
                output_folder = process_and_save_rois(tif_file, ROIs)

                # Running SOAC analysis
                st.write("Step 3: Running SOAC analysis...")
                snakes, junctions = soac_api(output_folder, parameter_file, executable_path, folder)

                # Analytics pipeline
                st.write("Step 4: Running analytics pipeline...")
                snakes = soac_analytics_pipeline(snakes, junctions)

                # Save results
                save_csv_file(folder, snakes, f'{os.path.basename(tif_file).split(".")[0]}_soac_results.csv')

                # Indicate successful processing of the current file
                file_log.success(f"File processed successfully: {os.path.basename(tif_file)}")

                # Update folder progress bar
                folder_progress.progress((folder_idx + 1) / len(files_to_process), text=f"Processed {folder_idx + 1} of {len(files_to_process)} files.")

            st.success("SOAC analysis completed for all selected files!")
    else:
        st.warning("No files selected for processing.")
