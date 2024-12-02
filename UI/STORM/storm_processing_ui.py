import os
import pandas as pd
import streamlit as st

from Analysis.STORM.molecule_merging import process_tracks
from Data_access.file_explorer import find_items, find_valid_folders, save_csv_file


def run_storm_processing_ui(storm_folder):
    """
    Streamlit UI for processing STORM analysis on selected folders and files.

    Args:
        storm_folder (str): Path to the folder containing data for STORM analysis.
    """
    if storm_folder is None or storm_folder == "":
        st.error("STORM database folder not set. Please configure it in the sidebar.")
        return
    
    st.info("This interface allows you to process STORM analysis for all valid `.tif` and corresponding `locs.csv` files.")

    # Find all valid folders
    with st.spinner("Searching for valid folders..."):
        valid_folders = find_valid_folders(
            storm_folder,
            required_files={'.tif', 'locs.csv'},
            exclude_files={'locs_blink_stats.csv', 'track_blink_stats.csv', 'mol_blink_stats.csv'}
        )
    st.write(f"Found **{len(valid_folders)}** valid folders for analysis.")

    # Check if no valid folders were found
    if not valid_folders:
        st.warning("No valid folders found. Please check the folder structure and contents.")
        return

    # Collect all `.tif` and corresponding `locs.csv` files across valid folders
    all_files = []
    for folder in valid_folders:
        tif_files = find_items(
            base_directory=folder,
            item='.tif',
            is_folder=False,
            check_multiple=True,
            search_by_extension=True
        )
        for tif_file in tif_files:
            czi_filename = os.path.basename(tif_file).split(".tif")[0]
            tm_file = find_items(
                base_directory=folder,
                item=f'{czi_filename}_locs.csv',
                is_folder=False,
                check_multiple=False,
                search_by_extension=False
            )
            if tm_file:  # Ensure `locs.csv` exists for each `.tif`
                all_files.append((folder, tif_file, tm_file))

    st.write(f"Total valid `.tif` and `locs.csv` file pairs found: **{len(all_files)}**")

    # If no valid files are found, stop further processing
    if not all_files:
        st.warning("No valid file pairs found for processing.")
        return

    # Multi-select for file selection
    default_selected = [f"{folder}/{os.path.basename(tif_file)}" for folder, tif_file, _ in all_files]
    selected_files = st.multiselect(
        "Select `.tif` files to process:",
        options=default_selected,
        default=default_selected,
        help="Choose the `.tif` files you want to process. Corresponding `locs.csv` files will be automatically selected."
    )

    # Extract selected file paths
    files_to_process = [
        (folder, tif_file, tm_file) for folder, tif_file, tm_file in all_files
        if f"{folder}/{os.path.basename(tif_file)}" in selected_files
    ]

    # Display table of selected files
    st.write("### Files to be Processed")
    if files_to_process:
        selected_files_df = pd.DataFrame(
            {"Folder": [folder for folder, _, _ in files_to_process],
             "TIF File": [os.path.basename(tif_file) for _, tif_file, _ in files_to_process],
             "Locs CSV File": [os.path.basename(tm_file) for _, _, tm_file in files_to_process]}
        )
        st.dataframe(selected_files_df)

        # Confirmation button to start processing
        if st.button("Start Processing"):
            st.success(f"Starting processing of {len(files_to_process)} file pair(s).")

            # Progress bars
            folder_progress = st.progress(0, text="Folder Progress")
            file_progress = st.progress(0, text="File Progress")

            # Create a placeholder for file-specific logs
            file_log = st.empty()

            # Process each selected file pair
            for file_idx, (folder, tif_file, tm_file) in enumerate(files_to_process):
                # Update file log placeholder with the current file being processed
                file_log.info(f"Processing file: **{os.path.basename(tif_file)}** and **{os.path.basename(tm_file)}** in folder: **{folder}**")

                # Update file progress bar
                file_progress.progress((file_idx + 1) / len(files_to_process), text=f"File {file_idx + 1} of {len(files_to_process)}")

                # Read the `locs.csv` file
                st.write("Step 1: Reading localizations file...")
                df = pd.read_csv(tm_file)

                # Process tracks and generate statistics
                st.write("Step 2: Processing tracks and generating statistics...")
                localizations, tracks, molecules = process_tracks(df, 'thunderstorm')

                # Save results
                st.write("Step 3: Saving results...")
                save_csv_file(folder, localizations, f'{os.path.basename(tm_file).split(".c")[0]}_locs_blink_stats.csv')
                save_csv_file(folder, tracks, f'{os.path.basename(tm_file).split(".c")[0]}_track_blink_stats.csv')
                save_csv_file(folder, molecules, f'{os.path.basename(tm_file).split(".c")[0]}_mol_blink_stats.csv')

                # Indicate successful processing of the current file pair
                file_log.success(f"File processed successfully: {os.path.basename(tif_file)} and {os.path.basename(tm_file)}")

                # Update folder progress bar
                folder_progress.progress((file_idx + 1) / len(files_to_process), text=f"Processed {file_idx + 1} of {len(files_to_process)} files.")

            st.success("STORM analysis completed for all selected files!")
    else:
        st.warning("No files selected for processing.")
