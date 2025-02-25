import os
import pandas as pd
import streamlit as st

from Analysis.STORM.molecule_merging import process_tracks
from Data_access.database_manager import DatabaseManager
from Data_access.file_explorer import save_csv_file



def find_unprocessed_files(storm_folder):
    """
    Finds folders with TIFF files but without localization data and constructs paths for processing.

    Args:
        storm_folder (str): The base folder where STORM files are stored.
        database_folder (str): The base folder where the database is stored.

    Returns:
        dict: A dictionary where each key is a `metadata_id` and the value contains:
              - 'Folder': Path to the folder containing the files.
              - 'Tif File': Full path to the `.tif` file.
              - 'Locs File': Full path to the `_locs.csv` file.
    """
    # Initialize database connection
    storm_db = DatabaseManager(storm_folder, "storm.db")

    # Retrieve folders without localization data
    unprocessed_folders = storm_db.get_folders_without_localizations()

    all_files = {}

    for metadata_id, folder_path in unprocessed_folders:
        # Construct the full folder path
        full_folder_path = os.path.join(storm_folder, folder_path)

        # Extract the base name to construct filenames
        base_name = os.path.basename(folder_path)

        # Construct paths for TIFF and localization files
        tif_file = os.path.join(full_folder_path, f"{base_name}.tif")
        locs_file = os.path.join(full_folder_path, f"{base_name}_locs.csv")

        # Check if both files exist
        if os.path.exists(tif_file) and os.path.exists(locs_file):
            all_files[metadata_id] = {
                "Folder": full_folder_path,
                "Tif File": tif_file,
                "Locs File": locs_file
            }

    # Close the database connection
    storm_db.close()

    return all_files


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

    # Retrieve all unprocessed files from the database
    with st.spinner("Fetching unprocessed files from the database..."):
        all_files = find_unprocessed_files(storm_folder)

    # Show the number of files found
    st.write(f"Found **{len(all_files)}** valid TIFF + LOCS file pairs for processing.")

    if not all_files or len(all_files) == 0:
        st.warning("No unprocessed files found. Please check the database.")
        return
    
    # Convert all_files dictionary to list of tuples for processing
    files_to_process = [
        (info["Folder"], info["Tif File"], info["Locs File"], metadata_id)
        for metadata_id, info in all_files.items()
    ]

    # Multi-select for file selection
    default_selected = [
        f"{folder}/{os.path.basename(tif_file)}"
        for folder, tif_file, _, _ in files_to_process
    ]
    
    selected_files = st.multiselect(
        "Select `.tif` files to process:",
        options=default_selected,
        default=default_selected,
        help="Choose the `.tif` files you want to process. Corresponding `locs.csv` files will be automatically selected."
    )

    # Filter only selected files
    selected_files_to_process = [
        (folder, tif_file, locs_file, metadata_id)
        for folder, tif_file, locs_file, metadata_id in files_to_process
        if f"{folder}/{os.path.basename(tif_file)}" in selected_files
    ]

    # Display selected files in a table
    st.write("### Files to be Processed")
    if selected_files_to_process:
        selected_files_df = pd.DataFrame(
            {
                "Folder": [folder for folder, _, _, _ in selected_files_to_process],
                "TIF File": [os.path.basename(tif_file) for _, tif_file, _, _ in selected_files_to_process],
                "Locs CSV File": [os.path.basename(locs_file) for _, _, locs_file, _ in selected_files_to_process],
            }
        )
        st.dataframe(selected_files_df)

        # Confirmation button to start processing
        if st.button("Start Processing"):
            st.success(f"Starting processing of {len(selected_files_to_process)} file pair(s).")

            # Progress bars
            folder_progress = st.progress(0, text="Folder Progress")
            file_progress = st.progress(0, text="File Progress")

            # Create a placeholder for file-specific logs
            file_log = st.empty()

            # Process each selected file pair
            for file_idx, (folder, tif_file, locs_file, metadata_id) in enumerate(selected_files_to_process):
                file_name = os.path.basename(tif_file)
                file_log.info(f"Processing file: **{file_name}**")

                # Update file progress bar
                file_progress.progress((file_idx + 1) / len(selected_files_to_process), text=f"File {file_idx + 1} of {len(selected_files_to_process)}")

                # Read the `locs.csv` file
                st.write("Step 1: Reading localizations file...")
                df = pd.read_csv(locs_file)

                # Process tracks and generate statistics
                st.write("Step 2: Processing tracks and generating statistics...")
                localizations, tracks, molecules = process_tracks(df, 'thunderstorm')

                # Save results
                st.write("Step 3: Saving results...")
                save_csv_file(folder, localizations, f'{os.path.basename(locs_file).split(".c")[0]}_locs_blink_stats.csv')
                save_csv_file(folder, tracks, f'{os.path.basename(locs_file).split(".c")[0]}_track_blink_stats.csv')
                save_csv_file(folder, molecules, f'{os.path.basename(locs_file).split(".c")[0]}_mol_blink_stats.csv')

                # Update the database to mark the file as processed
                storm_db = DatabaseManager(storm_folder, "storm.db")
                storm_db.mark_localizations_processed(metadata_id)
                storm_db.close()

                # Indicate successful processing of the current file pair
                file_log.success(f"✅ Processed: {file_name}")
                folder_progress.progress((file_idx + 1) / len(selected_files_to_process), text=f"Processed {file_idx + 1} of {len(selected_files_to_process)} files.")

            st.success("STORM analysis completed for all selected files!")
    else:
        st.warning("No files selected for processing.")
