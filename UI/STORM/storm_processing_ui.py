import os
import streamlit as st
import pandas as pd

from Analysis.STORM.Calculator.time_series_calculator import TimeSeriesCalculator
from Analysis.STORM.molecule_merging import MoleculeTracker
from Data_access.storm_db import STORMDatabaseManager

class STORMProcessor:
    """
    A class to manage the processing of STORM files, handling database queries, file selection, 
    and processing logic through a Streamlit UI.
    """

    def __init__(self, storm_folder):
        """
        Initializes the STORMProcessor.

        Args:
            storm_folder (str): Path to the STORM database folder.
        """
        self.storm_folder = storm_folder
        self.database = STORMDatabaseManager(storm_folder)

    def find_unprocessed_files(self):
        """
        Finds TIFF files that have not yet been processed into localizations.

        Returns:
            dict: A dictionary where each key is a `metadata_id`, and the value contains:
                  - 'Folder': Path to the folder containing the files.
                  - 'Tif File': Full path to the `.tif` file.
                  - 'Locs File': Full path to the `_locs.csv` file.
        """
        unprocessed_folders = self.database.storm_folders_without_localizations()
        all_files = {}

        for metadata_id, folder_path in unprocessed_folders:
            full_folder_path = os.path.join(self.storm_folder, folder_path)
            base_name = os.path.basename(folder_path)

            tif_file = os.path.join(full_folder_path, f"{base_name}.tif")
            locs_file = os.path.join(full_folder_path, f"{base_name}_locs.csv")

            if os.path.exists(tif_file) and os.path.exists(locs_file):
                all_files[metadata_id] = {
                    "Folder": full_folder_path,
                    "Tif File": tif_file,
                    "Locs File": locs_file
                }

        return all_files

    def run_processing_ui(self):
        """
        Streamlit UI for selecting and processing STORM analysis.
        """
        if not self.storm_folder:
            st.error("STORM database folder not set. Please configure it in the sidebar.")
            return

        st.info("This interface allows you to process STORM analysis for `.tif` and `locs.csv` file pairs.")

        with st.spinner("Fetching unprocessed files from the database..."):
            all_files = self.find_unprocessed_files()

        st.write(f"Found **{len(all_files)}** valid TIFF + LOCS file pairs for processing.")

        if not all_files:
            st.warning("No unprocessed files found. Please check the database.")
            return

        # Convert dictionary to list for selection
        files_to_process = [
            (info["Folder"], info["Tif File"], info["Locs File"], metadata_id)
            for metadata_id, info in all_files.items()
        ]

        # UI for selecting files
        default_selected = [
            f"{folder}/{os.path.basename(tif_file)}"
            for folder, tif_file, _, _ in files_to_process
        ]

        selected_files = st.multiselect(
            "Select `.tif` files to process:",
            options=default_selected,
            default=default_selected,
            help="Choose `.tif` files. Corresponding `locs.csv` will be processed automatically."
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

            if st.button("Start Processing"):
                self.process_selected_files(selected_files_to_process)
        else:
            st.warning("No files selected for processing.")

    def process_selected_files(self, selected_files):
        """
        Processes selected files, generating molecules from STORM analysis.

        Args:
            selected_files (list): List of tuples containing folder paths and metadata IDs.
        """
        st.success(f"Starting processing of {len(selected_files)} file pair(s).")

        folder_progress = st.progress(0, text="Folder Progress")
        file_progress = st.progress(0, text="File Progress")
        file_log = st.empty()

        for file_idx, (folder, tif_file, locs_file, metadata_id) in enumerate(selected_files):
            file_name = os.path.basename(tif_file)
            file_log.info(f"Processing file: **{file_name}**")

            file_progress.progress((file_idx + 1) / len(selected_files), text=f"File {file_idx + 1} of {len(selected_files)}")

            # Read localization file
            st.write("Step 1: Reading localizations file...")
            df = pd.read_csv(locs_file)

            # Process to generate molecules only
            st.write("Step 2: Generating molecules...")
            moltracker = MoleculeTracker(df, 'thunderstorm')
            molecules = moltracker.process_tracks()

            st.write("Step 3: Saving molecules to database...")
            self.database.save_molecules(molecules=molecules, metadata_id=metadata_id)

            st.write("Step 4: Computing time series...")
            total_frames, exposure_time = self.database.get_experiment_settings(metadata_id)
            time_series_calculator = TimeSeriesCalculator(molecules, interval=50, total_frames=total_frames, exposure_time=exposure_time)
            time_series_df = time_series_calculator.calculate_time_series_metrics()

            st.write("Step 5: Saving time series to database...")
            self.database.save_time_series(time_series_df, metadata_id)

            file_log.success(f"✅ Processed: {file_name}")
            folder_progress.progress((file_idx + 1) / len(selected_files), text=f"Processed {file_idx + 1} of {len(selected_files)} files.")

        st.success("STORM analysis completed for all selected files!")