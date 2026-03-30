import os
import time
import streamlit as st
import pandas as pd

from Analysis.STORM.Calculator.time_series_calculator import TimeSeriesCalculator
from Analysis.STORM.molecule_merging import MoleculeTracker
from Data_access.storm_db import STORMDatabaseManager

import json
import os
import time
import pandas as pd
import streamlit as st


class STORMProcessor:
    """
    A class to manage the processing of STORM files, handling database queries,
    file selection, and processing logic through a Streamlit UI.
    """

    def __init__(self, storm_folder, config_file=None):
        """
        Initializes the STORMProcessor.

        Args:
            storm_folder (str): Path to the STORM database folder.
            config_file (str, optional): Path to STORM processing config JSON.
        """
        self.storm_folder = storm_folder
        self.database = STORMDatabaseManager()

        # Default STORM tracking parameters
        self.min_frames_locs_to_tracks = 3
        self.max_gaps_locs_to_tracks = 2
        self.max_distance_merge_localizations = 200
        self.max_distance_merge_molecules = 100
        self.time_series_interval_seconds = 50

        try:
            # If user does not provide config path → use repo default
            if config_file is None:
                current_dir = os.path.dirname(os.path.abspath(__file__))
    
                config_file = os.path.join(
                    current_dir,
                    "..",
                    "Data_access",
                    "storm_merge_param.json"
                )
    
                config_file = os.path.normpath(config_file)
    
            # Check if file exists
            if not os.path.exists(config_file):
                raise FileNotFoundError(f"Config file not found: {config_file}")
    
            print(f"Using STORM config file: {config_file}")
    
            # Load config
            self._load_storm_tracking_config(config_file)
    
        except Exception as e:
            print(f"⚠️ Could not load STORM config. Using default parameters. Reason: {e}")

    def _load_storm_tracking_config(self, config_file):
        """
        Loads STORM tracking parameters from a JSON configuration file.
        """
        try:
            with open(config_file, "r") as f:
                config = json.load(f)

            params = config.get("storm_tracking_parameters", {})

            self.min_frames_locs_to_tracks = int(
                params.get("min_frames_locs_to_tracks", self.min_frames_locs_to_tracks)
            )
            self.max_gaps_locs_to_tracks = int(
                params.get("max_gaps_locs_to_tracks", self.max_gaps_locs_to_tracks)
            )
            self.max_distance_merge_localizations = float(
                params.get("max_distance_merge_localizations", self.max_distance_merge_localizations)
            )
            self.max_distance_merge_molecules = float(
                params.get("max_distance_merge_molecules", self.max_distance_merge_molecules)
            )
            self.time_series_interval_seconds = float(
                params.get("time_series_interval_seconds", self.time_series_interval_seconds)
            )

        except Exception as e:
            print(f"Could not load STORM tracking config. Using defaults. Reason: {e}")

    def find_unprocessed_files(self):
        """
        Finds TIFF files that have not yet been processed into localizations.

        Returns:
            dict: A dictionary where each key is a `experiment_id`, and the value contains:
                  - 'Folder': Path to the folder containing the files.
                  - 'Tif File': Full path to the `.tif` file.
                  - 'Locs File': Full path to the `_locs.csv` file.
        """
        unprocessed_folders = self.database.storm_folders_without_localizations()
        all_files = {}

        for folder in unprocessed_folders:
            folder_path = folder['folder_path']
            experiment_id = folder['experiment_id']

            print(f"Processing folder: {folder_path}")
            print(f"Experiment ID: {experiment_id}")

            # Normalize the folder path to ensure there are no issues with relative paths or ".."
            folder_path = os.sep.join(part for part in folder_path.split(os.sep) if part != "..")
            print(f"Normalized folder path: {folder_path}")
            full_folder_path = os.path.dirname(os.path.join(self.storm_folder, folder_path))
            print(f"Full folder path: {full_folder_path}")

            # Assume the base name of the file is the last part of the path without the file extension
            base_name = os.path.basename(folder_path).rsplit('.', 1)[0]  # Correctly split off the file extension
            print(f"Base name for files: {base_name}")

            # Construct full paths to the TIFF and localization CSV files
            tif_file = os.path.join(full_folder_path, f"{base_name}.tif")
            locs_file = os.path.join(full_folder_path, f"{base_name}_locs.csv")

            # Debugging prints to check paths
            print(f"TIF file path: {tif_file}")
            print(f"Locs file path: {locs_file}")

            print(os.path.exists(tif_file), os.path.exists(locs_file))

            # Check if both files exist
            if os.path.exists(tif_file) and os.path.exists(locs_file):
                all_files[str(experiment_id)] = {  # Convert ObjectId to string if needed
                    "Folder": full_folder_path,
                    "Tif File": tif_file,
                    "Locs File": locs_file
                }

        return all_files
    

    def process_selected_files(self, selected_files):
        """
        Processes selected files, generating molecules from STORM analysis.

        Args:
            selected_files (list): List of tuples containing folder paths and metadata IDs.
        """
        # Progress bars
        folder_progress = st.progress(0, text="Overall Progress")
        file_log = st.empty()
        step_log = st.empty()

        for file_idx, (folder, tif_file, locs_file, experiment_id) in enumerate(selected_files):
            file_name = os.path.basename(tif_file)
            file_log.info(f"Processing file: **{file_name}**")

            total_frames, exposure_time = self.database.get_experiment_settings(experiment_id)

            # Read localization file
            start_time = time.time()
            df = pd.read_csv(locs_file)
            elapsed_time = time.time() - start_time
            num_localizations = len(df)
            step_log.write(f"✅ Step 1: Read localizations file ({num_localizations} localizations) - {elapsed_time:.2f} sec")


            # Step 2: Generating molecules
            step_log.write("🔄 Step 2: Generating molecules...")
            moltracker = MoleculeTracker(
                df,
                'thunderstorm',
                min_frames=self.min_frames_locs_to_tracks,
                max_gaps=self.max_gaps_locs_to_tracks,
                localization_max_distance=self.max_distance_merge_localizations,
                max_distance=self.max_distance_merge_molecules
            )
            molecules = moltracker.process_tracks(step_log)

            # Step 3: Saving molecules to database
            step_log.write("🔄 Step 3: Saving molecules to database...")
            start_time = time.time()
            self.database.save_molecules(molecules, experiment_id)
            elapsed_time = time.time() - start_time
            step_log.write(f"✅ Step 3: Saved molecules to database - {elapsed_time:.2f} sec")


            # Step 4: Computing time series
            step_log.write("⏳ Step 4: Computing time series...")
            time_series_calculator = TimeSeriesCalculator(molecules, interval=self.time_series_interval_seconds, total_frames=total_frames, exposure_time=exposure_time)
            time_series_df = time_series_calculator.calculate_time_series_metrics()
            
            self.database.save_time_series(time_series_df, experiment_id)

            file_log.success(f"✅ Processed: {file_name}")
            folder_progress.progress((file_idx + 1) / len(selected_files), text=f"Processed {file_idx + 1} of {len(selected_files)} files.")

            # Clear step logs for the next file
            step_log.empty()

        st.success("🎉 STORM analysis completed for all selected files!")


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
            (info["Folder"], info["Tif File"], info["Locs File"], experiment_id)
            for experiment_id, info in all_files.items()
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
            (folder, tif_file, locs_file, experiment_id)
            for folder, tif_file, locs_file, experiment_id in files_to_process
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
