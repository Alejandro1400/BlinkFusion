import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px 
from Data_access.storm_db import STORMDatabaseManager
from UI.STORM.dashboard.comparison.storm_comp_analysis import comparison_analysis #, time_series_comparison
from UI.STORM.dashboard.comparison.storm_metrics_analysis import display_blinking_statistics, metrics_metadata_merge
from UI.STORM.dashboard.comparison.storm_filters import apply_selected_filters, display_filtered_metadata, fetch_and_display_filtered_data, get_pre_metrics, load_storm_metadata, select_filter_columns
from UI.STORM.dashboard.image_statistics.visualizations import display_histograms_image, display_time_series_image



class STORMDashboard:
    def __init__(self, storm_folder):
        """
        Initializes the STORMDashboard.

        Args:
            storm_folder (str): Path to the STORM database folder.
        """
        self.storm_folder = storm_folder
        self.database = STORMDatabaseManager()

    def run_storm_dashboard_ui(self):
        """
        Displays the PulseSTORM Dashboard UI for filtering, analyzing, and visualizing STORM data.

        Args:
            pulseSTORM_folder (str): Path to the folder containing PulseSTORM analysis data.
        """

        # Load unique metadata values and number of experiments with molecules
        metadata_values = load_storm_metadata(self.database)

        desc_columns = ["Experiment"] + list(metadata_values.keys())

        with st.expander("Filter Options", expanded=True):
            """
            Allows users to filter the metadata based on specific columns and values. 
            Updates other datasets (localizations, tracks, etc.) based on the filtered metadata.
            """
            selected_filter_columns = select_filter_columns(metadata_values)
            selected_filters = apply_selected_filters(selected_filter_columns, metadata_values)
            metadata_analysis = fetch_and_display_filtered_data(self.database, selected_filters)
            metadata_df, display_columns = display_filtered_metadata(metadata_analysis, selected_filter_columns)

            # Retrieve unique experiment IDs
            experiment_ids = list(metadata_analysis.keys())
            st.success(f"Number of experiments retrieved: {len(experiment_ids)}")  # Display the count

            # Initialize session state only once
            if "load_metrics" not in st.session_state:
                st.session_state.load_metrics = False
                st.session_state.grouped_molecules = None
                st.session_state.time_series_dict = None

            # Button to trigger loading
            if st.button("Load Molecules and Time Series Metrics"):
                grouped_molecules, time_series_dict = get_pre_metrics(experiment_ids)
                st.session_state.grouped_molecules = grouped_molecules
                st.session_state.time_series_dict = time_series_dict
                st.session_state.load_metrics = True

            # Show success message and use cached values from session_state
            if st.session_state.load_metrics:
                st.success("Molecules and time series metrics loaded.")
                grouped_molecules = st.session_state.grouped_molecules
                time_series_dict = st.session_state.time_series_dict

        # Calculate and display metrics
        st.markdown("___")
        st.write("### Metrics and Analysis")


        # Metrics table comparison
        metridata, metrics_columns = metrics_metadata_merge(grouped_molecules, time_series_dict, metadata_analysis, metadata_df)
        display_blinking_statistics(metridata, desc_columns, metrics_columns, grouped_molecules, time_series_dict, metadata_analysis)

        # Metrics and Time Series Graphs Comparison
        with st.expander("Comparison Analysis", expanded=True):
            comparison_analysis(metridata, desc_columns, metrics_columns)
            #time_series_comparison(time_series_dict, metadata_analysis)