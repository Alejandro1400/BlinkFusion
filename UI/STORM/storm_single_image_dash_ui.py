import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px 
from Data_access.storm_db import STORMDatabaseManager
from UI.STORM.dashboard.comparison.storm_comp_analysis import comparison_analysis #, time_series_comparison
from UI.STORM.dashboard.comparison.storm_metrics_analysis import display_blinking_statistics, metrics_metadata_merge
from UI.STORM.dashboard.comparison.storm_filters import apply_selected_filters, display_filtered_metadata, fetch_and_display_filtered_data, get_pre_metrics, load_storm_metadata, select_filter_columns
from UI.STORM.dashboard.image_statistics.visualizations import display_histograms_image, display_time_series_image



class STORMSingleImageDashboard:
    def __init__(self, storm_folder):
        """
        Initializes the STORMDashboard.

        Args:
            storm_folder (str): Path to the STORM database folder.
        """
        self.storm_folder = storm_folder
        self.database = STORMDatabaseManager()

    def run_storm_si_dashboard_ui(self):
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

            experiment_paths = list({entry["Experiment"] for entry in metadata_analysis.values()})
            selected_experiment = st.selectbox(
                "Select Image",
                experiment_paths,
                index=0,
                help="Choose an image by its identifier to analyze its time series data."
            )
            selected_id = next(
                (id_ for id_, entry in metadata_analysis.items() if entry["Experiment"] == selected_experiment),
                None  # default if not found
            )

            # Retrieve unique experiment IDs
            experiment_ids = [selected_id] if selected_id else []

            # Initialize session state only once
            if "load_metrics" not in st.session_state:
                st.session_state.load_metrics_si = False
                st.session_state.grouped_molecules_si = None
                st.session_state.time_series_dict_si = None

            # Button to trigger loading
            if st.button(f"Load Molecules and Time Series Metrics for {selected_experiment}"):
                grouped_molecules, time_series_dict = get_pre_metrics(experiment_ids)
                st.session_state.grouped_molecules_si = grouped_molecules
                st.session_state.time_series_dict_si = time_series_dict
                st.session_state.load_metrics_si = True

            # Show success message and use cached values from session_state
            if st.session_state.load_metrics_si:
                st.success("Molecules and time series metrics loaded.")
                grouped_molecules = st.session_state.grouped_molecules_si
                time_series_dict = st.session_state.time_series_dict_si


        # Metrics table comparison
        metridata, metrics_columns = metrics_metadata_merge(grouped_molecules, time_series_dict, metadata_analysis, metadata_df)

        st.markdown("___")
        st.write(f"### Image Statistics for {selected_experiment}")

        selected_molecules = grouped_molecules[selected_id]
        selected_timeseries_original = time_series_dict[selected_id]  # Keep original safe
        selected_timeseries = selected_timeseries_original.copy()     # Work on a copy
        selected_metridata = metridata[metridata['Experiment ID'] == selected_id].drop(columns='Experiment ID').set_index('Experiment')

        image_metadata_dict = selected_metridata.iloc[0].to_dict()
        # Extract Quasi-Equilibrium (QE) parameters and metadata from the dict
        qe_start, qe_end = map(int, image_metadata_dict['QE Period (s)'].split('-'))
        frames = image_metadata_dict['Frames']
        exp = image_metadata_dict['Exposure']
        frame_rate_inv = exp / 1000
        selected_timeseries['End Frame'] = (selected_timeseries['End Frame'] * frame_rate_inv).round()
        selected_timeseries['End Frame'] = (selected_timeseries['End Frame'] / 10).round() * 10 
        selected_timeseries = selected_timeseries.rename(columns={'End Frame': 'index'}).drop(columns=['Start Frame'])
        qe_dc = image_metadata_dict['QE Duty Cycle']
        qe_sf = image_metadata_dict['QE Survival Fraction']

        
        with st.expander("Time Series Analysis", expanded=True):
            """
            Image-specific analysis section for visualizing time series and histogram data.
            Users can select an image, analyze time series trends, and generate histograms.
            """

            display_time_series_image(
                selected_timeseries=selected_timeseries,
                qe_start=qe_start,
                qe_end=qe_end,
                qe_dc=qe_dc,
                qe_sf=qe_sf
            )
        
        with st.expander("Histogram Analysis", expanded=True):

            selected_option1, classification = display_histograms_image(
                selected_molecules=selected_molecules,
                selected_metridata=selected_metridata,
                qe_start=qe_start,
                qe_end=qe_end,
                frames=frames,
                exp=exp
            )

        with st.expander("Blinking Classification", expanded=True):

            # Prepare the classification keys and their IDs
            classification_keys = [
                "Blinks On Once", "Blinks On Mult. Times", "Blinks Off Once",
                "Blinks Off Mult. Times"
            ]
            classification_ids = ["A", "B", "C", "D"]  # IDs for each classification
            classification_mapping = dict(zip(classification_keys, classification_ids))

            # Grid layout (2x2 or adjust as needed)
            grid_cols = 2
            total_keys = len(classification_keys)
            rows = (total_keys + grid_cols - 1) // grid_cols  # Calculate the number of rows

            for row in range(rows):
                cols = st.columns(grid_cols)

                for col_idx, key in enumerate(classification_keys[row * grid_cols:(row + 1) * grid_cols]):
                    with cols[col_idx]:
                        # Display classification title
                        st.subheader(f"{key} ({len(classification[key])} Molecules)")

                        if classification[key]:  # If there are molecules in this category
                            # Generate molecule options with IDs
                            molecule_options = [
                                f"{classification_mapping[key]}{idx + 1}" for idx in range(len(classification[key]))
                            ]
                            selected_molecule_id = st.selectbox(
                                f"Select Molecule for {key}:",
                                options=molecule_options,
                                key=f"{key}_select",
                                help=f"Choose a molecule ID to display its intensity profile for {key}."
                            )

                            # Extract selected molecule data
                            if selected_molecule_id:
                                molecule_index = int(selected_molecule_id[1:]) - 1
                                molecule_data = classification[key][molecule_index]

                                # Display photobleaching and duty cycle side by side (above intensity profile)
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    dc = molecule_data.get("Duty Cycle", None)
                                    if isinstance(dc, (int, float)):
                                        st.write(f"**Duty Cycle**: {dc:.5f}")
                                    else:
                                        st.write("**Duty Cycle**: N/A")
                                with col2:
                                    st.write(f"**Switching Cycles**: {len(molecule_data['Tracks'])}")
                                with col3:
                                    st.write(f"**{molecule_data.get('Bleaching', 'N/A')}**")

                                # Retrieve the tracks for the selected molecule from classification data
                                tracks_ids = molecule_data["Tracks"]

                                # Filter the localizations using the retrieved tracks
                                selected_localizations_data = self.database.get_localizations_by_tracks(tracks_ids)

                                st.write(f"**Number of Localizations**: {len(selected_localizations_data)}")

                                # ---------------------------------------------------------------------
                                # FIX STARTS HERE: make FRAME unique BEFORE reindexing
                                # ---------------------------------------------------------------------

                                # Keep only the columns we need
                                df = selected_localizations_data[["FRAME", "INTENSITY"]].copy()

                                # Aggregate intensity so each FRAME appears only once
                                # Choose aggregation: mean (recommended), sum, or max
                                plot_data = (
                                    df.groupby("FRAME", as_index=True)["INTENSITY"]
                                    .mean()
                                    .sort_index()
                                    .to_frame()
                                )

                                # Choose the frame range (continuous x-axis)
                                if selected_option1 == "Quasi-Equilibrium Population":
                                    frame_range = np.arange(
                                        int(qe_start * (1000 / exp)),
                                        int(qe_end * (1000 / exp)) + 1,
                                        1
                                    )
                                else:
                                    frame_range = np.arange(0, frames + 1, 1)

                                # Reindex to include missing frames as NaN
                                plot_data = plot_data.reindex(frame_range)

                                # Interpolate ONLY one-frame gaps (single missing frame)
                                plot_data["INTENSITY"] = plot_data["INTENSITY"].interpolate(limit=1)

                                # Convert frames to time (seconds)
                                plot_data["TIME"] = plot_data.index * (exp / 1000)

                                # Reset index for Plotly (keeps TIME and INTENSITY as columns)
                                plot_data = plot_data.reset_index(drop=True)

                                # ---------------------------------------------------------------------
                                # FIX ENDS HERE
                                # ---------------------------------------------------------------------

                                # Plot intensity profile with Plotly as a bar chart
                                fig = px.bar(
                                    plot_data,
                                    x="TIME",
                                    y="INTENSITY",
                                    title=f"Intensity Profile for Molecule ID: {selected_molecule_id}",
                                    labels={"TIME": "Time (s)", "INTENSITY": "Intensity (Photons)"},
                                    template="plotly_white"
                                )

                                # Customize the layout and bar appearance
                                fig.update_traces(
                                    marker=dict(color="blue", line=dict(width=1, color="darkblue")),
                                    width=0.02
                                )
                                fig.update_layout(
                                    xaxis_title="Time (s)",
                                    yaxis_title="Intensity (Photons)",
                                    xaxis=dict(range=(plot_data["TIME"].min(), plot_data["TIME"].max())),
                                    hovermode="x unified",
                                    title_font=dict(size=16),
                                    font=dict(size=14)
                                )

                                st.plotly_chart(fig, use_container_width=True)

                        else:
                            st.write("No molecules")

