import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
from Analysis.STORM.analytics_storm import calculate_frequency
from Dashboard.graphs import plot_histograms, plot_time_series_interactive
from Data_access.storm_db import STORMDatabaseManager
from UI.STORM.dashboard.storm_comp_analysis import comparison_analysis, time_series_comparison
from UI.STORM.dashboard.storm_metrics_analysis import display_blinking_statistics, metrics_metadata_merge
from UI.STORM.dashboard.storm_filters import apply_selected_filters, display_filtered_metadata, fetch_and_display_filtered_data, get_pre_metrics, load_storm_metadata, select_filter_columns



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

            # Fetch datasets related to experiment IDs
            grouped_molecules, time_series_dict = get_pre_metrics(experiment_ids)

        # Calculate and display metrics
        st.markdown("___")
        st.write("### Metrics and Analysis")


        # Metrics table comparison
        metridata, metrics_columns = metrics_metadata_merge(grouped_molecules, time_series_dict, metadata_analysis, metadata_df)
        display_blinking_statistics(metridata, desc_columns, metrics_columns, grouped_molecules, time_series_dict, metadata_analysis)

        # Metrics and Time Series Graphs Comparison
        with st.expander("Comparison Analysis", expanded=True):
            comparison_analysis(metridata, desc_columns, metrics_columns)
            time_series_comparison(time_series_dict, metadata_analysis)

        with st.expander("Image Statistics", expanded=True):
            """
            Image-specific analysis section for visualizing time series and histogram data.
            Users can select an image, analyze time series trends, and generate histograms.
            """

            st.subheader("Time Series Analysis")

            # Time series columns excluding 'IDENTIFIER'
            time_series_columns = timeseries.columns.drop('IDENTIFIER')

            # Dropdown for selecting an image
            selected_id = st.selectbox(
                "Select Image",
                metadata_analysis['IDENTIFIER'].unique(),
                index=0,
                help="Choose an image by its identifier to analyze its time series data."
            )

            # Create two columns for selecting Y1 and Y2 axes
            col1, col2 = st.columns(2)

            with col1:
                selected_y1_label = st.selectbox(
                    'Y1 Axis:',
                    options=list(time_series_columns),
                    key='y1_image_select',
                    index=0,
                    help="Choose the primary metric for the Y1 axis."
                )

            with col2:
                available_legends = [col for col in time_series_columns if col != 'IDENTIFIER' and col != selected_y1_label]
                selected_y2_label = st.selectbox(
                    'Y2 Axis:',
                    options=list(available_legends),
                    key='y2_image_select',
                    index=0,
                    help="Choose the secondary metric for the Y2 axis."
                )

            # Filter data for the selected image
            selected_localizations = locs_analysis[locs_analysis['IDENTIFIER'] == selected_id]
            selected_tracks = tracks_analysis[tracks_analysis['IDENTIFIER'] == selected_id]
            selected_molecules = molecules_analysis[molecules_analysis['IDENTIFIER'] == selected_id]
            selected_metadata = metadata_analysis[metadata_analysis['IDENTIFIER'] == selected_id]
            selected_timeseries = timeseries_analysis[timeseries_analysis['IDENTIFIER'] == selected_id]
            selected_metrics = metrics[metrics['IDENTIFIER'] == selected_id]

            # Extract Quasi-Equilibrium (QE) parameters and metadata
            qe_start, qe_end = map(int, selected_metrics['QE Period (s)'].iloc[0].split('-'))
            frames = selected_metadata['FRAMES'].iloc[0]
            exp = selected_metadata['EXPOSURE'].iloc[0]
            qe_dc, qe_sf = selected_metrics['QE Duty Cycle'].iloc[0], selected_metrics['QE Survival Fraction'].iloc[0]

            # Display metadata as a DataFrame
            st.dataframe(selected_metadata.set_index('IDENTIFIER'), height=150)

            # Prepare time series data for plotting
            selected_columns = ['index', selected_y1_label, selected_y2_label]
            if not selected_timeseries.empty:
                time_metric = selected_timeseries[selected_columns].set_index('index').sort_index()

                # Align time indices to nearest 10 for cleaner visualization
                time_metric.index = time_metric.index.map(lambda x: int(np.ceil(x / 10) * 10))

                # Create and display the time series plot
                fig = plot_time_series_interactive(
                    metric1_data=time_metric[selected_y1_label],
                    metric2_data=time_metric[selected_y2_label],
                    metric1_name=selected_y1_label,
                    metric2_name=selected_y2_label,
                    qe_start=qe_start,
                    qe_end=qe_end,
                    qe_dc=qe_dc,
                    qe_sf=qe_sf
                )
                st.plotly_chart(fig, use_container_width=True)

                # Display time series data as a DataFrame
                st.write("Time Series Data")
                st.dataframe(time_metric, use_container_width=True, height=200)

            st.markdown("___")

            st.subheader("Histogram Analysis")

            # Create four columns for user controls
            col1, col2, col3, col4 = st.columns(4)

            # Option to select population type (Whole Population or QE Population)
            with col1:
                selected_option1 = st.radio(
                    "Select the tracks frequency to display",
                    ["Whole Population", "Quasi-Equilibrium Population"],
                    help="Choose whether to analyze the whole population or only the quasi-equilibrium population."
                )

            # Option to group data by molecule or track
            with col2:
                selected_option2 = st.radio(
                    "Select the grouping to display",
                    ["By Molecule", "By Track"],
                    help="Choose to group the data by molecules or tracks for histogram generation."
                )

            # Slider for adjusting the number of bins in histograms
            with col3:
                num_bins = st.slider(
                    "Number of Bins",
                    min_value=5,
                    max_value=50,
                    value=20,
                    help="Adjust the number of bins for the histograms."
                )

            # Checkbox for removing outliers
            with col4:
                remove_outliers = st.checkbox(
                    "Remove Outliers",
                    value=False,
                    help="Enable to exclude outliers from the histograms."
                )

            # Convert options to parameters for histogram calculation
            population_type = 'quasi' if selected_option1 == "Quasi-Equilibrium Population" else 'whole'
            grouping_type = 'molecule' if selected_option2 == "By Molecule" else 'track'

            # Calculate frequencies for histograms
            duty_cycle, photons, switching_cycles, on_time, classification = calculate_frequency(
                selected_qe_tracks=selected_tracks,
                selected_qe_molecules=selected_molecules,
                qe_start=qe_start,
                qe_end=qe_end,
                frames=frames,
                exp=exp,
                population=population_type,
                metric=grouping_type
            )

            # Generate and display histograms
            plot_histograms(
                duty_cycle=duty_cycle,
                photons=photons,
                switching_cycles=switching_cycles,
                on_time=on_time,
                metrics=selected_metrics,
                remove_outliers=remove_outliers,
                num_bins=num_bins,
                metric_type=grouping_type
            )


            st.subheader("Blinking Classification")

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
                                    # Change the duty cycle to 5 decimal places
                                    st.write(f"**Duty Cycle**: {molecule_data.get('Duty Cycle', 'N/A'):.5f}")
                                with col2:
                                    st.write(f"**Switching Cycles**: {len(molecule_data['Tracks'])}")
                                with col3:
                                    st.write(f"**{molecule_data.get('Bleaching', 'N/A')}**")
                                
                                # Retrieve the tracks for the selected molecule from classification data
                                tracks_ids = molecule_data['Tracks']

                                # Filter the localizations using the retrieved tracks
                                selected_localizations_data = selected_localizations[
                                    selected_localizations['TRACK_ID'].isin(tracks_ids)
                                ]

                                # Prepare the data for plotting (initially using frames)
                                plot_data = pd.DataFrame({
                                    'INTENSITY': selected_localizations_data['INTENSITY [PHOTON]'].values,
                                    'FRAME': selected_localizations_data['FRAME'].values  # Use frames directly
                                })
                                plot_data = plot_data.set_index('FRAME').sort_index()

                                # Interpolate one-frame gaps
                                gaps = plot_data.index.to_series().diff().loc[lambda x: x == 2].index  # Gaps of one frame
                                for gap_start in gaps:
                                    if (gap_start - 1) in plot_data.index and (gap_start + 1) in plot_data.index:
                                        mean_intensity = (
                                            plot_data.at[gap_start - 1, 'INTENSITY'] +
                                            plot_data.at[gap_start + 1, 'INTENSITY']
                                        ) / 2
                                        plot_data.at[gap_start, 'INTENSITY'] = mean_intensity

                                # Reindex to ensure a continuous x-axis range
                                if selected_option1 == "Quasi-Equilibrium Population":
                                    frame_range = np.arange(int(qe_start*(1000/exp)), int(qe_end*(1000/exp)) + 1, 1)  # Range for QE
                                else:
                                    frame_range = np.arange(0, frames + 1, 1)  # Full frame range

                                plot_data = plot_data.reindex(frame_range, fill_value=np.nan)

                                # Convert frames to time (at the end)
                                plot_data['TIME'] = plot_data.index * (exp / 1000)  # Convert frames to seconds
                                plot_data = plot_data.reset_index(drop=True)  # Reset index for Plotly plotting

                                # Replace NaNs with 0 for plotting purposes
                                #plot_data = plot_data.fillna(0)

                                # Plot intensity profile with Plotly as a bar chart
                                fig = px.bar(
                                    plot_data,
                                    x='TIME',
                                    y='INTENSITY',
                                    title=f"Intensity Profile for Molecule ID: {selected_molecule_id}",
                                    labels={'TIME': 'Time (s)', 'INTENSITY': 'Intensity (Photons)'},
                                    template='plotly_white'
                                )

                                # Customize the layout and bar appearance
                                fig.update_traces(marker=dict(color='blue', line=dict(width=1, color='darkblue')), width=0.02)
                                fig.update_layout(
                                    xaxis_title="Time (s)",
                                    yaxis_title="Intensity (Photons)",
                                    xaxis=dict(range=(plot_data['TIME'].min(), plot_data['TIME'].max())),
                                    hovermode='x unified',
                                    title_font=dict(size=16),
                                    font=dict(size=14)
                                )

                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.write("No molecules")



    


    
