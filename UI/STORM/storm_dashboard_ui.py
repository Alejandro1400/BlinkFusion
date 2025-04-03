from collections import defaultdict
import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

from Analysis.STORM.Calculator.molecule_metrics import MoleculeMetrics
from Analysis.STORM.Models.molecule import Molecule
from Analysis.STORM.Models.track import Track
from Analysis.STORM.analytics_storm import calculate_frequency
from Dashboard.graphs import plot_histograms, plot_intensity_vs_frame, plot_time_series_interactive
from Data_access.storm_db import STORMDatabaseManager



class STORMDashboard:
    def __init__(self, storm_folder):
        """
        Initializes the STORMDashboard.

        Args:
            storm_folder (str): Path to the STORM database folder.
        """
        self.storm_folder = storm_folder
        self.database = STORMDatabaseManager()


    def load_storm_metadata(self):
        """
        Load distinct metadata from the database where `tag='pulsestorm'`, but only for metadata entries
        that have associated molecules. Also, count the number of experiments that contain molecule data.

        Returns:
            tuple:
                - dict: Metadata dictionary where keys are metadata names and values are lists of unique values.
                - int: Number of files (experiments) that have at least one molecule.
        """
        # Aggregation pipeline to fetch distinct metadata with molecules
        pipeline = [
            {"$match": {
                "time_series": {"$exists": True, "$ne": []}  # Ensure the experiment has associated time series data
            }},
            {"$unwind": "$metadata"},  # Unwind metadata array
            {"$group": {
                "_id": "$metadata.name",
                "uniqueValues": {"$addToSet": "$metadata.value"}
            }},
            {"$sort": {"_id": 1}},  # Sort by metadata name
            {"$facet": {
                "metadataInfo": [
                    # Pass all previous grouped data
                    {"$project": {"_id": 1, "uniqueValues": 1}}
                ],
                "count": [
                    # Count the distinct experiment documents that have time series
                    {"$count": "numExperimentsWithTimeSeries"}
                ]
            }}
        ]

        metadata_result = list(self.database.experiments.aggregate(pipeline))
        

        database_metadata = {item['_id']: item['uniqueValues'] for item in metadata_result[0]['metadataInfo']}

        if metadata_result[0]['count']:
            num_experiments_with_molecules = metadata_result[0]['count'][0]['numExperimentsWithTimeSeries']
        else:
            num_experiments_with_molecules = 0

        return database_metadata, num_experiments_with_molecules
    

    def run_storm_dashboard_ui(self):
        """
        Displays the PulseSTORM Dashboard UI for filtering, analyzing, and visualizing STORM data.

        Args:
            pulseSTORM_folder (str): Path to the folder containing PulseSTORM analysis data.
        """

        # Load unique metadata values and number of experiments with molecules
        metadata_values, num_experiments = self.load_storm_metadata()

        with st.expander("Filter Options", expanded=True):
            """
            Allows users to filter the metadata based on specific columns and values. 
            Updates other datasets (localizations, tracks, etc.) based on the filtered metadata.
            """

            st.info("Apply filters to metadata columns. All data is included by default.")

            # Multiselect to choose metadata columns for filtering
            selected_filter_columns = st.multiselect(
                'Select columns to filter by:',
                options=list(metadata_values.keys()),
                key='filter_select',
                help="Choose metadata columns to filter the dataset."
            )

            # Initialize dictionary to store selected values for each column
            selected_filters = {}

            # Iterate through selected filter columns to allow filtering by values
            if selected_filter_columns:
                for col in selected_filter_columns:
                    unique_values = metadata_values[col]  # Get unique values from the DB query

                    selected_values = st.multiselect(
                        f"Filter {col}:",
                        options=unique_values,
                        default=unique_values,  # Default to all values
                        key=f'filter_{col}',
                        help=f"Select specific values for filtering the column '{col}'."
                    )

                    # Store selected values in the dictionary
                    selected_filters[col] = selected_values

            # Fetch metadata (apply filters if selected)
            if selected_filters:
                metadata_analysis = self.database.get_metadata(selected_filters)
            else:
                metadata_analysis = self.database.get_metadata()

            # Retrieve unique experiment IDs
            experiment_ids = list(metadata_analysis.keys())  

            st.success(f"Number of experiments retrieved: {len(experiment_ids)}")  # Display the count

            # Display filtered metadata
            st.markdown("___")
            st.write("### Data after Filtering:")
            st.write("Displaying unique files with their metadata values.")

            # Ensure selected columns include 'Experiment' (folder path)
            display_columns = ["Experiment"] + selected_filter_columns if selected_filter_columns else list(metadata_analysis.values())[0].keys()

            # Convert metadata dictionary to DataFrame for display
            if metadata_analysis:
                metadata_df = pd.DataFrame.from_dict(metadata_analysis, orient="index")
                metadata_df.index.name = "Experiment ID"  # Label index for clarity

                # Show filtered metadata
                st.dataframe(metadata_df[display_columns])

                # Display number of retrieved experiments
                st.success(f"Loaded {len(metadata_df)} experiment entries.")
            else:
                st.warning("No data found for the selected filters.")

            # Fetch datasets related to experiment IDs
            grouped_molecules = self.database.get_grouped_molecules_and_tracks(experiment_ids)
            # Print experiment IDs from grouped_molecules
            time_series_dict = self.database.get_grouped_time_series(experiment_ids)
            # Print experiment IDs from time_series_dict

        # Calculate and display metrics
        st.markdown("___")
        st.write("### Metrics and Analysis")

        try:
            molecule_metrics = MoleculeMetrics(grouped_molecules, time_series_dict, metadata_analysis)
            # Obtain molecular metrics based on filtered tracks and timeseries
            metrics = molecule_metrics.obtain_molecules_metrics()
            metrics_df = pd.DataFrame(metrics)

            # Add a column for the number of images and align it at the beginning
            metrics_df.insert(0, '# Images', len(metadata_analysis))
            metrics_columns = metrics_df.columns.drop('Experiment ID')

            metadata_df = pd.DataFrame.from_dict(metadata_analysis, orient='index').reset_index()
            metadata_df.rename(columns={'index': 'Experiment ID'}, inplace=True)

            # Merge metrics with metadata for context and further analysis
            if not metrics_df.empty and not metadata_df.empty:
                metridata = pd.merge(metadata_df, metrics_df, on='Experiment ID', how='inner')
                st.write(metridata)
                st.write("Metrics successfully calculated and merged with metadata.")
            else:
                st.warning("No metrics were calculated. Please check your data or filtering criteria.")
        except Exception as e:
            st.error(f"An error occurred while calculating metrics: {e}")


        with st.expander("Blinking Statistics", expanded=True):
            """
            Section for grouping metrics and displaying aggregated results. 
            Users can group data by selected columns and choose metrics to display.
            """

            # Create two columns for grouping and metric selection
            col1, col2 = st.columns(2)

            with col1:
                selected_group_columns = st.multiselect(
                    'Choose columns to group by (Be mindful of the hierarchical order):',
                    list(desc_columns),
                    key='group_by_select',
                    help="Select one or more metadata columns to group data hierarchically."
                )

            with col2:
                selected_metrics_columns = st.multiselect(
                    'Choose columns to display:',
                    list(metrics_columns),
                    key='metrics_select',
                    help="Select one or more metrics columns to display aggregated results."
                )

            # Display grouped data if metrics are available
            if not metrics.empty:
                # Combine selected group and metric columns
                selected_columns = selected_group_columns + selected_metrics_columns

                if not selected_columns:
                    selected_columns = metridata.columns.tolist()  # Default to all columns

                if not selected_group_columns:
                    selected_group_columns = ['IDENTIFIER']  # Default grouping column

                # Aggregate metrics based on group selections
                grouped_metrics = metridata.groupby(selected_group_columns).apply(aggregate_metrics)
                grouped_metrics = grouped_metrics[selected_metrics_columns]

                # Display the grouped metrics
                st.markdown("___")
                st.dataframe(grouped_metrics, height=300)

        with st.expander("Comparison Analysis", expanded=True):
            """
            Allows users to compare metrics using custom plots and visualizations.
            Users can configure axes, legends, and plot types dynamically.
            """

            st.subheader("Metrics Comparison")

            # Define columns for user inputs
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                x_axis = st.selectbox(
                    'X-Axis:',
                    options=list(desc_columns),
                    key='x_select',
                    help="Select a column for the X-axis of the plot."
                )

            with col2:
                metrics_columns = metrics.columns.drop(['IDENTIFIER', 'QE Period (s)', '# Images'])
                y_axis = st.selectbox(
                    'Y-Axis:',
                    options=list(metrics_columns),
                    key='y_select',
                    help="Select a column for the Y-axis of the plot."
                )

            with col3:
                available_legends = [col for col in desc_columns if col != 'IDENTIFIER' and col != x_axis]
                legend_column = st.selectbox(
                    'Legend (optional):',
                    options=['None'] + available_legends,
                    key='legend_select',
                    help="Choose a column to group data by legend (optional)."
                )

            with col4:
                plot_type = st.selectbox(
                    'Plot Type:',
                    options=['Bar', 'Line', 'Box', 'Violin'],
                    key='plot_type_select',
                    help="Select the type of plot to visualize data."
                )

            if not metridata.empty:
                # Group and aggregate metrics
                if legend_column != 'None':
                    grouped_metrics = metridata.groupby([x_axis, legend_column])[y_axis].mean().reset_index()
                else:
                    grouped_metrics = metridata.groupby(x_axis)[y_axis].mean().reset_index()

                # Configure plot settings
                plot_kwargs = {'x': x_axis, 'y': y_axis, 'height': 400}
                if legend_column != 'None':
                    plot_kwargs['color'] = legend_column

                # Generate the selected plot type
                if plot_type == 'Bar':
                    fig = px.bar(grouped_metrics, **plot_kwargs)
                elif plot_type == 'Line':
                    fig = px.line(grouped_metrics, **plot_kwargs)
                elif plot_type == 'Box':
                    fig = px.box(metridata, **plot_kwargs)
                elif plot_type == 'Violin':
                    fig = px.violin(metridata, **plot_kwargs)

                # Display the plot
                st.plotly_chart(fig, use_container_width=True)

                # Show aggregated data
                display_columns = [x_axis, y_axis] + ([legend_column] if legend_column != 'None' else [])
                display_df = grouped_metrics.set_index(x_axis)
                st.dataframe(display_df, use_container_width=True, height=200)

            st.markdown("___")
            st.subheader("Time Series Comparison")

            # Section for configuring time-series plots
            col1, col2, col3 = st.columns(3)
            time_series_columns = timeseries.columns.drop('IDENTIFIER')

            with col1:
                num_axes = st.selectbox(
                    'Number of Y-Axes:',
                    options=[1, 2],
                    index=0,
                    key='num_axes_select',
                    help="Select the number of Y-axes to display."
                )

            with col2:
                y1_label = st.selectbox(
                    'Y1 Axis:',
                    options=list(time_series_columns),
                    key='y1_select',
                    help="Select a column for the primary Y-axis."
                )

            y2_label = None
            if num_axes == 2:
                with col3:
                    y2_label = st.selectbox(
                        'Y2 Axis:',
                        options=list(time_series_columns),
                        key='y2_select',
                        help="Select a column for the secondary Y-axis."
                    )

            # Reset timeseries index and merge with metadata
            timeseries_analysis = timeseries_analysis.reset_index()
            timeseries_data = timeseries_analysis.merge(metadata_analysis, on='IDENTIFIER', how='inner').set_index('index')

            if not timeseries_data.empty:
                # Plot time-series data
                fig = go.Figure()

                # Primary Y-axis plot
                grouped_y1 = timeseries_data[y1_label].groupby(timeseries_data.index).mean()
                fig.add_trace(go.Scatter(x=grouped_y1.index, y=grouped_y1, mode='lines', name=y1_label))

                # Secondary Y-axis plot if applicable
                if y2_label:
                    grouped_y2 = timeseries_data[y2_label].groupby(timeseries_data.index).mean()
                    fig.add_trace(go.Scatter(x=grouped_y2.index, y=grouped_y2, mode='lines', name=y2_label, yaxis='y2'))

                    fig.update_layout(
                        yaxis2=dict(
                            title=y2_label,
                            overlaying='y',
                            side='right'
                        )
                    )

                # Update layout for readability
                fig.update_layout(
                    title="Time Series Comparison",
                    xaxis_title="Time (s)",
                    yaxis_title=y1_label,
                    hovermode="x"
                )

                # Display plot
                st.plotly_chart(fig, use_container_width=True)

                # Display data
                display_columns = ['Time (s)', y1_label] + ([y2_label] if y2_label else [])
                display_df = timeseries_data.reset_index().rename(columns={'index': 'Time (s)'})[display_columns]
                st.dataframe(display_df, use_container_width=True, height=200)


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



    


    
