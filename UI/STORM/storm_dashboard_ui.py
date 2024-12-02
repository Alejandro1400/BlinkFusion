import time
from matplotlib import pyplot as plt
import numpy as np
import streamlit as st
import pandas as pd
import os
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go

from Analysis.STORM.analytics_storm import aggregate_metrics, calculate_frequency, calculate_time_series_metrics, obtain_molecules_metrics
from Dashboard.graphs import plot_histograms, plot_intensity_vs_frame, plot_time_series_interactive
from Data_access.file_explorer import find_items, find_valid_folders
from Data_access.metadata_manager import read_tiff_metadata


@st.cache_data
def load_storm_data(pulseSTORM_folder):
    """
    Load and process STORM analysis data from the specified folder. This function iterates through
    valid folders, reads required files, calculates metrics, and compiles data into unified DataFrames.

    Args:
        pulseSTORM_folder (str): Path to the directory containing STORM analysis data.

    Returns:
        tuple: A tuple of pandas DataFrames containing:
            - localizations_df: Processed localizations data.
            - tracks_df: Processed tracks data.
            - molecules_df: Processed molecular data.
            - metadata_df: Metadata extracted from `.tif` files.
            - timeseries_df: Time-series metrics.
        Returns (None, None, None, None, None) if an error occurs.
    """
    # Lists to store data from all folders
    metadata = []
    localizations = []
    tracks = []
    molecules = []
    timeseries = []

    # Progress bar and status indicators
    status_text = st.text("Loading Localization Statistics...")
    progress_bar = st.progress(0, text="Initializing...")

    try:
        # Discover all valid folders containing the required files
        valid_folders = find_valid_folders(
            pulseSTORM_folder,
            required_files={'.tif', 'locs_blink_stats.csv', 'track_blink_stats.csv', 'mol_blink_stats.csv'}
        )
        total_folders = len(valid_folders)

        if total_folders == 0:
            st.warning("No valid folders found. Please check the directory structure.")
            status_text.empty()
            progress_bar.empty()
            return None, None, None, None, None
        

        # Process each folder
        for index, folder in enumerate(valid_folders):
            time.sleep(0.1)  # Allow UI updates for the progress bar

            # Load required files from the folder
            tif_file = find_items(folder, '.tif', is_folder=False, check_multiple=False, search_by_extension=True)
            locs = find_items(folder, 'locs_blink_stats.csv', is_folder=False, check_multiple=False, search_by_extension=True)
            track = find_items(folder, 'track_blink_stats.csv', is_folder=False, check_multiple=False, search_by_extension=True)
            mol = find_items(folder, 'mol_blink_stats.csv', is_folder=False, check_multiple=False, search_by_extension=True)

            # Ensure all required files exist
            if all([tif_file, locs, track, mol]):
                relative_path = os.path.relpath(folder, pulseSTORM_folder)
                progress_bar.progress((index + 1) / total_folders, text=f"Processing folder {relative_path} ({index + 1}/{total_folders})")

                # Process metadata
                pulsestorm_metadata = read_tiff_metadata(tif_file, root_tag=['pulsestorm', 'czi-pulsestorm'])
                metadata_dict = {item['id']: item['value'] for item in pulsestorm_metadata}

                def extract_image_name(path):
                    """Extract image name from file path."""
                    return os.path.basename(path).split('_')[0].replace(' ', '')

                meta_df = pd.DataFrame([metadata_dict])
                meta_df.columns = meta_df.columns.str.upper()
                meta_df['IDENTIFIER'] = relative_path
                meta_df['IMAGE'] = extract_image_name(tif_file)

                # Read and process CSV data
                locs_df = pd.read_csv(locs)
                locs_df = locs_df[locs_df['TRACK_ID'] != 0]  # Exclude untracked localizations
                locs_df['IDENTIFIER'] = relative_path

                track_df = pd.read_csv(track)
                track_df['IDENTIFIER'] = relative_path

                mol_df = pd.read_csv(mol)
                mol_df['IDENTIFIER'] = relative_path

                # Calculate time-series metrics
                frames = meta_df['FRAMES'].iloc[0]
                exposure = meta_df['EXPOSURE'].iloc[0]
                time_df = calculate_time_series_metrics(mol_df, track_df, interval=50, total_frames=frames, exposure_time=exposure)
                time_df['IDENTIFIER'] = relative_path

                # Append processed data to respective lists
                metadata.append(meta_df)
                localizations.append(locs_df)
                tracks.append(track_df)
                molecules.append(mol_df)
                timeseries.append(time_df)


        # Combine all lists into unified DataFrames
        metadata_df = pd.concat(metadata, ignore_index=True)
        localizations_df = pd.concat(localizations, ignore_index=True)
        tracks_df = pd.concat(tracks, ignore_index=True)
        molecules_df = pd.concat(molecules, ignore_index=True)
        timeseries_df = pd.concat(timeseries)

        # Clear progress indicators
        status_text.empty()
        progress_bar.empty()


        return localizations_df, tracks_df, molecules_df, metadata_df, timeseries_df

    except Exception as e:
        st.error(f"An error occurred during data loading: {e}")
        status_text.empty()
        progress_bar.empty()
        return None, None, None, None, None



def run_storm_dashboard_ui(pulseSTORM_folder):
    """
    Displays the PulseSTORM Dashboard UI for filtering, analyzing, and visualizing STORM data.

    Args:
        pulseSTORM_folder (str): Path to the folder containing PulseSTORM analysis data.
    """

    # Load the cached dataframes
    localizations, tracks, molecules, metadata, timeseries = load_storm_data(pulseSTORM_folder)

    # Check if data is successfully loaded
    if metadata is None or localizations.empty or tracks.empty or molecules.empty:
        st.error("No valid data found in the folder. Please check the folder structure and contents.")
        return

    # Display the number of images loaded in the dataset
    st.write(f" **{len(metadata)}** images loaded in the dataset.")

    # Initialize copies of datasets for filtering and analysis
    locs_analysis = localizations.copy()
    tracks_analysis = tracks.copy()
    metadata_analysis = metadata.copy()
    molecules_analysis = molecules.copy()
    timeseries_analysis = timeseries.copy()

    # Get column names from metadata for filtering
    desc_columns = metadata.columns

    with st.expander("Filter Options", expanded=True):
        """
        Allows users to filter the metadata based on specific columns and values. 
        Updates other datasets (localizations, tracks, etc.) based on the filtered metadata.
        """

        # Reset index if not unique to prevent filtering issues
        if not metadata.index.is_unique:
            metadata = metadata.reset_index(drop=True)

        st.info("Apply filters to metadata columns. All data is included by default.")

        # Multiselect to choose columns for filtering
        selected_filter_columns = st.multiselect(
            'Select columns to filter by:',
            options=list(desc_columns),
            key='filter_select',
            help="Choose metadata columns to filter the dataset."
        )

        # Apply filtering if columns are selected
        if selected_filter_columns:
            filter_mask = pd.Series([True] * len(metadata))
            for col in selected_filter_columns:
                unique_values = metadata[col].unique()

                # Multiselect for column-specific filter values
                selected_values = st.multiselect(
                    f"Filter {col}:",
                    options=unique_values,
                    default=unique_values,
                    key=f'filter_{col}',
                    help=f"Select specific values for filtering the column '{col}'."
                )

                # Update filter mask
                filter_mask &= metadata[col].isin(selected_values)

            # Apply the combined filter mask to metadata
            metadata_analysis = metadata[filter_mask].reset_index(drop=True)

            if metadata_analysis.empty:
                st.warning("No data found for the selected filters.")
                return
        else:
            metadata_analysis = metadata.copy()

        # Ensure 'IDENTIFIER' is included in the group columns if filtering is applied
        if 'IDENTIFIER' not in selected_filter_columns and selected_filter_columns:
            selected_filter_columns.append('IDENTIFIER')

        # Display filtered metadata
        st.markdown("___")
        st.write("Data after Filtering:")
        display_columns = selected_filter_columns if selected_filter_columns else metadata_analysis.columns.tolist()
        filtered_metadata = metadata_analysis[display_columns]

        if not metadata_analysis.empty:
            st.dataframe(filtered_metadata)
        else:
            st.warning("No data found for the selected groups.")

        # Filter other datasets based on the identifiers from metadata_analysis
        identifiers = metadata_analysis['IDENTIFIER'].unique()
        locs_analysis = localizations[localizations['IDENTIFIER'].isin(identifiers)].reset_index(drop=True)
        tracks_analysis = tracks[tracks['IDENTIFIER'].isin(identifiers)].reset_index(drop=True)
        molecules_analysis = molecules[molecules['IDENTIFIER'].isin(identifiers)].reset_index(drop=True)
        timeseries_analysis = timeseries[timeseries['IDENTIFIER'].isin(identifiers)]

    # Calculate and display metrics
    st.markdown("___")
    st.write("### Metrics and Analysis")

    try:
        # Obtain molecular metrics based on filtered tracks and timeseries
        metrics = obtain_molecules_metrics(tracks_analysis, timeseries_analysis, metadata_analysis)

        # Add a column for the number of images and align it at the beginning
        metrics.insert(0, '# Images', len(metadata_analysis))
        metrics_columns = metrics.columns.drop('IDENTIFIER')

        # Merge metrics with metadata for context and further analysis
        if not metrics.empty and not metadata_analysis.empty:
            metridata = pd.merge(metadata_analysis, metrics, on='IDENTIFIER', how='inner')
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
            num_bins=num_bins
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
                            col1, col2 = st.columns(2)
                            with col1:
                                # Change the duty cycle to 5 decimal places
                                st.write(f"**Duty Cycle**: {molecule_data.get('Duty Cycle', 'N/A'):.5f}")
                            with col2:
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



    


    
