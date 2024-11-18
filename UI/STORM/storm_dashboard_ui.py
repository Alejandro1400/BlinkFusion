import time
from matplotlib import pyplot as plt
import numpy as np
import streamlit as st
import pandas as pd
import os
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go

from Analysis.STORM.analytics_storm import aggregate_metrics, calculate_frequency, calculate_quasi_equilibrium, calculate_time_series_metrics, obtain_molecules_metrics
from Dashboard.graphs import create_histogram, plot_histograms, plot_intensity_vs_frame, plot_time_series_interactive
from Data_access.file_explorer import find_items, find_valid_folders
from Data_access.metadata_manager import read_tiff_metadata


@st.cache_data
def load_storm_data(pulseSTORM_folder):
    """
    Load and process data from a specified folder containing STORM analysis files. 
    This function iterates through folders, reads relevant data files, calculates metrics, 
    and compiles them into dataframes based on unique identifiers.
    
    Args:
        pulseSTORM_folder (str): The directory path that contains all data folders.

    Returns:
        tuple: A tuple containing processed pandas DataFrames or None in case of failure.
    """
    # Lists to store timeseries
    metadata = []
    localizations = []
    tracks = []
    molecules = []
    timeseries = []

    # Progress bar and status text
    status_text = st.text("Loading Localization Statistics...")
    progress_bar = st.progress(0, text="Loading Localization Statistics...")

    try:
        # Discover all valid folders
        valid_folders = find_valid_folders(
            pulseSTORM_folder,
            required_files={'.tif', 'locs_blink_stats.csv', 'track_blink_stats.csv', 'mol_blink_stats.csv'}
        )
        total_folders = len(valid_folders)

        # Process each folder
        for index, folder in enumerate(valid_folders):
            time.sleep(0.1)
            
            # Load required files
            tif_file = find_items(folder, '.tif', False, True)
            locs = find_items(folder, 'locs_blink_stats.csv', False, True)
            track = find_items(folder, 'track_blink_stats.csv', False, True)
            mol = find_items(folder, 'mol_blink_stats.csv', False, True)

            if all([tif_file, locs, track, mol]):
                relative_path = os.path.relpath(folder, pulseSTORM_folder)
                progress_bar.progress((index + 1) / total_folders, text=f"Loading Localizations statistics for {relative_path} ({index + 1} of {total_folders})")

                # Process metadata
                pulsestorm_metadata = read_tiff_metadata(tif_file, root_tag=['pulsestorm', 'czi-pulsestorm'])
                # Create an empty dictionary
                metadata_dict = {}

                # Loop through each item in the metadata list
                for item in pulsestorm_metadata:
                    # Check if the item's 'id' is not already in the dictionary
                    if item['id'] not in metadata_dict:
                        # If not present, add the item's 'id' and 'value' to the dictionary
                        metadata_dict[item['id']] = item['value']

                # Function to extract the image name before the first underscore
                def extract_image_name(path):
                    basename = os.path.basename(path)
                    image_name = basename.split('_')[0].replace(' ','')
                    return image_name

                meta_df = pd.DataFrame([metadata_dict])
                meta_df.columns = meta_df.columns.str.upper()
                meta_df['IDENTIFIER'] = relative_path
                meta_df['IMAGE'] = extract_image_name(tif_file)


                # Read CSV data
                locs_df = pd.read_csv(locs)
                locs_df = locs_df[locs_df['TRACK_ID'] != 0]
                locs_df['IDENTIFIER'] = relative_path
                track_df = pd.read_csv(track)
                track_df['IDENTIFIER'] = relative_path
                mol_df = pd.read_csv(mol)
                mol_df['IDENTIFIER'] = relative_path

                progress_bar.progress((index + 1) / total_folders, text=f"Calculating Time Series for {relative_path} ({index + 1} of {total_folders})")

                # Calculate time series and molecular metrics
                frames = meta_df['FRAMES'][0]
                exp = meta_df['EXPOSURE'][0]
                time_df = calculate_time_series_metrics(mol_df, track_df, 50, frames, exp)

                time_df['IDENTIFIER'] = relative_path

                # Append to lists
                metadata.append(meta_df)
                localizations.append(locs_df)
                tracks.append(track_df)
                molecules.append(mol_df)
                timeseries.append(time_df)

            # Update progress
            progress_bar.progress((index + 1) / total_folders)

        # Compile all into final DataFrames
        metadata_df = pd.concat(metadata)
        localizations_df = pd.concat(localizations)
        tracks_df = pd.concat(tracks)
        molecules_df = pd.concat(molecules)
        timeseries_df = pd.concat(timeseries)

        # Clear progress and status
        status_text.empty()
        progress_bar.empty()

        return localizations_df, tracks_df, molecules_df, metadata_df, timeseries_df

    except Exception as e:
        status_text.text(f"An error occurred: {e}")
        return None, None, None, None, None



# Function to run the PulseSTORM UI
def run_storm_dashboard_ui(pulseSTORM_folder):
    
    # Load the cached dataframe
    localizations, tracks, molecules, metadata, timeseries = load_storm_data(pulseSTORM_folder)

    st.write(f" **{len(metadata)}** images loaded in the Dataset.")

    if localizations.empty or tracks.empty or molecules.empty:
        st.error("No data found in the folder. Please check the folder path.")
        return

    # Initialize copies of all datasets for use after filtering
    locs_analysis = localizations.copy()
    tracks_analysis = tracks.copy()
    metadata_analysis = metadata.copy()
    molecules_analysis = molecules.copy()
    timeseries_analysis = timeseries.copy()

    desc_columns = metadata.columns

    with st.expander("Filter Options", expanded=True):

        if not metadata.index.is_unique:
            metadata = metadata.reset_index(drop=True)

        # Section for selecting filters
        st.write("Apply filters based on the selected columns (all values are selected by default).")
        selected_filter_columns = st.multiselect(
            'Select columns to filter by:',
            list(desc_columns),
            key='filter_select'
        )
        
        # Apply filters if any columns are selected for filtering
        if selected_filter_columns:
            filters = {}
            filter_mask = pd.Series([True] * len(metadata))
            for col in selected_filter_columns:
                unique_values = metadata[col].unique()
                selected_values = st.multiselect(
                    f"Filter {col}:",
                    unique_values,
                    default=unique_values,
                    key=f'filter_{col}'
                )
                filters[col] = selected_values
                filter_mask &= metadata[col].isin(selected_values)

            metadata_analysis = metadata[filter_mask]
        else:
            metadata_analysis = metadata.copy()

        # Display grouped data, ensure IDENTIFIER is included if not already in the group by list
        if 'IDENTIFIER' not in selected_filter_columns and selected_filter_columns:
            selected_filter_columns.append('IDENTIFIER')

        # If no columns are selected for grouping, use all available columns from filtered metadata
        display_columns = selected_filter_columns if selected_filter_columns else metadata_analysis.columns.tolist()
        filtered_metadata = metadata_analysis[display_columns]

        st.markdown("___")

        st.write("Data after Filtering:")
        if not metadata_analysis.empty:
            st.dataframe(filtered_metadata)
        else:
            st.write("No data found for the selected groups.")

        # Filter other datasets based on the identifiers from metadata_analysis
        identifiers = metadata_analysis['IDENTIFIER'].unique()
        locs_analysis = localizations[localizations['IDENTIFIER'].isin(identifiers)]
        tracks_analysis = tracks[tracks['IDENTIFIER'].isin(identifiers)]
        molecules_analysis = molecules[molecules['IDENTIFIER'].isin(identifiers)]
        timeseries_analysis = timeseries[timeseries['IDENTIFIER'].isin(identifiers)]


    metrics = obtain_molecules_metrics(tracks_analysis, timeseries_analysis, metadata_analysis)
    # Add empty column Images to the metrics dataframe at the beginning
    metrics.insert(0, '# Images', '')
    metrics_columns = metrics.columns.drop('IDENTIFIER')
    # Merge metrics with metadata to align additional context for aggregation
    if not metrics.empty and not metadata.empty:
        metridata = pd.merge(metadata_analysis, metrics, on='IDENTIFIER', how='inner')


    with st.expander("Blinking Statistics", expanded=True):

        # Section for selecting group by columns and metrics to display using two columns
        col1, col2 = st.columns(2)

        with col1:
            selected_group_columns = st.multiselect(
                'Choose columns to group by (Be mindful of the hierarchical order):',
                list(desc_columns),
                key='group_by_select'
            )

        with col2:
            selected_metrics_columns = st.multiselect(
                'Choose columns to display:',
                list(metrics_columns),
                key='metrics_select'
            )

        # Display grouped data
        if not metrics.empty :

            selected_columns = selected_group_columns + selected_metrics_columns

            # Determine columns for the final display based on user selections
            if not selected_columns:
                selected_columns = metridata.columns.tolist()  # All columns if none specifically chosen

            if not selected_group_columns:
                selected_group_columns = 'IDENTIFIER'

            # If no columns are selected for grouping, use all available columns from filtered metadata
            grouped_metrics = metridata.groupby(selected_group_columns).apply(aggregate_metrics)
            grouped_metrics = grouped_metrics[selected_metrics_columns]

            st.markdown("___")

            st.dataframe(grouped_metrics)


    with st.expander("Comparison Analysis", expanded=True):
        st.subheader("Metrics Comparison")
        # Section for selecting axis, label, and type of plot
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            x_axis = st.selectbox(
                'X-Axis:',
                options=list(desc_columns),
                key='x_select'
            )

        with col2:
            metrics_columns = metrics.columns.drop(['IDENTIFIER', 'QE Period (s)', '# Images'])
            y_axis = st.selectbox(
                'Y-Axis:',
                options=list(metrics_columns),
                key='y_select'
            )

        with col3:
            available_legends = [col for col in desc_columns if col != 'IDENTIFIER' and col != x_axis]
            legend_column = st.selectbox(
                'Legend (optional):',
                options=['None'] + available_legends,
                key='legend_select'
            )

        with col4:
            plot_type = st.selectbox(
                'Plot Type:',
                options=['Bar', 'Line', 'Box', 'Violin'],
                key='plot_type_select'
            )

        if not metridata.empty:
            # Perform grouping and aggregation
            if legend_column != 'None':
                grouped_metrics = metridata.groupby([x_axis, legend_column])[y_axis].mean().reset_index()
            else:
                grouped_metrics = metridata.groupby([x_axis])[y_axis].mean().reset_index()

            # Create plot based on user selection
            plot_kwargs = {'x': x_axis, 'y': y_axis, 'height': 400}
            if legend_column != 'None':
                plot_kwargs['color'] = legend_column

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

            # Display grouped data
            display_columns = []
            if legend_column != 'None':
                display_columns.append(legend_column)
            display_columns += [x_axis, y_axis]
            display_df = grouped_metrics.set_index(x_axis)
            st.dataframe(display_df, use_container_width=True, height=200)

        st.markdown("___")
        st.subheader("Time Series Comparison")
        # Section for selecting axis for timeseries comparison
        col1, col2, col3 = st.columns(3)

        time_series_columns = timeseries.columns.drop('IDENTIFIER')

        y2_label = None

        with col1:
            num_axes = st.selectbox(
                'Number of Y-Axes:',
                options=[1, 2],
                index=0,
                key='num_axes_select'
            )

        with col2: 
            y1_label = st.selectbox(
                'Y1 Axis:',
                options=list(time_series_columns),
                key='y1_select',
                index=0
            )

        legend = None  # Initialize legend option
        if num_axes == 1:
            with col3:
                legend = st.selectbox(
                    'Choose legend:',
                    options=['None'] + list(desc_columns),
                    key='legend_by_select'
                )

        if num_axes == 2:
            with col3:
                y2_label = st.selectbox(
                    'Y2 Axis:',
                    options=list(time_series_columns),
                    key='y2_select',
                    index=0
                )

        # Convert the timeseries_analysis index to a column 'Time'
        timeseries_analysis = timeseries_analysis.reset_index()
        timeseries_data = timeseries_analysis.merge(metadata_analysis, on='IDENTIFIER', how='inner').set_index('index')

        if not timeseries_data.empty:
            # Round index to the nearest 10 for alignment
            timeseries_data.index = timeseries_data.index.map(lambda x: int(np.ceil(x / 10) * 10))
            
            fig = go.Figure()

            # Compute and plot the primary y-axis
            if legend != 'None' and legend:
                # Group by index and legend if a legend is specified
                grouped = timeseries_data.groupby([timeseries_data.index, legend])[y1_label].mean().unstack()
                for col in grouped.columns:
                    fig.add_trace(go.Scatter(
                        x=grouped.index,
                        y=grouped[col],
                        mode='lines',
                        name=col  # Only display legend if there are multiple lines
                    ))
            else:
                # Calculate the mean across the entire dataset for each time frame
                mean_series = timeseries_data[y1_label].groupby(timeseries_data.index).mean()
                fig.add_trace(go.Scatter(
                    x=mean_series.index,
                    y=mean_series,
                    mode='lines',
                    name=y1_label,
                    showlegend=False  # Disable legend for single series
                ))

            # Plot secondary Y-axis if applicable
            if num_axes == 2:
                secondary_mean_series = timeseries_data[y2_label].groupby(timeseries_data.index).mean()
                fig.add_trace(go.Scatter(
                    x=secondary_mean_series.index,
                    y=secondary_mean_series,
                    mode='lines',
                    name=y2_label,
                    yaxis='y2',
                    line=dict(color='red'),  # Use a different color for the second y-axis
                    showlegend=False  # Disable legend for secondary series
                ))

                # Create a second y-axis on the right side
                fig.update_layout(
                    yaxis2=dict(
                        title=y2_label,
                        overlaying='y',
                        side='right',
                        showgrid=False,
                        tickfont=dict(color='red'),
                    )
                )

            # Customize layout
            fig.update_layout(
                title=f"{y1_label} Time Series Comparison",
                xaxis_title='Time (s)',
                yaxis_title=y1_label,
                yaxis=dict(tickfont=dict(color='blue')),
                hovermode='x'
            )

            st.plotly_chart(fig, use_container_width=True)

            # Dataframe display
            display_columns = []
            new_columns = ['Time (s)', y1_label] + ([y2_label] if y2_label else [])
            if legend and legend != 'None':
                display_columns.append(legend)
            display_columns += new_columns
            display_df = timeseries_data.reset_index()
            # Change index column as 'Time (s)'
            display_df = display_df.rename(columns={'index': 'Time (s)'}).set_index('IDENTIFIER')
            display_df = display_df[display_columns]
            st.dataframe(display_df, use_container_width=True, height=200)        



    with st.expander("Image Statistics", expanded=True):

        st.subheader("Time Series Analysis")

        time_series_columns = timeseries.columns.drop('IDENTIFIER')

        selected_id = st.selectbox(
            "Select Image", 
            metadata_analysis['IDENTIFIER'].unique(), 
            index=0)

        # Section for selecting group by columns and metrics to display using two columns
        col1, col2 = st.columns(2)

        with col1:
            selected_y1_label = st.selectbox(
                'Y1 Axis:',
                options=list(time_series_columns),
                key='y1_image_select',
                index=0
            )

        with col2:
            available_legends = [col for col in time_series_columns if col != 'IDENTIFIER' and col != selected_y1_label]
            selected_y2_label = st.selectbox(
                'Y2 Axis:',
                options=list(available_legends),
                key='y2_image_select',
                index=0
            )
        
        # Filter analysis dataframes based on the selection
        selected_localizations = locs_analysis[locs_analysis['IDENTIFIER'] == selected_id]
        selected_tracks = tracks_analysis[tracks_analysis['IDENTIFIER'] == selected_id]
        selected_molecules = molecules_analysis[molecules_analysis['IDENTIFIER'] == selected_id]
        selected_metadata = metadata_analysis[metadata_analysis['IDENTIFIER'] == selected_id]
        selected_timeseries = timeseries_analysis[timeseries_analysis['IDENTIFIER'] == selected_id]
        selected_metrics = metrics[metrics['IDENTIFIER'] == selected_id]

         # Extracting the start and end values from the string
        qe_start, qe_end = map(int, selected_metrics['QE Period (s)'].iloc[0].split('-'))
        frames = selected_metadata['FRAMES'].iloc[0]
        exp = selected_metadata['EXPOSURE'].iloc[0]
        qe_dc, qe_sf = selected_metrics['QE Duty Cycle'].iloc[0], selected_metrics['QE Survival Fraction'].iloc[0]


        # Show as a df the selected metadata
        st.dataframe(selected_metadata.set_index('IDENTIFIER'))

        selected_columns = ['index', selected_y1_label, selected_y2_label]

        if not selected_timeseries.empty:  
            
            time_metric = selected_timeseries[selected_columns]

            time_metric = time_metric.set_index('index').sort_index()
            time_metric.index = time_metric.index.map(lambda x: int(np.ceil(x / 10) * 10))

            # Calling the function
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


            st.write("Time Series Data")
            st.dataframe(time_metric, use_container_width=True, height=200)

        st.markdown("___")
        
        st.subheader("Histogram Analysis")

        col1, col2, col3, col4 = st.columns(4)
        # Select if it wants to see the data for whole Population or just QE
        with col1:
            selected_option1 = st.radio("Select the tracks frequency to display", ["Whole Population", "Quasi-Equilibrium Population"])

        # Select if it wants to display it by molecule or by track
        with col2:
            selected_option2 = st.radio("Select the grouping to display", ["By Molecule", "By Track"])

        with col3:
            num_bins = st.slider("Number of Bins", min_value=5, max_value=50, value=20)

        with col4:
            remove_outliers = st.checkbox("Remove Outliers", value=False)

        # Convert options to function parameters
        population_type = 'quasi' if selected_option1 == "Quasi-Equilibrium Population" else 'whole'
        grouping_type = 'molecule' if selected_option2 == "By Molecule" else 'track'

        # Call the function with dynamic parameters
        duty_cycle, photons, switching_cycles, on_time = calculate_frequency(
            selected_qe_tracks=selected_tracks,
            qe_start=qe_start,
            qe_end=qe_end,
            frames=frames,
            exp=exp,
            population=population_type,
            metric=grouping_type
        )

        plot_histograms(duty_cycle, photons, switching_cycles, on_time, selected_metrics, remove_outliers, num_bins)




    # Ask select a molecule 
    #selected_molecule = st.selectbox("Select Molecule", selected_molecules['MOLECULE_ID'].unique(), index=4)

    # Filter the tracks for those with that Mol ID
    # selected_tracks = selected_tracks[selected_tracks['MOLECULE_ID'] == selected_molecule]
    # Filter the localizations for those with the selected tracks id
    # selected_localizations = selected_localizations[selected_localizations['TRACK_ID'].isin(selected_tracks['TRACK_ID'])]

    # Prepare the DataFrame for plotting
    #plot_data = pd.DataFrame({
    #    'INTENSITY': selected_localizations['INTENSITY'].values,
    #    'FRAME': selected_localizations['FRAME'].values
    #})
    #plot_data = plot_data.set_index('FRAME').sort_index()

    # Interpolate only one-frame gaps
    # Identify gaps by checking where the difference between consecutive indices is 2
    #diff = plot_data.index.to_series().diff()  # Compute the difference between consecutive frames
    #gaps = diff[diff == 2].index  # Find indices where the difference is exactly 2

    # Calculate the mean for one-frame gaps
    #for gap_start in gaps:
    #    if gap_start + 1 in plot_data.index:  # Ensure that gap_start + 1 is a valid index
    #        continue  # Skip if there's no actual gap
        # Calculate the mean of the intensities around the gap
     #   mean_intensity = (plot_data.at[gap_start - 1, 'INTENSITY'] + plot_data.at[gap_start + 1, 'INTENSITY']) / 2
    #    plot_data.at[gap_start, 'INTENSITY'] = mean_intensity  # Set the intensity for the gap

    # Fill other gaps with NaN to avoid artificial zeros, which could be misleading in data analysis
    #plot_data = plot_data.reindex(range(plot_data.index.min(), plot_data.index.max() + 1), fill_value=0)

    # Plotting with Matplotlib
    #fig = plot_intensity_vs_frame(plot_data)
    #st.plotly_chart(fig, use_container_width=True)



    


    
