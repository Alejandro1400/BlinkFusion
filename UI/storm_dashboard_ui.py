import time
from matplotlib import pyplot as plt
import numpy as np
import streamlit as st
import pandas as pd
import os

from Analysis.STORM.analytics_storm import aggregate_metrics, calculate_molecule_metrics, calculate_quasi_equilibrium, calculate_time_series_metrics, obtain_molecules_metrics
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



    with st.expander("Blinking Statistics", expanded=True):

        metrics = obtain_molecules_metrics(tracks_analysis, timeseries_analysis, metadata_analysis)
        # Add empty column Images to the metrics dataframe at the beginning
        metrics.insert(0, '# Images', '')
        metrics_columns = metrics.columns.drop('IDENTIFIER')

        # Section for selecting group by columns and metrics to display using two columns
        col1, col2 = st.columns(2)

        with col1:
            st.write("Select columns to group by (leave empty to use all available columns):")
            selected_group_columns = st.multiselect(
                'Choose columns to group by (Be mindful of the hierarchical order):',
                list(desc_columns),
                key='group_by_select'
            )

        with col2:
            st.write("Select metrics to display:")
            selected_metrics_columns = st.multiselect(
                'Choose columns to display:',
                list(metrics_columns),
                key='metrics_select'
            )

        # Display grouped data
        if not metrics.empty :

            # Merge metrics with metadata to align additional context for aggregation
            metridata = pd.merge(metadata, metrics, on='IDENTIFIER', how='inner')
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

            st.write("Grouped Metrics:")
            st.dataframe(grouped_metrics)



    #mol_metrics, qe_tracks_m = obtain_molecules_metrics(molecules, tracks, time_df, exp)
    
    # Display metadata loaded
    st.write("metadata loaded:")
    # Reset the index and drop the old one to ensure it does not appear in the display
    #metrics_no_id = metrics.drop(columns=['IDENTIFIER'])
    #st.dataframe(metrics_no_id)

    # Selection box with state
    selected_id = st.selectbox("Select Image", metadata['IDENTIFIER'].unique(), index=0)

    # Filter dataframes based on the selection
    selected_localizations = localizations[localizations['IDENTIFIER'] == selected_id]
    selected_tracks = tracks[tracks['IDENTIFIER'] == selected_id]
    selected_molecules = molecules[molecules['IDENTIFIER'] == selected_id]
    selected_metadata = metadata[metadata['IDENTIFIER'] == selected_id]
    #selected_qe_tracks = qe_tracks[qe_tracks['IDENTIFIER'] == selected_id]
    
    selected_timeseries = timeseries[timeseries['IDENTIFIER'] == selected_id]
    #selected_metrics = metrics[metrics['IDENTIFIER'] == selected_id]   

    # Plot vs time 
    st.subheader("Time Plots")

    duty_cycles = selected_timeseries['Duty Cycle']
    survival_fraction = selected_timeseries['Survival Fraction']

    # Round up the indices to the nearest 10
    duty_cycles.index = duty_cycles.index.map(lambda x: int(np.ceil(x / 10) * 10))
    survival_fraction.index = survival_fraction.index.map(lambda x: int(np.ceil(x / 10) * 10))

    # Convert to pd.Series
    duty_cycles = pd.Series(duty_cycles)
    survival_fraction = pd.Series(survival_fraction)

    #Include only certain columns from metrics
    # From the identifier grab only after the last slash
    #metrics['IDENTIFIER'] = metrics['IDENTIFIER'].str.split('/').str[-1]
    # Round up to nearest 10 both the QE Start and QE End
    #metrics['QE Start'] = metrics['QE Start'].apply(lambda x: round(x / 10) * 10)
    #metrics['QE End'] = metrics['QE End'].apply(lambda x: round(x / 10) * 10)
    #st.write(metrics[['IDENTIFIER', 'DATE', 'SAMPLE', 'PARTICLE', 'Molecules', 'QE Start', 'QE Duty Cycle', 'QE Survival Fraction', 'QE Active Population', 'QE Switching Cycles per mol', 'QE Photons per SC', 'QE Mean Uncertainty', 'QE On Time per SC', 'QE End']])
    

    # Grab the QE start and End from selected_metrics and round up to nearest 10
    #qe_start = selected_metrics['QE Start'].iloc[0]
    #qe_end = selected_metrics['QE End'].iloc[0]
    #qe_dc = selected_metrics['QE Duty Cycle'].iloc[0]
    #qe_sf = selected_metrics['QE Survival Fraction'].iloc[0]
    #qe_start = round(qe_start / 10) * 10
    #qe_end = round(qe_end / 10) * 10

    # Obtain exposure time from metadata
    #exp = selected_metadata['EXPOSURE'].iloc[0]

    # Remove from duty_cycles and survival_fraction the last 2 values
    #duty_cycles = duty_cycles.iloc[:-2]
    #survival_fraction = survival_fraction.iloc[:-2]

    # Generate the interactive plot
    #interactive_fig = plot_time_series_interactive(duty_cycles, survival_fraction, qe_start, qe_end, qe_dc, qe_sf)

    # Display the plot in Streamlit
    #st.plotly_chart(interactive_fig, use_container_width=True)

    #duty_cycle, photons, switching_cycles, track_intensity_within_range = calculate_molecule_metrics(selected_qe_tracks, qe_start, qe_end, exp)

    #plot_histograms(duty_cycle, photons, switching_cycles, track_intensity_within_range)

    # Plot histograms of the data
    st.subheader("Histograms")
    
    # Track on_time histogram
    st.write("On Time Histogram")
    # Select number of bins
    histogram = create_histogram(selected_tracks, 'ON_TIME', 5, 'On Time per burst')
    st.pyplot(histogram)


    # Ask select a molecule 
    selected_molecule = st.selectbox("Select Molecule", selected_molecules['MOLECULE_ID'].unique(), index=4)

    # Filter the tracks for those with that Mol ID
    selected_tracks = selected_tracks[selected_tracks['MOLECULE_ID'] == selected_molecule]
    # Filter the localizations for those with the selected tracks id
    selected_localizations = selected_localizations[selected_localizations['TRACK_ID'].isin(selected_tracks['TRACK_ID'])]

    # Prepare the DataFrame for plotting
    plot_data = pd.DataFrame({
        'INTENSITY': selected_localizations['INTENSITY'].values,
        'FRAME': selected_localizations['FRAME'].values
    })
    plot_data = plot_data.set_index('FRAME').sort_index()

    # Interpolate only one-frame gaps
    # Identify gaps by checking where the difference between consecutive indices is 2
    diff = plot_data.index.to_series().diff()  # Compute the difference between consecutive frames
    gaps = diff[diff == 2].index  # Find indices where the difference is exactly 2

    # Calculate the mean for one-frame gaps
    for gap_start in gaps:
        if gap_start + 1 in plot_data.index:  # Ensure that gap_start + 1 is a valid index
            continue  # Skip if there's no actual gap
        # Calculate the mean of the intensities around the gap
        mean_intensity = (plot_data.at[gap_start - 1, 'INTENSITY'] + plot_data.at[gap_start + 1, 'INTENSITY']) / 2
        plot_data.at[gap_start, 'INTENSITY'] = mean_intensity  # Set the intensity for the gap

    # Fill other gaps with NaN to avoid artificial zeros, which could be misleading in data analysis
    plot_data = plot_data.reindex(range(plot_data.index.min(), plot_data.index.max() + 1), fill_value=0)

    # Plotting with Matplotlib
    fig = plot_intensity_vs_frame(plot_data)
    st.plotly_chart(fig, use_container_width=True)



    


    
