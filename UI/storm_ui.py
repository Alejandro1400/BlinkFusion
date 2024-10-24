from matplotlib import pyplot as plt
import streamlit as st
import pandas as pd
import os

from Analysis.STORM.analytics_storm import obtain_molecules_metrics
from Analysis.STORM.blink_statistics import trackmate_blink_statistics
from Dashboard.graphs import calculate_duty_cycle, calculate_survival_fraction, create_histogram
from Data_access.file_explorer import assign_structure_folders, find_items, find_valid_folders

# Function to load and cache the dataframe from multiple CSV files
@st.cache_data
def load_storm_data(pulseSTORM_folder):
    # List to store dataframes from all CSVs
    localizations = []
    tracks = []
    molecules = []
    images = []

    # Find the folder structure file (folder_structure.txt)
    structure_path = find_items(base_directory=pulseSTORM_folder, item='folder_structure.txt', is_folder=False)

    if structure_path:
        # Find all valid folders (folders that contain the required files)
        valid_folders = find_valid_folders(
            pulseSTORM_folder,
            required_files={'trackmate_locs_blink_stats.csv', 'trackmate_track_blink_stats.csv', 'trackmate_mol_blink_stats.csv'}
        )

        print(valid_folders)
        
        # Get the folder structure as a DataFrame
        image_df = assign_structure_folders(pulseSTORM_folder, structure_path, valid_folders)

        # Iterate through each valid folder
        for folder in valid_folders:
            locs = find_items(
                base_directory=folder, 
                item='trackmate_locs_blink_stats.csv', 
                is_folder=False, 
                search_by_extension=True
            )

            track = find_items(
                base_directory=folder,
                item='trackmate_track_blink_stats.csv',
                is_folder=False,
                search_by_extension=True
            )

            mol = find_items(
                base_directory=folder,
                item='trackmate_mol_blink_stats.csv',
                is_folder=False,
                search_by_extension=True
            )

            if locs and track and mol:
                # Load the dataframes
                locs_df = pd.read_csv(locs)
                track_df = pd.read_csv(track)
                mol_df = pd.read_csv(mol)
                
                # Add the folder column with just the last part of the path to each dataframe
                locs_df['folder'] = os.path.basename(folder)
                track_df['folder'] = os.path.basename(folder)
                mol_df['folder'] = os.path.basename(folder)

                # Obtain mol metrics
                mol_metrics = obtain_molecules_metrics(mol_df)

                # Merge mol_metrics into image_df correctly
                mol_metrics['folder'] = os.path.basename(folder)  # Ensure common identifier for merging
                image_df['folder'] = os.path.basename(folder)  # Ensure common identifier for merging
                image_df = pd.merge(image_df, mol_metrics, on='folder', how='left')

                # Append the dataframes to the lists
                localizations.append(locs_df)
                tracks.append(track_df)
                molecules.append(mol_df)
                images.append(image_df)

                # Eliminate from images the column identifier and folde
                image_df.drop(columns=['identifier','folder'], inplace=True)
    
    # Combine all dataframes into a single dataframe for each type
    localizations_df, tracks_df, molecules_df, images_df = map(pd.concat, [localizations, tracks, molecules, images])
    
    return localizations_df, tracks_df, molecules_df, images_df



# Function to run the PulseSTORM UI
def run_storm_ui(pulseSTORM_folder):
    st.subheader("Running PulseSTORM Analysis")
    
    # Load the cached dataframe
    localizations, tracks, molecules, images = load_storm_data(pulseSTORM_folder)

    if localizations.empty or tracks.empty or molecules.empty:
        st.error("No data found in the folder. Please check the folder path.")
        return
    
    # Display Images loaded
    st.write("Images loaded:")
    st.dataframe(images)  # Show the first few rows of the dataframe

    # Ask the user to select 
    # Ask the user to select a folder, initially do not filter
    selected_image = st.selectbox("Select Image", images['Image'].unique(), index=0)

    # Filter the dataframes based on the selected folder
    selected_localizations = localizations[localizations['folder'] == selected_image]
    selected_tracks = tracks[tracks['folder'] == selected_image]
    selected_molecules = molecules[molecules['folder'] == selected_image]

    # Plot histograms of the data
    st.subheader("Histograms")
    
    # Track on_time histogram
    st.write("On Time Histogram")
    # Select number of bins
    histogram = create_histogram(selected_tracks, 'ON_TIME', 5, 'On Time per burst')
    st.pyplot(histogram)


    # Molecule on_time histogram
    st.write("Molecule On Time Histogram")
    # Select number of bins
    histogram = create_histogram(selected_molecules, 'TOTAL_ON_TIME', 50, 'On Time per molecule')
    st.pyplot(histogram)

    # Plot vs time 
    st.subheader("Time Plots")

    duty_cycles = calculate_duty_cycle(molecules, tracks, range(0, 10000, 1000))  # Example range

    # Plotting with Streamlit
    st.write("Duty Cycle Over Time")
    st.line_chart(duty_cycles)

    survival_fraction = calculate_survival_fraction(molecules, tracks, range(0, 10000, 1000))  # Example range

    # Plotting with Streamlit
    st.write("Survival Fraction Over Time")
    st.line_chart(survival_fraction)

    # Ask select a molecule 
    selected_molecule = st.selectbox("Select Molecule", molecules['MOLECULE_ID'].unique(), index=4)

    # Filter the tracks for those with that Mol ID
    selected_tracks = tracks[tracks['MOLECULE_ID'] == selected_molecule]
    # Filter the localizations for those with the selected tracks id
    selected_localizations = localizations[localizations['TRACK_ID'].isin(selected_tracks['TRACK_ID'])]

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
    fig, ax = plt.subplots()
    ax.fill_between(plot_data.index, plot_data['INTENSITY'], color='blue', step='post', alpha=0.5)
    ax.set_title('Intensity vs Frame')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Intensity')
    ax.grid(True)

    # Display the plot in Streamlit
    st.pyplot(fig)



    


    
