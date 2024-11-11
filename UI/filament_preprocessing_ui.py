import os
import pandas as pd
import streamlit as st

from Data_access.file_explorer import find_items, find_valid_folders
from Data_access.metadata_manager import aggregate_metadata_info, append_metadata_tags, process_tiff_metadata, read_tiff_metadata

@st.cache_data
def load_filament_metadata(soac_folder):

    #try: 
    valid_folders = find_valid_folders(
        soac_folder,
        required_files={'.tif'}
    )

    valid_folders = [folder for folder in valid_folders if not folder.endswith('ROIs')]

    database_metadata = {}

    for folder in valid_folders:
            
        # Find tif files in folder
        tif_file = find_items(base_directory=folder, item='.tif', is_folder=False, check_multiple=False, search_by_extension=True)

        # Obtain metadata from the tif file
        metadata = read_tiff_metadata(tif_file, root_tag = 'pulsestorm')


        # Process each metadata item
        for item in metadata:
            # Initialize a tuple to represent the value and type
            entry = (item['value'], item['type'])
            
            # Check if the id from the metadata is already in the database_metadata
            if item['id'] in database_metadata:
                # Check if the value already exists in the list; if not, append it
                if entry not in database_metadata[item['id']]:
                    database_metadata[item['id']].append(entry)
            else:
                # Create a new entry in the dictionary with the value and type in a tuple inside a list
                database_metadata[item['id']] = [entry]
        

    return database_metadata



def run_filament_preprocessing_ui(soac_folder):
    
    db_metadata = load_filament_metadata(soac_folder)

    # Initialize the session state variable if it does not already exist
    if 'selected_for_folder' not in st.session_state:
        st.session_state.selected_for_folder = []
        st.session_state.selected_for_folder.append("Date")

    if 'metadata_values' not in st.session_state:
        st.session_state.metadata_values = {}

    if 'show_add_form' not in st.session_state:
        st.session_state.show_add_form = False

    # Ensure reload_metadata state is initialized
    if 'reload_metadata' not in st.session_state:
        st.session_state.reload_metadata = False

    upload_path = st.text_input("Enter the path to the folder or file you wish to upload.")

    with st.expander("Tif File Metadata"):
        if upload_path:
            all_files_metadata = {}
            # Check if upload path is folder or file
            if os.path.isdir(upload_path):
                st.write("Folder selected, all assigned metada values will apply to all files in the folder.")

                # Find all files in the folder
                files = find_items(base_directory=upload_path, item='.tif', is_folder=False, check_multiple=True, search_by_extension=True)

                # Search for metadata in the files (prop, 'acquisition-time-local', 'pixel-size-x', 'pixel-size-y', 'ALC Laser')
                for file in files:
                    tif_metadata = process_tiff_metadata(file)

                    print(f"Metadata: {tif_metadata}")

                    all_files_metadata[file] = tif_metadata



            elif os.path.isfile(upload_path):
                st.write("File selected, metadata will be added to the file.")

                # Extract specific metadata entries
                tif_metadata = process_tiff_metadata(upload_path)

                all_files_metadata[file] = tif_metadata

            # Write the unique ids and their types as a table
            # Aggregating metadata information
            summary_df = aggregate_metadata_info(all_files_metadata)

            st.write(f"Files to upload: {len(files)}")

            st.write("Tif Metadata to be added (Count displays how many files have values):")
            st.table(summary_df)


        else:
            st.write("Please select a folder or file to continue.")


    with st.expander("Database Metadata"):
        st.write(f"Metadata found in Database: {soac_folder}")

        if len(db_metadata) == 0:
            st.write("No metadata found in the database.")

        for metadata_id, entries in db_metadata.items():
            with st.container():
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # Prepare current values list for the selectbox
                    current_values = [entry[0] for entry in entries]
                    current_values.append(None)
                    current_values.append("Add new...")  # Option to add a new value

                    # Attempt to get the currently selected value or default to the first available value
                    current_selection = st.session_state.metadata_values.get(metadata_id, [(current_values[0], entries[0][1])])[0][0]
                    
                    if current_selection not in current_values:
                        current_values.insert(0, current_selection)

                    selected_value = st.selectbox(f"{metadata_id} ({entries[0][1]})", current_values, index=current_values.index(current_selection), key=f"{metadata_id}_value_select")
                  

                    if selected_value == "Add new...":
                        new_value = st.text_input("Enter new value", key=f"{metadata_id}_new_value")
                        if new_value:
                            # Directly update or append the new value tuple to session state
                            st.session_state.metadata_values[metadata_id] = [(new_value, entries[0][1])]
                            db_metadata[metadata_id].append((new_value, entries[0][1]))
                    elif selected_value == None:
                        st.session_state.metadata_values[metadata_id] = [(None, entries[0][1])]
                    else:
                        st.session_state.metadata_values[metadata_id] = [(selected_value, entries[0][1])]


                with col2:
                    folder_selected = st.checkbox("Folder", key=f"{metadata_id}_folder")
                    if folder_selected:
                        if metadata_id not in st.session_state.selected_for_folder:
                            st.session_state.selected_for_folder.append(metadata_id)
                    else:
                        if metadata_id in st.session_state.selected_for_folder:
                            st.session_state.selected_for_folder.remove(metadata_id)
                        
    
    # Use an expander for the form
    with st.expander("Add Metadata"):
        with st.form(key='new_metadata_form'):
            new_id = st.text_input('ID')
            new_type = st.selectbox('Type', ['string', 'float', 'int'])
            new_value = st.text_input('Value')
            new_folder = st.checkbox('Folder')

            submit_button = st.form_submit_button(label='Add Metadata')
            if submit_button:
                # Validate ID uniqueness and value type consistency
                if new_id in st.session_state.metadata_values:
                    st.error('This ID already exists. Please use a unique ID.')
                elif not validate_value_with_type(new_value, new_type):
                    st.error('The type of the value does not match the selected type. Please correct it.')
                elif new_id and new_value:
                    # Update db_metadata and session state only if validations pass
                    if new_id not in st.session_state.metadata_values:
                        st.session_state.metadata_values[new_id] = [(new_value, new_type)]
                        if new_folder:
                            st.session_state.selected_for_folder.append(new_id)

                    # Clear form fields after successful submission
                    new_id = ""
                    new_value = ""
                    new_folder = False

                    st.success('Metadata added successfully!')
                else:
                    st.error('Both ID and Value must be provided.')
        
    with st.expander("Preview"):
         # Display selected folder order
        st.write("Metadata overview with folder hierarchy (Date is always first):")

        # Display metadata with type information directly associated with each ID
        formatted_metadata = [{
            'root_tag': 'pulsestorm',
            'id': key,
            'type': values[0][1],  # Access type of the first tuple
            'value': values[0][0],  # Access value of the first tuple
            'Folder': (st.session_state.selected_for_folder.index(key) + 1 if key in st.session_state.selected_for_folder else None)
        } for key, values in st.session_state.metadata_values.items() if values[0][0] is not None]

        # Remove root_tag from the table and set ID as index
        if formatted_metadata:
            df = pd.DataFrame(formatted_metadata)
            df = df.drop(columns=['root_tag'])
            df.set_index('id', inplace=True)
            # Change folder column to int
            df['Folder'] = df['Folder'].astype('Int64')
            st.table(df)
        else:
            st.write("No metadata loaded.")

        if st.button("Upload Files"):
            if all_files_metadata:  # Check if any files have been selected and metadata processed
                num_files = len(all_files_metadata)
                st.write(f"{num_files} Files will be uploaded to '{soac_folder}' with the following folder hierarchy:")
                
                # Display the selected metadata keys that are marked for folder creation
                selected_metadata = [key for key in st.session_state.selected_for_folder]
                st.write(f"Metadata hierarchy: {selected_metadata}")

                # Remove 'Folder' key from each dictionary in the list
                for entry in formatted_metadata:
                    entry.pop('Folder')

                # Process each file and their associated metadata
                for file, metadata in all_files_metadata.items():

                    # Combined metadata by merging corresponding dictionaries
                    combined_metadata = formatted_metadata + metadata


                    folder_path = []
                    for folder in st.session_state.selected_for_folder:
                        # Search for this id in the combined_metadata list
                        for data in combined_metadata:
                            if folder in data['id']:
                                # Assuming the index of the id in the 'id' list corresponds to the index in the 'value' list
                                folder_path.append(data['value'])
                                            
                    
                    # Create the folder path based on selected hierarchy
                    print(f"Folder Path: {folder_path}")
                    path_directory = os.path.join(soac_folder, *folder_path)
                    print(f"Path Directory: {path_directory}")
                    # Add filename to the path
                    path_directory = os.path.join(path_directory, os.path.basename(file))
                    
                    st.write(f"File: {file} is being uploaded to: {path_directory}")

                    append_metadata_tags(file, path_directory, combined_metadata)

                    db_metadata = load_filament_metadata(soac_folder)
            else:
                st.write("No files have been processed for upload.")


        


def validate_value_with_type(value, value_type):
    """Helper function to validate the type of the value matches the expected type."""
    try:
        if value_type == 'int':
            int(value)
        elif value_type == 'float':
            float(value)
        elif value_type == 'string':
            str(value)
        return True
    except ValueError:
        return False

    
        
    