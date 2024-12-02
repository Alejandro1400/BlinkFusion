import os
import time
import pandas as pd
import streamlit as st

from Data_access.file_explorer import find_items, find_valid_folders
from Data_access.metadata_manager import (
    aggregate_metadata_info,
    append_metadata_tags,
    process_tiff_metadata,
    read_tiff_metadata
)

@st.cache_data
def load_filament_metadata(soac_folder, reload_trigger):
    """
    Loads metadata for filaments from a given folder, ensuring only valid folders and .tif files are processed.

    Args:
        soac_folder (str): The base folder containing SOAC data.
        reload_trigger (int): A value to force cache invalidation when reloading is required.

    Returns:
        dict: A dictionary of metadata where keys are IDs and values are lists of (value, type) tuples.
    """
    # Initialize progress and status
    status_text = st.text("Loading filaments data...")
    progress_bar = st.progress(0, text="Searching for valid folders...")

    # Identify valid folders containing the required .tif files
    try:
        valid_folders = find_valid_folders(
            soac_folder,
            required_files={'.tif'}
        )
        valid_folders = [folder for folder in valid_folders if not folder.endswith('ROIs')]
    except Exception as e:
        st.error(f"Error finding valid folders: {e}")
        status_text.empty()
        progress_bar.empty()
        return {}

    # Initialize metadata storage
    database_metadata = {}

    # Process each valid folder
    for index, folder in enumerate(valid_folders):
        time.sleep(0.01)  # Allow UI updates for the progress bar

        try:
            # Find a .tif file in the folder
            tif_file = find_items(
                base_directory=folder,
                item='.tif',
                is_folder=False,
                check_multiple=False,
                search_by_extension=True
            )

            if not tif_file:
                st.warning(f"No .tif file found in {folder}. Skipping...")
                continue

            # Read metadata from the .tif file
            metadata = read_tiff_metadata(tif_file, root_tag='pulsestorm')

            # Process and aggregate metadata
            for item in metadata:
                entry = (item['value'], item['type'])  # Value and type tuple
                if item['id'] in database_metadata:
                    if entry not in database_metadata[item['id']]:
                        database_metadata[item['id']].append(entry)
                else:
                    database_metadata[item['id']] = [entry]

        except Exception as e:
            st.error(f"Error processing folder {folder}: {e}")

        # Update the progress bar periodically
        if index % 5 == 0 or index == len(valid_folders) - 1:
            progress = (index + 1) / len(valid_folders)
            progress_bar.progress(progress, text=f"Processing folder {index + 1} of {len(valid_folders)}")

    # Clear progress indicators
    status_text.empty()
    progress_bar.empty()

    return database_metadata


def reload_metadata():
    """
    Increment the reload trigger in the session state.
    This can be used to force a cache reload for the metadata.
    """
    if 'reload_trigger' in st.session_state:
        st.session_state.reload_trigger += 1
    else:
        st.session_state.reload_trigger = 1


def run_filament_preprocessing_ui(soac_folder):
    """
    Displays the UI for filament preprocessing, including loading metadata,
    handling user input for uploading files or folders, and displaying metadata summaries.

    Args:
        soac_folder (str): Path to the SOAC folder for loading metadata.
    """

    # Validate SOAC folder
    if soac_folder is None or not os.path.exists(soac_folder):
        st.error("Please select a valid SOAC folder in the Welcome Menu.")
        return

    # Initialize session state variables if not already set
    st.session_state.setdefault('reload_trigger', 0)
    st.session_state.setdefault('selected_for_folder', ["Date"])
    st.session_state.setdefault('metadata_values', {})
    st.session_state.setdefault('show_add_form', False)

    # Load metadata for the given folder
    db_metadata = load_filament_metadata(soac_folder, st.session_state.reload_trigger)

    # Input for upload path
    upload_path = st.text_input("Enter the path to the folder or file you wish to upload.", 
                                help="Path to the folder or file containing filament .tif files.")

    with st.expander("Tif File Metadata"):
        if upload_path:
            all_files_metadata = {}

            # Check if the upload path is a folder
            if os.path.isdir(upload_path):
                st.write("Folder selected. All assigned metadata values will apply to all files in the folder.")

                # Find all `.tif` files in the folder
                files = find_items(
                    base_directory=upload_path, 
                    item='.tif', 
                    is_folder=False, 
                    check_multiple=True, 
                    search_by_extension=True
                )

                # Process metadata for each file in the folder
                for file in files:
                    try:
                        tif_metadata = process_tiff_metadata(file)
                        all_files_metadata[file] = tif_metadata
                    except Exception as e:
                        st.warning(f"Error processing metadata for file {file}: {e}")

            # Check if the upload path is a file
            elif os.path.isfile(upload_path):
                st.write("File selected. Metadata will be added to the file.")

                try:
                    tif_metadata = process_tiff_metadata(upload_path)
                    all_files_metadata[upload_path] = tif_metadata
                except Exception as e:
                    st.error(f"Error processing metadata for file {upload_path}: {e}")
                    return

            else:
                st.error("The provided path is neither a valid file nor a folder. Please check the input.")
                return

            # Display summary of metadata as a table
            if all_files_metadata:
                summary_df = aggregate_metadata_info(all_files_metadata)

                st.write(f"Files to upload: {len(all_files_metadata)}")
                st.write("Tif Metadata to be added (Count displays how many files have values):")
                st.table(summary_df)
            else:
                st.warning("No metadata could be extracted from the provided path.")

        else:
            st.info("Please enter a folder or file path to continue.")



    with st.expander("Database Metadata"):
        """
        Displays metadata information found in the database and allows users to manage metadata values.
        Provides options for selecting, adding, or associating metadata to folders.
        """

        st.write(f"Metadata found in Database: {soac_folder}")

        # Check if the database has any metadata
        if len(db_metadata) == 0:
            st.write("No metadata found in the database. Ensure that metadata files exist and try reloading.")
        else:
            st.info("Select metadata values from the database. Use the 'Add new...' option to include custom values.")

        for metadata_id, entries in db_metadata.items():
            with st.container():
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # Prepare current values list for the selectbox
                    current_values = [entry[0] for entry in entries]
                    current_values.append(None)  # Allow "None" as a valid selection
                    current_values.append("Add new...")  # Option to add a new value

                    # Attempt to get the currently selected value or default to the first available value
                    current_selection = st.session_state.metadata_values.get(
                        metadata_id, 
                        [(current_values[0], entries[0][1])] if entries else [(None, None)]
                    )[0][0]

                    # Ensure the current selection is part of the dropdown options
                    if current_selection not in current_values:
                        current_values.insert(0, current_selection)

                    # Dropdown to select metadata value
                    selected_value = st.selectbox(
                        f"{metadata_id} ({entries[0][1]})",
                        current_values,
                        index=current_values.index(current_selection),
                        key=f"{metadata_id}_value_select"
                    )

                    # Handle the addition of a new value
                    if selected_value == "Add new...":
                        new_value = st.text_input(
                            "Enter new value",
                            key=f"{metadata_id}_new_value"
                        )
                        if new_value:
                            st.session_state.metadata_values[metadata_id] = [(new_value, entries[0][1])]
                            db_metadata[metadata_id].append((new_value, entries[0][1]))
                    elif selected_value is None:
                        st.session_state.metadata_values[metadata_id] = [(None, entries[0][1])]
                    else:
                        st.session_state.metadata_values[metadata_id] = [(selected_value, entries[0][1])]

                with col2:
                    # Checkbox to mark the metadata as folder-specific
                    folder_selected = st.checkbox(
                        "Folder",
                        key=f"{metadata_id}_folder",
                        value=metadata_id in st.session_state.selected_for_folder
                    )
                    if folder_selected:
                        if metadata_id not in st.session_state.selected_for_folder:
                            st.session_state.selected_for_folder.append(metadata_id)
                    else:
                        if metadata_id in st.session_state.selected_for_folder:
                            st.session_state.selected_for_folder.remove(metadata_id)
                        
    
    # Use an expander for the form to add new metadata
    with st.expander("Add Metadata"):
        st.info("Use this form to add custom metadata. Ensure each ID is unique, and the value matches the selected type.")

        with st.form(key='new_metadata_form'):
            # Input fields for adding metadata
            new_id = st.text_input(
                'ID',
                help="Enter a unique identifier for the metadata. It must not duplicate existing IDs."
            )
            new_type = st.selectbox(
                'Type',
                ['string', 'float', 'int'],
                help="Select the data type for the metadata value."
            )
            new_value = st.text_input(
                'Value',
                help="Enter the value for the metadata. Ensure it matches the selected type."
            )
            new_folder = st.checkbox(
                'Folder',
                help="Mark if this metadata should be used to create a folder hierarchy."
            )

            # Submit button for the form
            submit_button = st.form_submit_button(label='Add Metadata')
            if submit_button:
                # Validate uniqueness of ID and type consistency
                if new_id in st.session_state.metadata_values:
                    st.error('This ID already exists. Please use a unique ID.')
                elif not validate_value_with_type(new_value, new_type):
                    st.error('The type of the value does not match the selected type. Please correct it.')
                elif new_id and new_value:
                    # Add new metadata to the session state
                    st.session_state.metadata_values[new_id] = [(new_value, new_type)]
                    if new_folder:
                        st.session_state.selected_for_folder.append(new_id)

                    # Clear form fields after successful submission
                    st.success('Metadata added successfully!')
                else:
                    st.error('Both ID and Value must be provided.')

    # Use an expander for metadata preview
    with st.expander("Preview"):
        st.info("This section displays the metadata overview with folder hierarchy. You can review and upload files.")

        # Display metadata overview with folder hierarchy
        st.write("Metadata overview with folder hierarchy (Date is always first):")
        formatted_metadata = [
            {
                'root_tag': 'pulsestorm',
                'id': key,
                'type': values[0][1],
                'value': values[0][0],
                'Folder': (
                    st.session_state.selected_for_folder.index(key) + 1
                    if key in st.session_state.selected_for_folder
                    else None
                ),
            }
            for key, values in st.session_state.metadata_values.items()
            if values[0][0] is not None
        ]

        if formatted_metadata:
            df = pd.DataFrame(formatted_metadata)
            df = df.drop(columns=['root_tag'])
            df.set_index('id', inplace=True)
            df['Folder'] = df['Folder'].astype('Int64')
            st.table(df)
        else:
            st.write("No metadata loaded.")

        # Upload files with metadata
        if st.button("Upload Files"):
            if all_files_metadata:
                num_files = len(all_files_metadata)
                st.write(f"{num_files} Files will be uploaded to '{soac_folder}' with the following folder hierarchy:")
                
                selected_metadata = [key for key in st.session_state.selected_for_folder]
                st.write(f"Metadata hierarchy: {selected_metadata}")

                for entry in formatted_metadata:
                    entry.pop('Folder')

                # Progress bar for file uploads
                progress_bar = st.progress(0, text="Uploading files...")

                for file, metadata in all_files_metadata.items():
                    combined_metadata = formatted_metadata + metadata
                    folder_path = []

                    for folder in st.session_state.selected_for_folder:
                        for data in combined_metadata:
                            if folder in data['id']:
                                folder_path.append(data['value'])

                    path_directory = os.path.join(soac_folder, *folder_path, os.path.basename(file))
                    progress_bar.progress(
                        (list(all_files_metadata.keys()).index(file) + 1) / num_files,
                        text=f"Uploading file {file} ({list(all_files_metadata.keys()).index(file) + 1} of {num_files})"
                    )

                    append_metadata_tags(file, path_directory, combined_metadata)

                reload_metadata()
            else:
                st.warning("No files have been processed for upload.")



def validate_value_with_type(value, value_type):
    """
    Validates if the given value matches the expected data type.

    Args:
        value (str): The value to validate, typically provided as a string input.
        value_type (str): The expected type of the value. Supported types: 'int', 'float', 'string'.

    Returns:
        bool: True if the value matches the expected type, False otherwise.
    """
    try:
        if value_type == 'int':
            # Check if the value can be cast to an integer
            int(value)
        elif value_type == 'float':
            # Check if the value can be cast to a float
            float(value)
        elif value_type == 'string':
            # For string, ensure it can be converted and is not None
            str(value)
        else:
            # Unsupported type, return False
            return False
        return True
    except (ValueError, TypeError):
        return False


    
        
    