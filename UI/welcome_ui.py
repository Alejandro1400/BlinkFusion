import streamlit as st
from Data_access.file_explorer import find_items

def show_welcome():
    """
    Displays the welcome page of the app.
    Includes introduction and configuration options for SOAC Filament and STORM Analysis file paths.
    """
    
    # Introduction Section
    with st.expander("Introduction"):
        st.write("Welcome to the app! This app is designed to help you analyze and visualize data from SOAC Filament and STORM Analysis.")

    # SOAC Filament File Configuration Section
    with st.expander("SOAC Filament File Configuration"):
        st.write("Configure the folder and files used in SOAC Analysis.")

        # Automatically search for required SOAC files
        config_path = find_items(item='ridge_detector_param.json', is_folder=False)
        parameter_file = find_items(item='batch_parameters.txt', is_folder=False)
        executable_path = find_items(item='batch_soax_v3.7.0.exe', is_folder=False)

        # Input for Filament Database Folder
        st.session_state.soac_database_folder = st.text_input(
            "Database Filament Folder",
            st.session_state.get('soac_database_folder', ""),
            help="Path to the folder containing SOAC Filament files."

        )

        # Inputs for each required file path, with session state updates
        st.session_state.soac_config_path = st.text_input(
            "Config Path",
            config_path if config_path else "",
            help="Path to the 'ridge_detector_param.json' configuration file."
        )

        st.session_state.soac_parameter_file = st.text_input(
            "Parameter File",
            parameter_file if parameter_file else "",
            help="Path to the 'batch_parameters.txt' file."
        )

        st.session_state.soac_executable_path = st.text_input(
            "Executable Path",
            executable_path if executable_path else "",
            help="Path to the 'batch_soax_v3.7.0.exe' executable file."
        )

    # STORM Analysis File Configuration Section
    with st.expander("STORM Analysis File Configuration"):
        st.write("Configure the folder and files used in STORM Analysis.")

        # Input for STORM Database Folder
        st.session_state.storm_database_folder = st.text_input(
            "Database STORM Folder",
            st.session_state.get('storm_database_folder', ""),
            help="Path to the folder containing STORM Analysis files."
        )
        