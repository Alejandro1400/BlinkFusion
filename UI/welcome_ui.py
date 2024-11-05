import streamlit as st

from Data_access.file_explorer import find_items

def show_welcome():
    
    # Explanation Section
    with st.expander("Introduction"):
        st.write("In this app you can use SOAC.")

    # 
    with st.expander("SOAC Filament File Configuration"):
        st.write("Configure the folder and files used in SOAC Analysis")

        st.session_state.soac_database_folder = st.text_input("Database Filament Folder", st.session_state.soac_database_folder if 'soac_database_folder' in st.session_state else "")

        config_path = find_items(item='ridge_detector_param.json', is_folder=False)
        parameter_file = find_items(item='batch_parameters.txt', is_folder=False)
        executable_path = find_items(item='batch_soax_v3.7.0.exe', is_folder=False)

        # Set up session state for paths
        st.session_state.soac_config_path = st.text_input(
            "Config Path",
            config_path if config_path else ""
        )

        st.session_state.soac_parameter_file = st.text_input(
            "Parameter File",
            parameter_file if parameter_file else ""
        )

        st.session_state.soac_executable_path = st.text_input(
            "Executable Path",
            executable_path if executable_path else ""
        )


    with st.expander("STORM Analysis File Configuration"):
        st.write("Configure the folder and files used in STORM Analysis")

        st.session_state.storm_database_folder = st.text_input("Database STORM Folder", st.session_state.storm_database_folder if 'storm_database_folder' in st.session_state else "")
        