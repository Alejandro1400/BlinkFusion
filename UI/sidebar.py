import streamlit as st

from config import connect_storage

def setup_sidebar():
    with st.sidebar:
        selected_page = st.selectbox("Select Analysis Type", ["SOAC Filament Analysis", "PulseSTORM Analysis"])

        # Configuration Expander for SOAC Filament
        with st.expander("Filament Storage Configuration", expanded=False):
            filament_storage = st.radio("Connect Filament Storage via", ["Box", "File Explorer"], key="filament_storage")
            
            if filament_storage == "Box":
                auth_method = st.selectbox(
                    "Choose Authentication Method", 
                    ["Authentication File", "Developer Token"], 
                    key="box_auth_method_filament"
                )

                auth_file = None
                developer_token = None
                fila_box_client = None
                if auth_method == "Authentication File":
                    auth_file = st.text_input("Enter Box Auth File", key="filament_box_auth_file")
                
                elif auth_method == "Developer Token":
                    developer_token = st.text_input("Enter Box Developer Token", key="filament_box_developer_token")

                if st.button("Connect to Box for Filament", key="connect_box_filament"):
                    st.session_state.fila_box_client = connect_storage(auth_method, auth_file, developer_token)

            elif filament_storage == "File Explorer":
                filament_folder = st.text_input("Enter Folder Path for Filament Data", key="filament_folder_path")
                if st.button("Set Folder Path for Filament", key="set_folder_filament"):
                    # Assuming a function to validate or set the folder path
                    st.success(f"Folder path set: {filament_folder}")

        # Configuration Expander for pulseSTORM
        with st.expander("pulseSTORM Storage Configuration", expanded=False):
            pulseSTORM_storage = st.radio("Connect pulseSTORM Storage via", ["Box", "File Explorer"], key="pulseSTORM_storage")
            
            if pulseSTORM_storage == "Box":
                auth_method = st.selectbox(
                    "Choose Authentication Method", 
                    ["Authentication File", "Developer Token"], 
                    key="box_auth_method_pulseSTORM"
                )

                auth_file = None
                developer_token = None
                storm_box_client = None
                if auth_method == "Authentication File":
                    auth_file = st.file_uploader("Upload Box Authentication File", type=['txt'], key="pulseSTORM_box_auth_file")
                
                elif auth_method == "Developer Token":
                    developer_token = st.text_input("Enter Box Developer Token", key="pulseSTORM_box_developer_token")

                if st.button("Connect to Box for pulseSTORM", key="connect_box_pulseSTORM"):
                    st.session_state.storm_box_client = connect_storage(auth_method, auth_file, developer_token)

            elif pulseSTORM_storage == "File Explorer":
                # Open a file dialog to select folder path
                pulseSTORM_folder = st.text_input("Enter Folder Path for pulseSTORM Data", key="pulseSTORM_folder_path")
                if st.button("Set Folder Path for pulseSTORM", key="set_folder_pulseSTORM"):
                    # Assuming a function to validate or set the folder path
                    st.success(f"Folder path set: {filament_folder}")

    return {"selected_page": selected_page}
