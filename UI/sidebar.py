import streamlit as st

def setup_sidebar():
    with st.sidebar:
        # Select analysis type
        selected_analysis_type = st.selectbox("Select Analysis Type", ["SOAC Filament Analysis", "PulseSTORM Analysis"], key="analysis_type")

        # Configuration Expander for Filament
        if selected_analysis_type == "SOAC Filament Analysis":
            with st.expander("Filament Storage Configuration", expanded=False):
                filament_folder = st.text_input("Enter Folder Path for Filament Data", key="filament_folder_path")
                if st.button("Set Filament Path", key="set_filament_path"):
                    st.session_state.filament_folder = filament_folder
                    st.success(f"Folder path set to: {filament_folder}")

        # Configuration Expander for pulseSTORM
        elif selected_analysis_type == "PulseSTORM Analysis":
            with st.expander("pulseSTORM Storage Configuration", expanded=False):
                pulseSTORM_folder = st.text_input("Enter Folder Path for pulseSTORM Data", key="pulseSTORM_folder_path")
                if st.button("Set PulseSTORM Path", key="set_pulseSTORM_path"):
                    st.session_state.pulseSTORM_folder = pulseSTORM_folder
                    st.success(f"Folder path set to: {pulseSTORM_folder}")

        # Select page for both analyses
        selected_page = st.selectbox("Select Page", ["History", "Compare"], key="selected_page")

        # Start button to confirm selection
        if st.button("Start", key="start_button"):
            return {
                "analysis_type": selected_analysis_type,
                "selected_page": selected_page
            }

    return None
