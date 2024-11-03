import streamlit as st

def setup_sidebar():
    with st.sidebar:
        # Select analysis type
        selected_analysis_type = st.selectbox("Select Analysis Type", ["SOAC Filament Analysis", "PulseSTORM Analysis"], key="selected_analysis_type")

        # Configuration Expander for Filament
        if selected_analysis_type == "SOAC Filament Analysis":
            with st.expander("Filament Storage Configuration", expanded=False):
                st.text_input("Enter Folder Path for Filament Data", key="filament_folder")

        # Configuration Expander for pulseSTORM
        elif selected_analysis_type == "PulseSTORM Analysis":
            with st.expander("pulseSTORM Storage Configuration", expanded=False):
                st.text_input("Enter Folder Path for pulseSTORM Data", key="pulseSTORM_folder")

        # Select page for both analyses
        st.selectbox("Select Page", ["History", "Compare"], key="selected_page")

        # Start button to confirm selection
        if st.button("Start", key="start_button"):
            # The configuration can be handled based on the need here, perhaps setting flags or processing data
            st.session_state.start_analysis = True  # This flag can be checked in main to proceed with analysis
