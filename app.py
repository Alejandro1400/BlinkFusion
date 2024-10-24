# app.py
import streamlit as st
from UI.soac_filament_ui import run_soac_ui
#from UI.storm_ui import run_storm_ui
from UI.sidebar import setup_sidebar
from UI.storm_ui import run_storm_ui


def main():
    st.title("PulseSTORM")

    # Setup the sidebar with configurations
    config = setup_sidebar()

    if config is not None:
        if config["analysis_type"] == "SOAC Filament Analysis":
            st.header("SOAC Filament Analysis")
            # Check if the folder and file are available and proceed
            if 'filament_folder' in st.session_state:
                run_soac_ui(st.session_state.filament_folder)
            else:
                st.error("Filament folder or file is not set. Please configure the path name.")

        elif config["analysis_type"] == "PulseSTORM Analysis":
            st.header("PulseSTORM Analysis")
            # Check if the folder and file are available and proceed
            if 'pulseSTORM_folder' in st.session_state:
                run_storm_ui(st.session_state.pulseSTORM_folder)
            else:
                st.error("pulseSTORM folder or file is not set. Please configure the path name.")

if __name__ == "__main__":
    main()