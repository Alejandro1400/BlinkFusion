# app.py
import streamlit as st
from UI.soac_ui import run_soac_ui
#from UI.storm_ui import run_storm_ui
from UI.sidebar import setup_sidebar
from UI.storm_ui import run_storm_ui

def main():
    st.title("PulseSTORM")

    # Call setup sidebar
    setup_sidebar()

    # Proceed with analysis if the Start button was clicked
    if 'start_analysis' in st.session_state and st.session_state.start_analysis:
        analysis_type = st.session_state.selected_analysis_type
        st.header(f"{analysis_type} Analysis")

        folder_key = 'filament_folder' if analysis_type == "SOAC Filament Analysis" else 'pulseSTORM_folder'
        
        if folder_key in st.session_state and st.session_state[folder_key]:
            # Assuming these functions handle the UI for each analysis type
            run_function = run_soac_ui if analysis_type == "SOAC Filament Analysis" else run_storm_ui
            run_function(st.session_state[folder_key])
        else:
            st.error(f"{analysis_type} folder or file is not set. Please configure the path name.")
    else:
        st.info("Please configure the analysis type and path before starting.")

if __name__ == "__main__":
    main()