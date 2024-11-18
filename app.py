# app.py
import streamlit as st
from UI.SOAC.filament_dashboard_ui import run_filament_dashboard_ui
from UI.SOAC.filament_preprocessing_ui import run_filament_preprocessing_ui
from UI.sidebar import setup_sidebar
from UI.STORM.storm_dashboard_ui import run_storm_dashboard_ui
from UI.STORM.storm_preprocessing_ui import run_storm_preprocessing_ui
from UI.welcome_ui import show_welcome

st.set_page_config(page_title="PulseSTORM", page_icon=":microscope:", layout="wide")

def display_content():
    # Display content based on the selected option
    if 'selected_option' in st.session_state:
        if "Welcome" in st.session_state.selected_option:
            st.title("Welcome to PulseSTORM")
            st.write("Please select a category from the sidebar to begin.")
            show_welcome()
        elif "Analysis" in st.session_state.selected_option:
            section, operation = st.session_state.selected_option.split(' - ')
            st.title(f"{section} {operation}")
            
            if section == "Filament Analysis":
                if operation == "Preprocessing":
                    run_filament_preprocessing_ui(st.session_state.soac_database_folder)
                elif operation == "Processing":
                    st.write("Processing Filament Analysis")
                elif operation == "Batch":
                    st.write("Batch Filament Analysis")
                elif operation == "Dashboard":
                    run_filament_dashboard_ui(st.session_state.soac_database_folder)
            elif section == "STORM Analysis":
                if operation == "Preprocessing":
                    run_storm_preprocessing_ui(st.session_state.storm_database_folder)
                    print("Preprocessing STORM Analysis")
                elif operation == "Processing":
                    st.write("Processing STORM Analysis")
                elif operation == "Batch":
                    st.write("Batch STORM Analysis")
                elif operation == "Dashboard":
                    run_storm_dashboard_ui(st.session_state.storm_database_folder)

        elif "Database" in st.session_state.selected_option:
            _, action = st.session_state.selected_option.split(' - ')
            st.title(f"Database {action}")
            st.write(f"Details for database {action.lower()} will be displayed here.")

def main():
    setup_sidebar()

    # Check if an option has been made and display relevant content
    if 'selected_option' in st.session_state:
        display_content()
    else:
        st.subheader("Select a category from the sidebar to get started.")

if __name__ == "__main__":
    main()