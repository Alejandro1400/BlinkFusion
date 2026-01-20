import streamlit as st
from UI.SOAC.filament_dashboard_ui import run_filament_dashboard_ui
from UI.SOAC.filament_preprocessing_ui import run_filament_preprocessing_ui
from UI.SOAC.filament_processing_ui import run_filament_processing_ui
from UI.STORM.storm_dashboard_ui import STORMDashboard
from UI.STORM.storm_processing_ui import STORMProcessor
from UI.STORM.storm_single_image_dash_ui import STORMSingleImageDashboard
from UI.sidebar import setup_sidebar
#from UI.STORM.storm_dashboard_ui import run_storm_dashboard_ui
from UI.STORM.storm_preprocessing_ui import run_storm_preprocessing_ui
from UI.welcome_ui import show_welcome

# Streamlit page configuration
st.set_page_config(
    page_title="PulseSTORM", 
    page_icon=":microscope:", 
    layout="wide"
)


def display_content():
    """
    Display the content of the app based on the selected option from the sidebar.
    The content varies depending on the selected category and operation.
    """
    if 'selected_option' in st.session_state:
        selected_option = st.session_state.selected_option

        # Welcome section
        if "Welcome" in selected_option:
            st.title("Welcome to PulseSTORM")
            st.write("Please configure the file database connection and then select a category from the sidebar.")
            show_welcome()

        # Analysis sections
        elif "Analysis" in selected_option:
                section, operation = selected_option.split(' - ')
                st.title(f"{section} {operation}")

                # Handle Filament Analysis
                if section == "Filament Analysis":
                    if operation == "Preprocessing":
                        if 'soac_database_folder' in st.session_state:
                            run_filament_preprocessing_ui(st.session_state.soac_database_folder)
                        else:
                            st.error("SOAC database folder not set. Please configure it in the sidebar.")
                    elif operation == "Batch Processing":
                        if 'soac_database_folder' in st.session_state:
                            run_filament_processing_ui(st.session_state.soac_database_folder, 
                                                        st.session_state.soac_config_path,
                                                        st.session_state.soac_parameter_file,
                                                        st.session_state.soac_executable_path)
                        else:
                            st.error("SOAC database folder not set. Please configure it in the sidebar.")
                    elif operation == "Dashboard":
                        if 'soac_database_folder' in st.session_state:
                            run_filament_dashboard_ui(st.session_state.soac_database_folder)
                        else:
                            st.error("SOAC database folder not set. Please configure it in the sidebar.")
                    else:
                        st.warning(f"Operation '{operation}' is not yet implemented for Filament Analysis.")

                # Handle STORM Analysis
                elif section == "STORM Analysis":
                    if operation == "Preprocessing":
                        if 'storm_database_folder' in st.session_state:
                            run_storm_preprocessing_ui(st.session_state.storm_database_folder)
                        else:
                            st.error("STORM database folder not set. Please configure it in the sidebar.")
                    elif operation == "Batch Processing":
                        if 'storm_database_folder' in st.session_state:
                            storm_proc = STORMProcessor(st.session_state.storm_database_folder)
                            storm_proc.run_processing_ui()
                        else:
                            st.error("STORM database folder not set. Please configure it in the sidebar.")
                    elif operation == "Comparison Dashboard":
                        if 'storm_database_folder' in st.session_state:
                            storm_dash = STORMDashboard(st.session_state.storm_database_folder)
                            storm_dash.run_storm_dashboard_ui()
                        else:
                            st.error("STORM database folder not set. Please configure it in the sidebar.")
                    elif operation == "Single Image Dashboard":
                        if 'storm_database_folder' in st.session_state:
                            storm_si_dash = STORMSingleImageDashboard(st.session_state.storm_database_folder)
                            storm_si_dash.run_storm_si_dashboard_ui()
                        else:
                            st.error("STORM database folder not set. Please configure it in the sidebar.")
                    else:
                        st.warning(f"Operation '{operation}' is not yet implemented for STORM Analysis.")
                else:
                    st.error(f"Unrecognized section: '{section}'")
    else:
        st.subheader("Select a category from the sidebar to get started.")

def main():
    """
    Main function of the Streamlit app.
    - Sets up the sidebar.
    - Checks if an option is selected and displays the corresponding content.
    """
    # Initialize the sidebar with options
    setup_sidebar()

    # Display the main content
    display_content()

if __name__ == "__main__":
    main()
