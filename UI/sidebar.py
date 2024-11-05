import streamlit as st

def setup_sidebar():
    st.sidebar.title("PulseSTORM")
    st.sidebar.text("Wiesner Group")  

    # Main categories
    general_options = ["Welcome", "Filament Analysis", "STORM Analysis", "Database"]
    selected_general = st.sidebar.radio("Select Category", general_options)

    # Display secondary options based on the main selection
    if selected_general == "Welcome":
        st.session_state.selected_option = "Welcome"
    elif selected_general == "Filament Analysis":
        filament_options = ["Preprocessing", "Processing", "Batch", "Dashboard"]
        selected_filament = st.sidebar.radio("Select Operation:", filament_options)
        st.session_state.selected_option = f"Filament Analysis - {selected_filament}"
    elif selected_general == "STORM Analysis":
        storm_options = ["Preprocessing", "Processing", "Batch", "Dashboard"]
        selected_storm = st.sidebar.radio("Select Operation:", storm_options)
        st.session_state.selected_option = f"STORM Analysis - {selected_storm}"
    elif selected_general == "Database":
        database_options = ["Overview", "Upload", "Modify"]
        selected_database = st.sidebar.radio("Manage Database:", database_options)
        st.session_state.selected_option = f"Database - {selected_database}"

    # If non is selected, default to Welcome
    if "selected_option" not in st.session_state:
        st.session_state.selected_option = "Welcome"

