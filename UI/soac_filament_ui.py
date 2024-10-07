# soac_filament_analysis.py
import streamlit as st

from Data_access import box_connection

def run_soac_ui(box_client):
    st.subheader("Running SOAC Filament Analysis")
    # Example function to demonstrate using box_client
    if box_client:
        database_json = box_connection.database_stats(box_client, root_dir='Filament Data')
        st.write("Data loaded successfully from Box:", database_json)
    else:
        st.error("Box client is not connected. Please check the connection.")
