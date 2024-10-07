# app.py
import streamlit as st
from UI.soac_filament import run_soac_ui
from UI.sidebar import setup_sidebar


def main():
    st.title("PulseSTORM")


    # Setup the sidebar with configurations
    config = setup_sidebar()

    if config["selected_page"] == "SOAC Filament Analysis":
        st.header("SOAC Filament Analysis")
        # Check if the box_client is available and proceed
        if 'fila_box_client' in st.session_state and st.session_state.fila_box_client is not None:
            run_soac_ui(st.session_state.fila_box_client)
        else:
            st.error("Box client is not connected. Please configure the connection correctly.")
    elif config["selected_page"] == "PulseSTORM Analysis":
        st.header("PulseSTORM Analysis")
        # Implement PulseSTORM specific functionality

    # Maybe add further dynamic handling based on config selections

if __name__ == "__main__":
    main()
