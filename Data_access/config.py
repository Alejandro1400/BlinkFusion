import streamlit as st

from Data_access.box_connection import box_signin

def connect_storage(auth_method, auth_file=None, developer_token=None):
    """
    Connects to the Box storage using either an authentication file or a developer token.
    
    Args:
    auth_method (str): Method of authentication ("Authentication File" or "Developer Token")
    auth_file (UploadedFile, optional): The uploaded Box authentication file.
    developer_token (str, optional): The developer token for Box API.

    Returns:
    bool: True if connection was successful, False otherwise.
    """
    if auth_method == "Authentication File" and auth_file is not None:
        # Assuming auth_file is an uploaded file, you would need to save it temporarily or process its contents
        box_client, user_login = box_signin(auth_file_path=auth_file)
    elif auth_method == "Developer Token" and developer_token:
        box_client, user_login = box_signin(developer_token=developer_token)
    else:
        st.error("Authentication details are missing.")
        return 

    if box_client:
        st.success(f"Connected to Box as {user_login}")
        return box_client
    else:
        st.error("Failed to connect to Box.")
