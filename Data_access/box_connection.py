from boxsdk import OAuth2, Client
import tkinter as tk
from tkinter import filedialog

# Function to load credentials
def load_credentials(file_path):
    # Obtain path file from file name

    credentials = {}
    try:
        with open(file_path, 'r') as file:
            for line in file:
                key, value = line.strip().split(' = ')
                credentials[key] = value.strip("'")
        return credentials
    except Exception as e:
        print(f"Failed to load credentials: {e}")
        return None


def box_signin(auth_file_path=None):
    # Try to find the auth_info file in the current directory
    credentials = load_credentials(auth_file_path)
    if credentials is None or not all(key in credentials for key in ['DEVELOPER_TOKEN']):
        print(f"Invalid credentials or '{auth_file_path}' file not found.")
        return None
    try:
        oauth2 = OAuth2(None, None, access_token=credentials['DEVELOPER_TOKEN'])
        box_client = Client(oauth2)
        print('Connected to Box as:', box_client.user().get().login)
        return box_client
    except Exception as error:
        print("Box authentication failed:", error)
        return None

