from boxsdk import OAuth2, Client
import tkinter as tk
from tkinter import filedialog

TOKEN = 'RKGvqKckeOSo3sLdNb66vHJsHx59GHMJ'

oauth2 = OAuth2(None, None, access_token=TOKEN)
box = Client(oauth2)

me = box.user().get()
print('Logged in to Box as:', me.login)

# Define the root folder ID (use 0 for the root)
MY_FOLDER_ID = 0
my_folder = box.folder(MY_FOLDER_ID).get()
print('Current folder:', my_folder)

# Function to find a folder by name
def find_folder_by_name(folder_name, parent_folder_id=0):
    parent_folder = box.folder(parent_folder_id).get()
    items = parent_folder.get_items()
    for item in items:
        if item.type == 'folder' and item.name == folder_name:
            return item
    return None

# Recursive function to print folder contents hierarchically
def print_folder_contents(folder, prefix=""):
    items = folder.get_items()
    for item in items:
        print(prefix + f"{item.name} ({item.type})")
        if item.type == 'folder':
            # Recursively print the contents of the sub-folder
            print_folder_contents(box.folder(item.id), prefix + "  ")

# Function to download a .tif file
def download_first_tif_file(folder):
    items = folder.get_items()
    for item in items:
        if item.type == 'file' and item.name.endswith('.tif'):
            downloaded_file = item.get().content()
            with open(item.name, 'wb') as open_file:
                open_file.write(downloaded_file)
            print(f"Downloaded {item.name}")
            break

# Function to upload a file
def upload_file_to_folder(folder_id):
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename()
    if file_path:
        file_name = file_path.split('/')[-1]
        with open(file_path, 'rb') as file_stream:
            box.folder(folder_id).upload_stream(file_stream, file_name)
        print(f"Uploaded {file_name} to Folder ID {folder_id}")
    else:
        print("No file selected")

# Find and print the contents of the "For Alejandro" folder hierarchically
alejandro_folder = find_folder_by_name("For Alejandro", MY_FOLDER_ID)
if alejandro_folder:
    print('For Alejandro folder found:')
    print_folder_contents(alejandro_folder, "  ")
    download_first_tif_file(alejandro_folder)
    upload_file_to_folder(alejandro_folder.id)
else:
    print("Folder 'For Alejandro' not found.")
