from boxsdk import OAuth2, Client

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

# Find and print the contents of the "For Alejandro" folder hierarchically
alejandro_folder = find_folder_by_name("For Alejandro", MY_FOLDER_ID)
if alejandro_folder:
    print('For Alejandro folder found:')
    print_folder_contents(alejandro_folder, "  ")
else:
    print("Folder 'For Alejandro' not found.")
