from boxsdk import OAuth2, Client

TOKEN = 'WzWFCnbGepKEEjdGpdRahIIA5QMXUJur'

oauth2 = OAuth2(None, None, access_token=TOKEN)
box = Client(oauth2)

me = box.user().get()
print('logged in to Box as', me.login)

print(me.response_object)

MY_FOLDER_ID = 0
my_folder = box.folder(MY_FOLDER_ID).get()
print('current folder', my_folder)

items = my_folder.get_items()
for item in items:
    print(item.name, item.type)