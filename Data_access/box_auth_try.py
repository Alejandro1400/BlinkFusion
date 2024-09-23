from flask import Flask, request, redirect
from boxsdk import OAuth2, Client

app = Flask(__name__)

# Replace 'YOUR_CLIENT_ID' and 'YOUR_CLIENT_SECRET' with your actual client ID and client secret
CLIENT_ID = 'u51vrokpq5em23cvmcnrxgbvm5o1td9u'
CLIENT_SECRET = '9MvRANbovdY4vgCPxakdJKcu4cTjcR2q'
REDIRECT_URI = 'https://localhost:5000/oauth2callback'

def store_tokens(access_token, refresh_token):
    # Here you would store the tokens securely
    print("Access Token:", access_token)
    print("Refresh Token:", refresh_token)

oauth = OAuth2(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    store_tokens=store_tokens,
)

@app.route('/')
def index():
    auth_url, csrf_token = oauth.get_authorization_url(REDIRECT_URI)
    # Store the CSRF token securely to validate later
    app.secret_key = csrf_token  # Simple example; use a more secure method in production
    return redirect(auth_url)

@app.route('/oauth2callback')
def oauth2callback():
    code = request.args.get('code')
    state = request.args.get('state')
    
    # Retrieve the CSRF token from a secure storage (adjust this accordingly)
    assert state == app.secret_key, "CSRF token mismatch"

    access_token, refresh_token = oauth.authenticate(code)
    client = Client(oauth)
    user_info = client.user().get()
    return f"You have been successfully authenticated! User ID: {user_info.id}"

if __name__ == '__main__':
    app.run(debug=True)


