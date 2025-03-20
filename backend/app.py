from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
import requests
import base64
import os
import random
from flask_cors import CORS
from deepface import DeepFace
from PIL import Image
import io
import re
import cv2

app = Flask(__name__)
CORS(app)

# Spotify credentials (replace with your own)
SPOTIFY_CLIENT_ID = '1f3ca3c9c91c4f3c8927f36094d3ed81'
SPOTIFY_CLIENT_SECRET = 'f11d2dc642ee4119b98e775d6a4bce46'

# Function to get Spotify API token
def get_spotify_token():
    auth_url = 'https://accounts.spotify.com/api/token'
    auth_header = base64.b64encode(f"{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}".encode()).decode()
    headers = {
        'Authorization': f'Basic {auth_header}',
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    data = {'grant_type': 'client_credentials'}
    response = requests.post(auth_url, headers=headers, data=data)
    return response.json().get('access_token')

# Function to get a playlist based on mood
def get_playlist_for_mood(mood):
    token = get_spotify_token()
    search_url = 'https://api.spotify.com/v1/search'
    headers = {'Authorization': f'Bearer {token}'}
    keywords = ['global', 'international', 'top hits', 'happy', 'sad', 'relaxing']
    keyword = random.choice(keywords)
    
    params = {
        'q': f'{mood} {keyword} playlist',
        'type': 'playlist',
        'limit': 1
    }
    response = requests.get(search_url, headers=headers, params=params)
    playlists = response.json().get('playlists', {}).get('items', [])
    if playlists:
        return playlists[0]['external_urls']['spotify']
    return None

# Function to get a motivational quote
def get_quote():
    response = requests.get('https://zenquotes.io/api/random')
    if response.status_code == 200:
        return response.json()[0]['q'] + " -" + response.json()[0]['a']
    return "Stay positive and keep going!"

# Function to analyze mood from image
def analyze_mood(image_data):
    try:
        image_bytes = re.sub('^data:image/.+;base64,', '', image_data)
        image = Image.open(io.BytesIO(base64.b64decode(image_bytes)))
        
        # Convert PIL image to NumPy array
        img_np = np.array(image)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)  # Convert to OpenCV format
        
        # Perform facial emotion analysis
        result = DeepFace.analyze(img_np, actions=['emotion'], enforce_detection=False)
        
        # Extract dominant mood
        mood = result[0]['dominant_emotion']
        return mood
    except Exception as e:
        return str(e)

# Route to predict mood from an image
@app.route('/predict_mood', methods=['POST'])
def predict_mood():
    data = request.json
    image_data = data.get('image', '')

    if not image_data:
        return jsonify({'error': 'No image provided.'}), 400

    mood = analyze_mood(image_data)
    playlist_url = get_playlist_for_mood(mood)
    quote = get_quote()

    return jsonify({
        'mood': mood,
        'playlist': playlist_url,
        'quote': quote
    })

if __name__ == '__main__':
    app.run(debug=True)
