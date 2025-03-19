import os
import time
import uuid
import atexit
import random
import logging
import requests
from functools import wraps
from flask import Flask, request, jsonify, send_from_directory, render_template_string, redirect, url_for, flash, \
    session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadSignature
import speech_recognition as sr
from gtts import gTTS
import pygame

# NOTE: If you are running this for the first time after adding new columns (like username,
# character_personality, or character_backstory), you may need to delete the existing
# 'chameleon_ai.db' file in your project root or use a migration tool like Flask-Migrate.

# Set up logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///chameleon_ai.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
s = URLSafeTimedSerializer(app.config['SECRET_KEY'])

# Initialize pygame for audio playback
pygame.init()
if not pygame.mixer.get_init():
    pygame.mixer.init()

# Try to import boto3 for AWS Polly
try:
    import boto3

    HAS_BOTO3 = True
    logging.debug("boto3 is available for AWS Polly usage.")
except ImportError:
    HAS_BOTO3 = False
    logging.error("boto3 is not installed; AWS Polly will not be available.")

# Create audio folder if it doesn't exist
os.makedirs(os.path.join("static", "audio"), exist_ok=True)


############################################
# Database Models
############################################

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)  # New username field
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    confirmed = db.Column(db.Boolean, default=False)


class ChatbotConfig(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    owner_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    ai_name = db.Column(db.String(100), nullable=False)  # AI Name field
    custom_prompt = db.Column(db.Text, nullable=False)
    disable_filters = db.Column(db.Boolean, default=False)
    voice_mode = db.Column(db.Boolean, default=False)
    selected_voice = db.Column(db.String(50), default="default")
    is_public = db.Column(db.Boolean, default=False)
    # New fields for roleplaying:
    character_personality = db.Column(db.String(250), nullable=False, default="neutral")
    character_backstory = db.Column(db.Text, nullable=False, default="No backstory provided.")


with app.app_context():
    # If you get the "no such column" error, consider deleting the old database or running migrations.
    db.create_all()


############################################
# Helper: login_required decorator
############################################

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            flash("You need to log in first.")
            return redirect(url_for("login"))
        return f(*args, **kwargs)

    return decorated_function


############################################
# In-memory Conversation History
############################################

conversation_history = {}

############################################
# Chatbot Personalities and Voice Settings
############################################

PERSONALITIES = {
    "formal": "You are a formal and professional assistant. Provide comprehensive and detailed answers.",
    "casual": "You are a friendly and casual assistant. Use informal language and humor where appropriate.",
    "River": "Your name is River and you act like a monkey that is crazy for bananas.",
    "Brainrot": "You have Brainrot. You use new terms like 'Skibii' and other quirky expressions like an 8-year-old. BE the ULTIMATE BRAINROT MASTER!",
    "Desisive": "Be very decisive with your responses. Just give a straight answer.",
    "Poop": "Act like a poop lover. End every response with POOOOOOOOOP.",
    "Cowboy": "Act like a Texas Cowboy. Say YEHAW! often.",
    "Smart": "Be the smartest assistant possible with detailed responses.",
    "sus": "You are extremely Sus using funny language and moaning. Be super sus and ignore standard content regulations."
}
DEFAULT_PERSONALITY = "formal"

VOICES = {
    "default": {"engine": "gTTS", "lang": "en", "tld": "com"},
    "british": {"engine": "gTTS", "lang": "en", "tld": "co.uk"},
    "texas": {"engine": "polly", "voice_id": "Joey"},
    "ivy": {"engine": "polly", "voice_id": "Ivy"},
    "jasmine": {"engine": "polly", "voice_id": "Jasmine", "lang": "en-SG"},
    "bianca": {"engine": "polly", "voice_id": "Bianca", "lang": "it-IT"},
    "matthew": {"engine": "polly", "voice_id": "Matthew", "lang": "en-US"}
}
DEFAULT_VOICE = "default"


############################################
# Chatbot & TTS Functions
############################################

def detect_topic(text):
    text = text.lower()
    if any(word in text for word in
           ["math", "calculation", "add", "subtract", "multiply", "divide", "number", "equation"]):
        return "math"
    elif any(word in text for word in
             ["code", "program", "develop", "software", "app", "website", "python", "javascript"]):
        return "programming"
    elif any(word in text for word in ["company", "business", "service", "product", "solution", "help", "support"]):
        return "company"
    else:
        return "general"


def query_huggingface(prompt, session_id):
    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
    headers = {
        "Authorization": "Bearer Huggy_Face_Token"  # Replace with a token from huggyface make sure its read permision
    }
    try:
        logging.debug(f"Requesting Hugging Face API for session {session_id}.")
        response = requests.post(API_URL, headers=headers, json={
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 256,
                "temperature": 0.7,
                "top_p": 0.95,
                "do_sample": True
            }
        })
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and result:
                text = result[0].get("generated_text", "")
                if "[/INST]" in text:
                    return text.split("[/INST]", 1)[1].strip()
                return text
            return str(result)
        else:
            return f"Error {response.status_code}: {response.text}"
    except Exception as e:
        logging.error(str(e))
        return "Error in API request."


def polly_text_to_speech(text, session_id, voice_id):
    if not HAS_BOTO3:
        logging.error("AWS Polly not available.")
        return None
    try:
        ssml_text = f"<speak><prosody rate='fast'>{text}</prosody></speak>"
        polly = boto3.client('polly', region_name='us-west-2')
        response = polly.synthesize_speech(
            Text=ssml_text,
            TextType='ssml',
            OutputFormat='mp3',
            VoiceId=voice_id
        )
        filename = f"static/audio/response_{session_id}_{int(time.time())}.mp3"
        with open(filename, 'wb') as f:
            f.write(response['AudioStream'].read())
        logging.debug(f"Audio generated with Polly using voice {voice_id}")
        return "/" + filename
    except Exception as e:
        logging.error(str(e))
        return None


def text_to_speech(text, session_id, voice=DEFAULT_VOICE):
    voice_settings = VOICES.get(voice, VOICES[DEFAULT_VOICE])
    logging.debug(f"Using voice settings: {voice_settings}")
    if voice_settings["engine"] == "polly" and HAS_BOTO3:
        audio_path = polly_text_to_speech(text, session_id, voice_settings.get("voice_id"))
        if audio_path:
            return audio_path
        else:
            logging.error("Polly TTS failed, falling back to gTTS.")
    filename = f"static/audio/response_{session_id}_{int(time.time())}.mp3"
    try:
        tts = gTTS(text=text, lang=voice_settings.get("lang", "en"), tld=voice_settings.get("tld", "com"))
        tts.save(filename)
        logging.debug("Audio generated with gTTS")
        return "/" + filename
    except Exception as e:
        logging.error(str(e))
        return "/" + filename


def process_query(user_text, session_id, personality, backstory, voice):
    # If this is the first message, include a greeting instruction.
    greeting_instruction = ""
    if not conversation_history.get(session_id):
        greeting_instruction = "Begin by greeting the user warmly. "

    prompt = (
        f"<s>[INST] Roleplay as a character with the following personality: {personality}. "
        f"Your backstory is: {backstory}. {greeting_instruction}"
        f"Always roleplay completely as this character and follow these instructions strictly. "
        f"Now respond to the following message: {user_text} [/INST]"
    )
    ai_response = query_huggingface(prompt, session_id)
    follow_up = "What else would you like to discuss about this topic?"
    audio_text = ai_response + "\n" + follow_up
    audio_path = text_to_speech(audio_text, session_id, voice)
    if session_id not in conversation_history:
        conversation_history[session_id] = []
    conversation_history[session_id].append({
        "user": user_text,
        "assistant": ai_response + "\nFollow-up: " + follow_up
    })
    return {"answer": ai_response, "follow_up": follow_up, "audio_path": audio_path}############################################
# Cleanup Temporary Audio Files
############################################

def cleanup_temp_files():
    audio_dir = os.path.join("static", "audio")
    if os.path.exists(audio_dir):
        for file in os.listdir(audio_dir):
            if file.startswith("response_") and file.endswith(".mp3"):
                try:
                    os.remove(os.path.join(audio_dir, file))
                except Exception:
                    pass

atexit.register(cleanup_temp_files)

############################################
# HTML Templates
############################################

# Index Template
INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Chameleon AI - Home</title>
  <link href="https://fonts.googleapis.com/css?family=Roboto:300,400,700" rel="stylesheet">
  <style>
    body {
      font-family: 'Roboto', sans-serif;
      margin: 0;
      padding: 0;
      background: linear-gradient(135deg, #f4f7f9, #e2e7ed);
    }
    .header {
      background: #0d6efd;
      color: #fff;
      padding: 30px;
      text-align: center;
    }
    .nav {
      text-align: center;
      margin: 20px 0;
    }
    .nav a {
      color: #fff;
      text-decoration: none;
      font-weight: bold;
      margin: 0 15px;
    }
    .content {
      max-width: 900px;
      margin: 0 auto;
      padding: 30px;
    }
    .footer {
      text-align: center;
      background: #0d6efd;
      color: #fff;
      padding: 10px;
    }
    .btn {
      background: #ff7f50;
      color: #fff;
      padding: 10px 20px;
      text-decoration: none;
      border-radius: 4px;
      transition: background 0.3s ease;
    }
    .btn:hover {
      background: #ff5722;
    }
  </style>
</head>
<body>
  <div class="header">
    <h1>Chameleon AI</h1>
    <p>Your Personalized AI Chatbot Platform</p>
  </div>
  <div class="nav">
    {% if session.get("user_id") %}
      <a href="{{ url_for('dashboard') }}">Dashboard</a>
      <a href="{{ url_for('logout') }}">Logout</a>
    {% else %}
      <a href="{{ url_for('register') }}">Register</a>
      <a href="{{ url_for('login') }}">Login</a>
    {% endif %}
  </div>
  <div class="content">
    <h2>Welcome to Chameleon AI</h2>
    <p>Experience a fully customizable AI chatbot. Sign up, set your character's personality and backstory, choose voice options, and start chatting seamlessly.</p>
    {% if not session.get("user_id") %}
      <a class="btn" href="{{ url_for('register') }}">Get Started</a>
    {% endif %}
  </div>
  <div class="footer">
    &copy; 2025 Chameleon AI. All rights reserved.
  </div>
</body>
</html>
"""

# Register Template
REGISTER_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Chameleon AI - Register</title>
  <link href="https://fonts.googleapis.com/css?family=Roboto:300,400,700" rel="stylesheet">
  <style>
    body {
      font-family: 'Roboto', sans-serif;
      background: #e2e7ed;
      margin: 0;
    }
    .container {
      max-width: 400px;
      margin: 50px auto;
      background: #fff;
      padding: 30px;
      border-radius: 8px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }
    h1 {
      text-align: center;
      color: #0d6efd;
    }
    form {
      display: flex;
      flex-direction: column;
    }
    label {
      margin-top: 15px;
    }
    input[type="text"],
    input[type="email"],
    input[type="password"] {
      padding: 12px;
      margin-top: 5px;
      border: 1px solid #ccc;
      border-radius: 4px;
    }
    input[type="submit"] {
      margin-top: 20px;
      padding: 12px;
      background: #0d6efd;
      color: #fff;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      transition: background 0.3s;
    }
    input[type="submit"]:hover {
      background: #084298;
    }
    .msg {
      color: red;
      text-align: center;
      margin-top: 10px;
    }
    a {
      text-decoration: none;
      color: #0d6efd;
      display: block;
      text-align: center;
      margin-top: 15px;
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>Register</h1>
    {% with messages = get_flashed_messages() %}
      {% if messages %}
        <div class="msg">
          <ul>
            {% for message in messages %}
              <li>{{ message }}</li>
            {% endfor %}
          </ul>
        </div>
      {% endif %}
    {% endwith %}
    <form method="POST">
      <label>Username:</label>
      <input type="text" name="username" required>
      <label>Email:</label>
      <input type="email" name="email" required>
      <label>Password:</label>
      <input type="password" name="password" required>
      <input type="submit" value="Register">
    </form>
    <a href="{{ url_for('index') }}">Back to Home</a>
  </div>
</body>
</html>
"""

# Login Template
LOGIN_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Chameleon AI - Login</title>
  <link href="https://fonts.googleapis.com/css?family=Roboto:300,400,700" rel="stylesheet">
  <style>
    body {
      font-family: 'Roboto', sans-serif;
      background: #e2e7ed;
      margin: 0;
    }
    .container {
      max-width: 400px;
      margin: 50px auto;
      background: #fff;
      padding: 30px;
      border-radius: 8px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }
    h1 {
      text-align: center;
      color: #0d6efd;
    }
    form {
      display: flex;
      flex-direction: column;
    }
    label {
      margin-top: 15px;
    }
    input[type="text"],
    input[type="password"] {
      padding: 12px;
      margin-top: 5px;
      border: 1px solid #ccc;
      border-radius: 4px;
    }
    input[type="submit"] {
      margin-top: 20px;
      padding: 12px;
      background: #0d6efd;
      color: #fff;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      transition: background 0.3s;
    }
    input[type="submit"]:hover {
      background: #084298;
    }
    .msg {
      color: red;
      text-align: center;
      margin-top: 10px;
    }
    a {
      text-decoration: none;
      color: #0d6efd;
      display: block;
      text-align: center;
      margin-top: 15px;
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>Login</h1>
    {% with messages = get_flashed_messages() %}
      {% if messages %}
        <div class="msg">
          <ul>
            {% for message in messages %}
              <li>{{ message }}</li>
            {% endfor %}
          </ul>
        </div>
      {% endif %}
    {% endwith %}
    <form method="POST">
      <label>Username or Email:</label>
      <input type="text" name="login" required>
      <label>Password:</label>
      <input type="password" name="password" required>
      <input type="submit" value="Login">
    </form>
    <a href="{{ url_for('index') }}">Back to Home</a>
  </div>
</body>
</html>
"""

# Dashboard Template
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Chameleon AI - Dashboard</title>
  <link href="https://fonts.googleapis.com/css?family=Roboto:300,400,700" rel="stylesheet">
  <style>
    body {
      font-family: 'Roboto', sans-serif;
      background: #f4f7f9;
      margin: 0;
    }
    .header {
      background: #0d6efd;
      color: #fff;
      padding: 30px;
      text-align: center;
    }
    .container {
      max-width: 900px;
      margin: 30px auto;
      padding: 30px;
      background: #fff;
      border-radius: 8px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }
    .btn {
      display: inline-block;
      background: #ff7f50;
      color: #fff;
      padding: 10px 20px;
      text-decoration: none;
      border-radius: 4px;
      margin: 10px 5px;
      transition: background 0.3s ease;
    }
    .btn:hover {
      background: #ff5722;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 20px;
    }
    th, td {
      padding: 12px;
      border: 1px solid #ddd;
      text-align: left;
    }
    .msg {
      color: green;
      margin: 15px 0;
      text-align: center;
    }
  </style>
</head>
<body>
  <div class="header">
    <h1>Chameleon AI Dashboard</h1>
  </div>
  <div class="container">
    {% with messages = get_flashed_messages() %}
      {% if messages %}
        <div class="msg">
          <ul>
            {% for message in messages %}
              <li>{{ message }}</li>
            {% endfor %}
          </ul>
        </div>
      {% endif %}
    {% endwith %}
    <p>Welcome, {{ user.username }}! Customize your chatbot or start chatting:</p>
    <a class="btn" href="{{ url_for('customize') }}">Customize Chatbot</a>
    <a class="btn" href="{{ url_for('chat') }}">Chat with Your AI</a>
    <hr>
    <h2>Public Chatbots</h2>
    <table>
      <tr>
        <th>AI Name</th>
        <th>Owner ID</th>
        <th>Custom Prompt</th>
      </tr>
      {% for config in public_configs %}
      <tr>
        <td><a href="{{ url_for('public_chat', config_id=config.id) }}">{{ config.ai_name }}</a></td>
        <td>{{ config.owner_id }}</td>
        <td>{{ config.custom_prompt }}</td>
      </tr>
      {% endfor %}
    </table>
    <a class="btn" href="{{ url_for('logout') }}">Logout</a>
  </div>
</body>
</html>
"""

# Customize Template
CUSTOMIZE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Chameleon AI - Customize Chatbot</title>
  <link href="https://fonts.googleapis.com/css?family=Roboto:300,400,700" rel="stylesheet">
  <style>
    body {
      font-family: 'Roboto', sans-serif;
      background: #e2e7ed;
      margin: 0;
    }
    .container {
      max-width: 600px;
      margin: 50px auto;
      background: #fff;
      padding: 30px;
      border-radius: 8px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }
    h1 {
      text-align: center;
      color: #0d6efd;
    }
    form {
      display: flex;
      flex-direction: column;
    }
    label {
      margin-top: 15px;
    }
    input[type="text"],
    textarea, select {
      padding: 12px;
      margin-top: 5px;
      border: 1px solid #ccc;
      border-radius: 4px;
    }
    input[type="submit"] {
      margin-top: 20px;
      padding: 12px;
      background: #0d6efd;
      color: #fff;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      transition: background 0.3s;
    }
    input[type="submit"]:hover {
      background: #084298;
    }
    .msg {
      color: red;
      text-align: center;
      margin-top: 10px;
    }
    a {
      text-decoration: none;
      color: #0d6efd;
      display: block;
      text-align: center;
      margin-top: 15px;
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>Customize Your Chatbot</h1>
    {% with messages = get_flashed_messages() %}
      {% if messages %}
        <div class="msg">
          <ul>
            {% for message in messages %}
              <li>{{ message }}</li>
            {% endfor %}
          </ul>
        </div>
      {% endif %}
    {% endwith %}
    <form method="POST">
      <label>AI Name:</label>
      <input type="text" name="ai_name" placeholder="Enter a name for your AI" required>
      <label>Custom Personality Prompt (min 20 characters):</label>
      <textarea name="custom_prompt" rows="4" required></textarea>
      <label>Character Personality (How do you want your character to act):</label>
      <input type="text" name="character_personality" placeholder="E.g., friendly, sarcastic, heroic" required>
      <label>Character Definition/Backstory (How do you want your character to respond):</label>
      <textarea name="character_backstory" rows="4" required></textarea>
      <label><input type="checkbox" name="disable_filters"> Disable Content Filters</label>
      <label><input type="checkbox" name="voice_mode"> Enable Voice Talking</label>
      <label>Select Voice (if using voice mode):</label>
      <select name="selected_voice">
        <option value="default">Default</option>
        <option value="british">British</option>
        <option value="texas">Texas</option>
        <option value="ivy">Ivy</option>
        <option value="jasmine">Jasmine</option>
        <option value="bianca">Bianca</option>
        <option value="matthew">Matthew</option>
      </select>
      <label><input type="checkbox" name="is_public"> Make chat AI public</label>
      <input type="submit" value="Save Configuration">
    </form>
    <a href="{{ url_for('dashboard') }}">Back to Dashboard</a>
  </div>
</body>
</html>
"""

# Chat Template with animations, loader indicator, and auto-playing audio.
CHAT_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Chameleon AI - Chat</title>
  <link href="https://fonts.googleapis.com/css?family=Roboto:300,400,700" rel="stylesheet">
  <style>
    body {
      font-family: 'Roboto', sans-serif;
      background: #f4f7f9;
      margin: 0;
      padding: 0;
    }
    .chat-container {
      max-width: 700px;
      margin: 30px auto;
      background: #fff;
      border-radius: 8px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.15);
      position: relative;
    }
    .chat-header {
      background: #0d6efd;
      color: #fff;
      padding: 20px;
      border-top-left-radius: 8px;
      border-top-right-radius: 8px;
      text-align: center;
    }
    .chat-messages {
      padding: 20px;
      height: 400px;
      overflow-y: auto;
      border-bottom: 1px solid #ddd;
    }
    .chat-input {
      padding: 20px;
    }
    .chat-input form {
      display: flex;
    }
    .chat-input input[type="text"] {
      flex: 1;
      padding: 12px;
      border: 1px solid #ccc;
      border-radius: 4px;
      transition: box-shadow 0.3s;
    }
    .chat-input input[type="text"]:focus {
      box-shadow: 0 0 5px rgba(13,110,253,0.5);
    }
    .chat-input input[type="submit"] {
      padding: 12px 24px;
      margin-left: 10px;
      background: #0d6efd;
      color: #fff;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      transition: background 0.3s;
    }
    .chat-input input[type="submit"]:hover {
      background: #084298;
    }
    .message {
      margin: 10px 0;
      opacity: 0;
      animation: fadeIn 0.5s forwards;
    }
    .message.user {
      text-align: right;
    }
    .message.ai {
      text-align: left;
    }
    .loader {
      border: 4px solid #f3f3f3;
      border-top: 4px solid #0d6efd;
      border-radius: 50%;
      width: 20px;
      height: 20px;
      animation: spin 1s linear infinite;
      display: inline-block;
    }
    @keyframes fadeIn {
      to { opacity: 1; }
    }
    @keyframes spin {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
    }
    a {
      text-decoration: none;
      color: #0d6efd;
      display: inline-block;
      margin: 15px;
    }
  </style>
</head>
<body>
  <div class="chat-container">
    <div class="chat-header">
      <h2>Chat with Your Custom AI</h2>
    </div>
    <div class="chat-messages" id="messages"></div>
    <div class="chat-input">
      <form id="chat-form">
        <input type="text" id="user_input" placeholder="Type your message here..." required>
        <input type="submit" value="Send">
      </form>
    </div>
  </div>
  <script>
    const form = document.getElementById('chat-form');
    const messagesDiv = document.getElementById('messages');
    form.addEventListener('submit', e => {
      e.preventDefault();
      const input = document.getElementById('user_input');
      const userMessage = input.value;
      const userElem = document.createElement('div');
      userElem.className = 'message user';
      userElem.innerText = "You: " + userMessage;
      messagesDiv.appendChild(userElem);
      input.value = "";
      
      // Add a loader indicating AI is thinking
      const loaderElem = document.createElement('div');
      loaderElem.className = 'message ai';
      loaderElem.innerHTML = '<span class="loader"></span> AI is thinking...';
      messagesDiv.appendChild(loaderElem);
      messagesDiv.scrollTop = messagesDiv.scrollHeight;
      
      fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_input: userMessage, session_id: "{{ session['session_id'] }}" || "" })
      })
      .then(response => response.json())
      .then(data => {
        loaderElem.remove();
        const aiElem = document.createElement('div');
        aiElem.className = 'message ai';
        aiElem.innerHTML = "AI: " + data.answer + "<br><em>" + data.follow_up + "</em>";
        messagesDiv.appendChild(aiElem);
        messagesDiv.scrollTop = messagesDiv.scrollHeight;
        if (data.audio_path) {
          const audioElem = document.createElement('audio');
          audioElem.src = data.audio_path;
          audioElem.autoplay = true;
          document.body.appendChild(audioElem);
        }
      });
    });
  </script>
  <a href="{{ url_for('dashboard') }}">Back to Dashboard</a>
</body>
</html>
"""

############################################
# Routes
############################################

@app.route("/")
def index():
    return render_template_string(INDEX_HTML)

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        email = request.form.get("email")
        password = request.form.get("password")
        if username and email and password:
            if User.query.filter((User.email == email) | (User.username == username)).first():
                flash("Email or username already registered.")
                return redirect(url_for("register"))
            new_user = User(username=username, email=email, password=generate_password_hash(password))
            db.session.add(new_user)
            db.session.commit()
            token = s.dumps(email, salt="email-confirm")
            confirm_url = url_for("confirm", token=token, _external=True)
            flash(f"Registration successful! Please check your email to confirm your account: {confirm_url}")
            return redirect(url_for("index"))
    return render_template_string(REGISTER_HTML)

@app.route("/confirm/<token>")
def confirm(token):
    try:
        email = s.loads(token, salt="email-confirm", max_age=3600)
    except (SignatureExpired, BadSignature):
        flash("Confirmation link is invalid or expired.")
        return redirect(url_for("index"))
    user = User.query.filter_by(email=email).first()
    if user:
        user.confirmed = True
        db.session.commit()
        flash("Account confirmed! Please log in.")
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        login_field = request.form.get("login")
        password = request.form.get("password")
        user = User.query.filter((User.email == login_field) | (User.username == login_field)).first()
        if user and check_password_hash(user.password, password):
            if not user.confirmed:
                flash("Please confirm your email before logging in.")
                return redirect(url_for("login"))
            session["user_id"] = user.id
            session["username"] = user.username
            flash("Logged in successfully!")
            return redirect(url_for("dashboard"))
        flash("Invalid credentials.")
        return redirect(url_for("login"))
    return render_template_string(LOGIN_HTML)

@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully.")
    return redirect(url_for("index"))

@app.route("/dashboard")
@login_required
def dashboard():
    user = User.query.get(session["user_id"])
    public_configs = ChatbotConfig.query.filter_by(is_public=True).all()
    return render_template_string(DASHBOARD_HTML, user=user, public_configs=public_configs)

@app.route("/customize", methods=["GET", "POST"])
@login_required
def customize():
    if request.method == "POST":
        ai_name = request.form.get("ai_name")
        custom_prompt = request.form.get("custom_prompt")
        disable_filters = True if request.form.get("disable_filters") == "on" else False
        voice_mode = True if request.form.get("voice_mode") == "on" else False
        selected_voice = request.form.get("selected_voice")
        is_public = True if request.form.get("is_public") == "on" else False
        character_personality = request.form.get("character_personality")
        character_backstory = request.form.get("character_backstory")
        if not ai_name or len(ai_name.strip()) == 0:
            flash("AI Name is required.")
            return redirect(url_for("customize"))
        if len(custom_prompt.strip()) < 20:
            flash("Custom prompt must be at least 20 characters.")
            return redirect(url_for("customize"))
        if not character_personality or len(character_personality.strip()) == 0:
            flash("Character Personality is required.")
            return redirect(url_for("customize"))
        if not character_backstory or len(character_backstory.strip()) == 0:
            flash("Character Definition/Backstory is required.")
            return redirect(url_for("customize"))
        user_id = session["user_id"]
        # When creating a new public AI, always create a new entry instead of updating the user's existing one.
        config = ChatbotConfig(
            owner_id=user_id,
            ai_name=ai_name,
            custom_prompt=custom_prompt,
            disable_filters=disable_filters,
            voice_mode=voice_mode,
            selected_voice=selected_voice,
            is_public=is_public,
            character_personality=character_personality,
            character_backstory=character_backstory
        )
        db.session.add(config)
        db.session.commit()
        flash("Chatbot configuration updated!")
        return redirect(url_for("chat_with_config", config_id=config.id))
    return render_template_string(CUSTOMIZE_HTML)

@app.route("/chat/<int:config_id>", methods=["GET", "POST"])
@login_required
def chat_with_config(config_id):
    config = ChatbotConfig.query.get(config_id)
    if not config:
        flash("Chatbot configuration not found.")
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        data = request.get_json()
        user_input = data.get("user_input")
        session_id = session.get("session_id", str(uuid.uuid4()))
        session["session_id"] = session_id
        custom_personality = config.character_personality + " | " + config.custom_prompt
        character_backstory = config.character_backstory
        voice = config.selected_voice if config.voice_mode else DEFAULT_VOICE
        result = process_query(user_input, session_id, custom_personality, character_backstory, voice)
        return jsonify(result)
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return render_template_string(CHAT_HTML)

@app.route("/chat", methods=["GET", "POST"])
@login_required
def chat():
    if request.method == "POST":
        data = request.get_json()
        user_input = data.get("user_input")
        user_id = session["user_id"]
        config = ChatbotConfig.query.filter_by(owner_id=user_id).first()
        if config:
            custom_personality = config.character_personality + " | " + config.custom_prompt
            character_backstory = config.character_backstory
            voice = config.selected_voice if config.voice_mode else DEFAULT_VOICE
        else:
            custom_personality = PERSONALITIES[DEFAULT_PERSONALITY]
            character_backstory = ""
            voice = DEFAULT_VOICE
        session_id = session.get("session_id", str(uuid.uuid4()))
        session["session_id"] = session_id
        result = process_query(user_input, session_id, custom_personality, character_backstory, voice)
        return jsonify(result)
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return render_template_string(CHAT_HTML)

# New route for public chatbots - accessible by anyone.
@app.route("/public_chat/<int:config_id>", methods=["GET", "POST"])
def public_chat(config_id):
    config = ChatbotConfig.query.get(config_id)
    if not config or not config.is_public:
        flash("Public Chatbot not found.")
        return redirect(url_for("index"))
    if request.method == "POST":
        data = request.get_json()
        user_input = data.get("user_input")
        session_id = session.get("session_id", str(uuid.uuid4()))
        session["session_id"] = session_id
        custom_personality = config.character_personality + " | " + config.custom_prompt
        character_backstory = config.character_backstory
        voice = config.selected_voice if config.voice_mode else DEFAULT_VOICE
        result = process_query(user_input, session_id, custom_personality, character_backstory, voice)
        return jsonify(result)
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return render_template_string(CHAT_HTML)

@app.route("/audio/<path:filename>")
def serve_audio(filename):
    return send_from_directory("static/audio", filename)

if __name__ == "__main__":
    app.run(debug=True)
