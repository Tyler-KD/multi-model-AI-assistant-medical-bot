import os
import uuid
import requests
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, jsonify, make_response, abort, send_file, after_this_request
import threading
import time
import glob
import tempfile
from pydub import AudioSegment

from io import BytesIO
from elevenlabs.client import ElevenLabs

from config import Config

config = Config()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/frontend'
app.config['SPEECH_DIR'] = 'uploads/speech'
# Convert MB to bytes for MAX_CONTENT_LENGTH
app.config['MAX_CONTENT_LENGTH'] = config.api.max_image_upload_size * 1024 * 1024  # Convert MB to bytes
app.config['ELEVEN_LABS_API_KEY'] = config.speech.eleven_labs_api_key
app.config['API_URL'] = "http://localhost:8000"  # Your FastAPI backend URL

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ElebenLabs Client
client = ElevenLabs(
    api_key = app.config['ELEVEN_LABS_API_KEY'],
)

# Define allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def cleanup_old_audio():
    """Deletes all .mp3 files in the uploads/speech folder every 5 minutes."""
    while True:
        try:
            files = glob.glob(f"{app.config['SPEECH_DIR']}/*.mp3")
            for file in files:
                os.remove(file)
            print("Cleaned up old speech files.")
        except Exception as e:
            print(f"Error during cleanup: {e}")
        time.sleep(300)  # Runs every 5 minutes

# Start background cleanup thread
cleanup_thread = threading.Thread(target=cleanup_old_audio, daemon=True)
cleanup_thread.start()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    data = request.form
    prompt = data.get('message')
    
    # Get session cookie if it exists in the request
    session_cookie = request.cookies.get('session_id')
    
    # Process any uploaded file
    uploaded_file = None
    if 'file' in request.files:
        file = request.files['file']
        
        if file and file.filename != '':
            # Validate file type
            if not allowed_file(file.filename):
                return jsonify({
                    "status": "error",
                    "agent": "System",
                    "response": "Unsupported file type. Allowed formats: PNG, JPG, JPEG"
                }), 400
            
            # Validate file size before saving
            file.seek(0, os.SEEK_END)
            file_size = file.tell()
            file.seek(0)  # Reset file pointer
            
            if file_size > app.config['MAX_CONTENT_LENGTH']:
                return jsonify({
                    "status": "error", 
                    "agent": "System",
                    "response": f"File too large. Maximum size allowed: {config.api.max_image_upload_size}MB"
                }), 413
            
            # Save file securely
            filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            uploaded_file = filepath
    
    try:
        # Create cookies dict if session exists
        cookies = {}
        if session_cookie:
            cookies['session_id'] = session_cookie
        
        if uploaded_file:
            # API request with image
            with open(uploaded_file, "rb") as image_file:
                files = {"image": (os.path.basename(uploaded_file), image_file, "image/jpeg")}
                data = {"text": prompt}
                response = requests.post(
                    f"{app.config['API_URL']}/upload", 
                    files=files, 
                    data=data,
                    cookies=cookies
                )
            
            # Remove temporary file after sending
            try:
                os.remove(uploaded_file)
            except Exception as e:
                print(f"Failed to remove temporary file: {str(e)}")
        else:
            # API request for text only
            payload = {
                "query": prompt,
                "conversation_history": []  # Empty array since backend maintains history
            }
            response = requests.post(
                f"{app.config['API_URL']}/chat", 
                json=payload,
                cookies=cookies
            )
        
        if response.status_code == 200:
            result = response.json()
            
            # Create response object to modify
            response_data = {
                "status": "success",
                "agent": result["agent"],
                "response": result["response"]
            }
            
            # Add result image URL if it exists
            if "result_image" in result:
                # Prefix with the FastAPI URL
                response_data["result_image"] = f"{app.config['API_URL']}{result['result_image']}"
            
            flask_response = jsonify(response_data)
            
            # Extract session cookie from response if it exists
            if 'session_id' in response.cookies:
                # Set the cookie in our Flask response
                flask_response.set_cookie('session_id', response.cookies['session_id'])
            
            return flask_response
        else:
            return jsonify({
                "status": "error",
                "agent": "System",
                "response": f"Error: {response.status_code} - {response.text}"
            }), response.status_code
    except Exception as e:
        print(f"Exception: {str(e)}")
        return jsonify({
            "status": "error",
            "agent": "System",
            "response": f"Error: {str(e)}"
        }), 500

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    """Endpoint to transcribe speech using ElevenLabs API"""
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    audio_file = request.files['audio']
    
    if audio_file.filename == '':
        return jsonify({"error": "No audio file selected"}), 400
    
    try:
        # Save the audio file temporarily
        temp_dir = app.config['SPEECH_DIR']
        os.makedirs(temp_dir, exist_ok=True)
        
        # Save with a generic extension
        # temp_audio = os.path.join(temp_dir, f"speech_{uuid.uuid4()}.webm")
        temp_audio = f"./{temp_dir}/speech_{uuid.uuid4()}.webm"
        audio_file.save(temp_audio)
        
        # Debug: Print file size to check if it's empty
        file_size = os.path.getsize(temp_audio)
        print(f"Received audio file size: {file_size} bytes")
        
        if file_size == 0:
            return jsonify({"error": "Received empty audio file"}), 400
        
        # Convert to MP3 using ffmpeg directly
        # mp3_path = os.path.join(temp_dir, f"speech_{uuid.uuid4()}.mp3")
        mp3_path = f"./{temp_dir}/speech_{uuid.uuid4()}.mp3"
        
        try:
            # Use pydub with format detection
            audio = AudioSegment.from_file(temp_audio)
            audio.export(mp3_path, format="mp3")
            
            # Debug: Print MP3 file size
            mp3_size = os.path.getsize(mp3_path)
            print(f"Converted MP3 file size: {mp3_size} bytes")

            with open(mp3_path, "rb") as mp3_file:
                audio_data = mp3_file.read()
            print(f"Converted audio file into byte array successfully!")

            transcription = client.speech_to_text.convert(
                file=audio_data,
                model_id="scribe_v1", # Model to use, for now only "scribe_v1" is supported
                tag_audio_events=True, # Tag audio events like laughter, applause, etc.
                language_code="eng", # Language of the audio file. If set to None, the model will detect the language automatically.
                diarize=True, # Whether to annotate who is speaking
            )
            
            # Clean up temp files
            try:
                os.remove(temp_audio)
                os.remove(mp3_path)
                print(f"Deleted temp files: {temp_audio}, {mp3_path}")
            except Exception as e:
                # pass
                print(f"Could not delete file: {e}")
            
            if transcription.text:
                return jsonify({"transcript": transcription.text})
            else:
                return jsonify({"error": f"API error: {transcription}", "details": transcription.text}), 500

        except Exception as e:
            print(f"Error processing audio: {str(e)}")
            return jsonify({"error": f"Error processing audio: {str(e)}"}), 500
                
    except Exception as e:
        print(f"Transcription error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/generate-speech', methods=['POST'])
def generate_speech():
    """Endpoint to generate speech securely"""
    try:
        data = request.json
        text = data.get("text", "")
        selected_voice_id = data.get("voice_id", "EXAMPLE_VOICE_ID")  # Replace with a valid voice ID

        if not text:
            return jsonify({"error": "Text is required"}), 400
        
        # Define API request to ElevenLabs
        elevenlabs_url = f"https://api.elevenlabs.io/v1/text-to-speech/{selected_voice_id}/stream"
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": app.config['ELEVEN_LABS_API_KEY']
        }
        payload = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }

        # Send request to ElevenLabs API
        response = requests.post(elevenlabs_url, headers=headers, json=payload)

        if response.status_code != 200:
            return jsonify({"error": f"Failed to generate speech, status: {response.status_code}", "details": response.text}), 500
        
        # Save the audio file temporarily
        temp_dir = app.config['SPEECH_DIR']
        os.makedirs(temp_dir, exist_ok=True)

        # Save the audio file temporarily
        temp_audio_path = f"./{temp_dir}/{uuid.uuid4()}.mp3"
        with open(temp_audio_path, "wb") as f:
            f.write(response.content)

        # Send back the generated audio file
        return send_file(temp_audio_path, mimetype="audio/mpeg")

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Add a custom error handler for request entity too large
@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        "status": "error",
        "agent": "System",
        "response": f"File too large. Maximum size allowed: {config.api.max_image_upload_size}MB"
    }), 413

if __name__ == '__main__':
    app.run(debug=True, port=5000)