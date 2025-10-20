# app.py
from flask import Flask, render_template, request, jsonify, make_response
import requests
import uuid
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['API_URL'] = "http://localhost:8000"  # Your FastAPI backend URL

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

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
            filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            uploaded_file = filepath
    
    try:
        # Create headers with cookies if they exist
        cookies = {}
        if session_cookie:
            cookies['session_id'] = session_cookie
        
        if uploaded_file:
            # API request with image
            files = {"image": (os.path.basename(uploaded_file), open(uploaded_file, "rb"), "image/jpeg")}
            data = {"text": prompt}
            response = requests.post(
                f"{app.config['API_URL']}/upload", 
                files=files, 
                data=data,
                cookies=cookies  # Use cookies parameter instead of headers
            )
            
            # Close file and try to remove it after sending
            files["image"][1].close()
        else:
            # API request for text only
            payload = {
                "query": prompt,
                "conversation_history": []  # Empty array since backend maintains history
            }
            response = requests.post(
                f"{app.config['API_URL']}/chat", 
                json=payload,
                cookies=cookies  # Use cookies parameter instead of headers
            )
        
        if response.status_code == 200:
            result = response.json()
            
            # Create response object to modify
            flask_response = jsonify({
                "status": "success",
                "agent": result["agent"],
                "response": result["response"]
            })
            
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
            })
    except Exception as e:
        print(f"Exception: {str(e)}")
        return jsonify({
            "status": "error",
            "agent": "System",
            "response": f"Error: {str(e)}"
        })

if __name__ == '__main__':
    app.run(debug=True, port=5000)