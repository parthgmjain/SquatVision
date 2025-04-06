# app.py
from flask import Flask, request, render_template, redirect, url_for
import os
from werkzeug.utils import secure_filename
from PoseModule import process_video  # ✅ Make sure this is here

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
PROCESSED_FOLDER = 'static/processed'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    video = request.files['video']
    if video.filename == '':
        return "No selected file"
    
    filename = secure_filename(video.filename)
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    processed_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)

    # Save uploaded video
    video.save(upload_path)

    # ✅ Actually run pose detection
    process_video(upload_path, processed_path)

    video_url = url_for('static', filename=f'processed/{filename}')
    return render_template('processed.html', video_url=video_url)

if __name__ == '__main__':
    app.run(debug=True)
