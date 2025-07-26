import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import os
from flask import Flask, render_template, Response, request, redirect,url_for,flash
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'MP4', 'MPG', 'AVI', 'MOV', 'MKV'}

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# MediaPipe Initialization
pixels_per_cm_factor = 4.0
mp_pose = mp.solutions.pose

# Helper Function to get Mid Hip Coordinates
def get_mid_hip_y_inverted(landmarks, image_height):
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    mid_hip_y_pixel = (left_hip.y + right_hip.y) / 2 * image_height
    return image_height - mid_hip_y_pixel

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def process_video_for_jump_height(video_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Could not open video file."}
    
    # Configuration Parameters ---------------
    SMOOTHING_BUFFER_SIZE = 5
    IDLE_THRESHOLD_PIXELS = 10
    JUMP_START_THRESHOLD_PIXELS = 25
    JUMP_LANDING_THRESHOLD_PIXELS = 10
    MIN_JUMP_DISPLACEMENT_PIXELS = 50

    # Variables for jump detection and height calculation
    baseline_y = -1.0
    min_y_during_jump = float('inf')
    is_jumping = False

    # Store all detected jump heights
    all_jump_heights_cm = []

    hip_y_buffer = deque(maxlen=SMOOTHING_BUFFER_SIZE)


    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_height, frame_width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        current_smoothed_hip_y = None

        if results.pose_landmarks:
            current_hip_y_inverted = get_mid_hip_y_inverted(results.pose_landmarks.landmark, frame_height)
            hip_y_buffer.append(current_hip_y_inverted)

            if len(hip_y_buffer) == SMOOTHING_BUFFER_SIZE:
                current_smoothed_hip_y = np.mean(hip_y_buffer)
            else:
                current_smoothed_hip_y = current_hip_y_inverted


           # Baseline Manangement
            if baseline_y == 1.0:
                baseline_y = current_smoothed_hip_y
            
            if not is_jumping:
                if abs(current_smoothed_hip_y - baseline_y) < IDLE_THRESHOLD_PIXELS:
                    baseline_y = (baseline_y * 0.95) + (current_smoothed_hip_y * 0.05)

                if current_smoothed_hip_y > baseline_y + JUMP_START_THRESHOLD_PIXELS:
                    is_jumping = True
                    min_y_during_jump = current_smoothed_hip_y
            else:
                if current_smoothed_hip_y > min_y_during_jump:
                    min_y_during_jump = current_smoothed_hip_y

                if current_smoothed_hip_y < (baseline_y - JUMP_LANDING_THRESHOLD_PIXELS) and (min_y_during_jump - baseline_y > MIN_JUMP_DISPLACEMENT_PIXELS):

                    jump_height_pixels = min_y_during_jump - baseline_y

                    if pixels_per_cm_factor is not None and pixels_per_cm_factor > 0:
                        jump_height_cm = jump_height_pixels / pixels_per_cm_factor
                        all_jump_heights_cm.append(round(jump_height_cm,2))

                    else:
                        all_jump_heights_cm.append(f"{round(jump_height_pixels, 2)} pixels (Calibrate!)")

                    is_jumping = False
                    min_y_during_jump = float('inf')

    cap.release()
    pose.close()

    if not all_jump_heights_cm:
        return {"message": "No jumps detected in the video or jump was too small."}
    else:
        return {"jumps": all_jump_heights_cm}
    
@app.route('/', methods= ['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'video_file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['video_file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            flash('File successfully uploaded, processing...')

            # Process the video
            results = process_video_for_jump_height(file_path)

            # os.remove(file_path)  

            if "error" in results:
                flash(f"Error processing video: {results['error']}")
                return render_template('index.html', results = None)
            else:
                return render_template('index.html', results=results)
        else:
            flash('Allowed video types are MP4, MPG, AVI, MOV, MKV')
            return redirect(request.url)
    return render_template('index.html', results=None)


if __name__ == '__main__':
    app.run(debug=True)
