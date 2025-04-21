from flask import Flask, request, jsonify, send_file
import cv2
import os
import numpy as np
import face_recognition
from retinaface import RetinaFace
from flask_cors import CORS
from datetime import datetime

# Initialize Flask app with CORS support and configure storage directories
app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
FACES_FOLDER = 'faces'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(FACES_FOLDER, exist_ok=True)

# Configure RetinaFace detector - no need to instantiate as it's a module
# Store face encodings for detected faces
face_encodings_map = {}

# Handle video file upload
@app.route('/upload', methods=['POST'])
def upload_video():
    file = request.files['video']
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    return jsonify({'message': 'Video uploaded successfully', 'video_path': file_path})

# Global variables to track progress
# Track progress of face extraction and video processing
extraction_progress = 0
processing_progress = 0

@app.route('/processing_progress', methods=['GET'])
def get_processing_progress():
    global processing_progress
    return jsonify({'progress': processing_progress})

@app.route('/extraction_progress', methods=['GET'])
def get_extraction_progress():
    global extraction_progress
    return jsonify({'progress': extraction_progress})

# Extract unique faces from video using RetinaFace and face_recognition
@app.route('/extract_faces', methods=['POST'])
def extract_faces():
    global extraction_progress
    extraction_progress = 0
    data = request.json
    video_path = data['video_path']
    cap = cv2.VideoCapture(video_path)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_duration = total_frames / fps if fps > 0 else 0
    
    if video_duration < 10:
        frame_skip = max(3, int(fps / 5))
    elif video_duration < 30:
        frame_skip = max(5, int(fps / 3))
    else:
        frame_skip = max(8, int(fps / 2))
    
    max_frames_to_process = min(75, total_frames // frame_skip)
    
    face_id = 0
    unique_face_encodings = []
    unique_face_crops = []
    similarity_threshold = 0.6
    
    frame_count = 0
    processed_frames = 0
    
    while True:
        ret, frame = cap.read()
        if not ret or processed_frames >= max_frames_to_process:
            break
            
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue
        
        processed_frames += 1
        extraction_progress = int((processed_frames / max_frames_to_process) * 100)
        
        original_frame = frame.copy()
        
        height, width = frame.shape[:2]
        if width > 1280:
            scale_factor = 1280 / width
            new_width = 1280
            new_height = int(height * scale_factor)
            frame = cv2.resize(frame, (new_width, new_height))
        
        faces = RetinaFace.detect_faces(frame)
        
        if not faces:
            continue
            
        batch_face_crops = []
        batch_face_coords = []
        
        for face_key in faces:
            face = faces[face_key]
            if face['score'] < 0.5:
                continue
                
            x1, y1, x2, y2 = face['facial_area']
            x, y = x1, y1
            w, h = x2 - x1, y2 - y1
            x, y = max(0, x), max(0, y)
            w, h = max(1, w), max(1, h)
            
            margin_x = int(w * 0.10)
            margin_y = int(h * 0.10)
            x_with_margin = max(0, x - margin_x)
            y_with_margin = max(0, y - margin_y)
            w_with_margin = min(frame.shape[1] - x_with_margin, w + 2 * margin_x)
            h_with_margin = min(frame.shape[0] - y_with_margin, h + 2 * margin_y)
            
            if width > 1280:
                orig_scale = width / new_width
                orig_x = int(x_with_margin * orig_scale)
                orig_y = int(y_with_margin * orig_scale)
                orig_w = int(w_with_margin * orig_scale)
                orig_h = int(h_with_margin * orig_scale)
                
                orig_x = max(0, orig_x)
                orig_y = max(0, orig_y)
                orig_w = min(original_frame.shape[1] - orig_x, orig_w)
                orig_h = min(original_frame.shape[0] - orig_y, orig_h)
                
                face_crop = original_frame[orig_y:orig_y+orig_h, orig_x:orig_x+orig_w]
            else:
                face_crop = frame[y_with_margin:y_with_margin+h_with_margin, x_with_margin:x_with_margin+w_with_margin]
                
            if face_crop.size > 0:
                batch_face_crops.append(face_crop)
                batch_face_coords.append((x_with_margin, y_with_margin, w_with_margin, h_with_margin))
        
        batch_rgb_faces = []
        for face_crop in batch_face_crops:
            rgb_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            batch_rgb_faces.append(rgb_face)
            
        for i, rgb_face in enumerate(batch_rgb_faces):
            face_locations = face_recognition.face_locations(rgb_face, model="cnn")
            best_face_encoding = None
            
            if face_locations:
                encodings = face_recognition.face_encodings(rgb_face, face_locations)
                if encodings:
                    best_face_encoding = encodings[0]
            
            if best_face_encoding is None:
                h, w = rgb_face.shape[:2]
                center = (w // 2, h // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, 15, 1.0)
                rotated = cv2.warpAffine(rgb_face, rotation_matrix, (w, h))
                
                rot_face_locations = face_recognition.face_locations(rotated, model="cnn")
                if rot_face_locations:
                    rot_encodings = face_recognition.face_encodings(rotated, rot_face_locations)
                    if rot_encodings:
                        best_face_encoding = rot_encodings[0]
            
            if best_face_encoding is not None:
                is_duplicate = False
                if unique_face_encodings:
                    face_distances = face_recognition.face_distance(unique_face_encodings, best_face_encoding)
                    is_duplicate = np.min(face_distances) < similarity_threshold
                
                if not is_duplicate:
                    face_id += 1
                    face_encodings_map[face_id] = best_face_encoding
                    unique_face_encodings.append(best_face_encoding)
                    unique_face_crops.append(batch_face_crops[i])
                    face_path = os.path.join(FACES_FOLDER, f'face_{face_id}.jpg')
                    cv2.imwrite(face_path, batch_face_crops[i])
                    extraction_progress = int((processed_frames / max_frames_to_process) * 100)
    cap.release()
    return jsonify({'message': 'Faces extracted', 'faces': list(face_encodings_map.keys()), 'total_faces': face_id})

# Process video by blurring selected faces using face recognition
@app.route('/process_video', methods=['POST'])
def process_video():
    global processing_progress
    processing_progress = 0
    
    data = request.json
    video_path = data['video_path']
    selected_face_ids = data['selected_faces']
    selected_encodings = [face_encodings_map[i] for i in selected_face_ids if i in face_encodings_map]
    
    cap = cv2.VideoCapture(video_path)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps if fps > 0 else 0
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_filename = f'processed_video_{timestamp}.mp4'
    output_path = os.path.join(PROCESSED_FOLDER, output_filename)
    abs_output_path = os.path.abspath(output_path)
    
    temp_output_path = os.path.join(PROCESSED_FOLDER, f'temp_{timestamp}.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))
    
    process_this_frame = False
    frame_count = 0

    face_locations = []
    face_encodings = []
    matches = []
    last_face_locations = []
    last_face_encodings = []
    last_matches = []
    resize_for_processing = False
    processing_width = width
    progress_interval = max(1, total_frames // 100)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if video_duration < 10:
            process_this_frame = (frame_count % 2 == 0)
        elif video_duration < 30:
            process_this_frame = (frame_count % 3 == 0)
        else:
            process_this_frame = (frame_count % 4 == 0)
        
        output_frame = frame.copy()
        
        if width > 1280:
            resize_for_processing = True
            scale_factor = 1280 / width
            processing_width = 1280
            processing_height = int(height * scale_factor)
            processing_frame = cv2.resize(frame, (processing_width, processing_height))
        else:
            resize_for_processing = False
            processing_frame = frame
            
        if process_this_frame:
            faces = RetinaFace.detect_faces(processing_frame)
            face_locations = []
            rgb_frame = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2RGB)
            
            if faces:
                for face_key in faces:
                    face = faces[face_key]
                    if face['score'] < 0.5:
                        continue
                    x1, y1, x2, y2 = face['facial_area']
                    face_locations.append((y1, x2, y2, x1))
                
                if face_locations:
                    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                else:
                    face_encodings = []
            else:
                face_locations = face_recognition.face_locations(rgb_frame, model="cnn")
                if not face_locations:
                    face_locations = face_recognition.face_locations(rgb_frame, model="hog")
                if face_locations:
                    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                else:
                    face_encodings = []
            
            matches = []
            for encoding in face_encodings:
                face_matches = face_recognition.compare_faces(selected_encodings, encoding, tolerance=0.5)
                matches.append(True in face_matches)
                
            last_face_locations = face_locations
            last_face_encodings = face_encodings
            last_matches = matches
        else:
            face_locations = last_face_locations
            face_encodings = last_face_encodings
            matches = last_matches
        
        if face_locations:
            for i, ((top, right, bottom, left), should_blur) in enumerate(zip(face_locations, matches)):
                if should_blur:
                    if resize_for_processing:
                        scale_factor = width / processing_width
                        top = int(top * scale_factor)
                        right = int(right * scale_factor)
                        bottom = int(bottom * scale_factor)
                        left = int(left * scale_factor)
                        
                        top = max(0, top)
                        right = min(width, right)
                        bottom = min(height, bottom)
                        left = max(0, left)
                    
                    face_region = output_frame[top:bottom, left:right]
                    if face_region.size > 0:
                        kernel_size = min(99, max(25, (bottom - top) // 2))
                        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
                        output_frame[top:bottom, left:right] = cv2.GaussianBlur(
                            face_region, (kernel_size, kernel_size), 30)
        
        out.write(output_frame)
        frame_count += 1
        
        if frame_count % progress_interval == 0 or frame_count == total_frames:
            progress = (frame_count / total_frames) * 100
            processing_progress = progress
    
    cap.release()
    out.release()
    
    try:
        import subprocess
        ffmpeg_cmd = f'ffmpeg -i "{temp_output_path}" -i "{video_path}" -c:v copy -c:a aac -map 0:v:0 -map 1:a:0? -shortest -y "{output_path}"'
        subprocess.run(ffmpeg_cmd, check=True, shell=True, creationflags=subprocess.CREATE_NO_WINDOW)
        
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)
    except Exception as e:
        if os.path.exists(temp_output_path) and not os.path.exists(output_path):
            os.rename(temp_output_path, output_path)
        elif os.path.exists(temp_output_path):
            os.remove(output_path) if os.path.exists(output_path) else None
            os.rename(temp_output_path, output_path)
    
    return jsonify({
        'message': 'Video processed', 
        'output_path': output_path, 
        'abs_output_path': abs_output_path,
        'progress': 100
    })

# Download processed video file
@app.route('/download/<filename>', methods=['GET'])
def download_video(filename):
    return send_file(os.path.join(PROCESSED_FOLDER, filename), as_attachment=True)

# Serve extracted face images
@app.route('/faces/<filename>', methods=['GET'])
def serve_face(filename):
    # Use absolute path to the faces folder in the project root
    faces_abs_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), FACES_FOLDER)
    return send_file(os.path.join(faces_abs_path, filename))

if __name__ == '__main__':
    app.run(debug=True)
