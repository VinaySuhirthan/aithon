from flask import Flask, send_from_directory
from flask_cors import CORS
import os
import cv2
import time
import threading
from datetime import datetime
from ultralytics import YOLO
from supabase import create_client, Client
from dotenv import load_dotenv

# ─── Load env ────────────────────────────────────────────────────────────────
load_dotenv()

# ─── Flask app ───────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/list')
@app.route('/list.html')
def list_page():
    return send_from_directory('.', 'list.html')

# ─── Supabase setup ───────────────────────────────────────────────────────────
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# ─── YOLO setup ───────────────────────────────────────────────────────────────
model = YOLO('yolov8m.pt')  # Upgraded to Medium for better accuracy

UPDATE_INTERVAL = 5  # seconds

# Video sources configuration
VIDEO_SOURCES = [
    {"path": "Wax Museum.mp4", "spot": "Wax Museum", "capacity": 200},
    {"path": "Rose Garden.mp4", "spot": "Rose Garden", "capacity": 50},
    {"path": "Deer Park.mp4", "spot": "Deer Park", "capacity": 100},
    {"path": "Honeymoon Boat House.mp4", "spot": "Honeymoon Boat House", "capacity": 150},
    {"path": "Tribal research centre meseum.mp4", "spot": "Tribal Museum", "capacity": 200}
]

def count_people_in_frame(frame):
    # Use better confidence and standard image size for accuracy
    results = model(frame, conf=0.25, imgsz=640, verbose=False)
    person_count = 0
    for result in results:
        for box in result.boxes:
            if int(box.cls) == 0:  # class 0 = person in COCO
                person_count += 1
    return person_count

def update_supabase(avg_count, spot_name, capacity):
    try:
        # Calculate percentage based on capacity
        crowd_percentage = min(int((avg_count / capacity) * 100), 100)
        
        supabase.table('crowd_data').update({
            'crowd_level': crowd_percentage,
            'last_updated': datetime.now().isoformat()
        }).eq('spot_name', spot_name).execute()
        print(f"✓ Updated {spot_name}: {avg_count} people ({crowd_percentage}%)")
        return True
    except Exception as e:
        print(f"✗ Error updating Supabase for {spot_name}: {e}")
        return False

def process_video(video_path, spot_name, capacity):
    print(f"🎥 Starting processing for: {spot_name} ({video_path})")
    while True:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video file '{video_path}' for {spot_name}")
            time.sleep(10)
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_count = 0
        person_counts = []
        last_update_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                if person_counts:
                    avg_count = int(sum(person_counts) / len(person_counts))
                    update_supabase(avg_count, spot_name, capacity)
                break

            # Process every 2nd frame (increased from 5th) for better tracking
            if frame_count % 2 == 0:
                person_count = count_people_in_frame(frame)
                person_counts.append(person_count)

            frame_count += 1

            current_time = time.time()
            if current_time - last_update_time >= UPDATE_INTERVAL:
                if person_counts:
                    avg_count = int(sum(person_counts) / len(person_counts))
                    update_supabase(avg_count, spot_name, capacity)
                person_counts = []
                last_update_time = current_time

            # Small sleep to yield to other threads
            time.sleep(0.01)

        cap.release()
        print(f"🔄 Restarting video loop for {spot_name}")
        time.sleep(1)

# ─── Entry point ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # Start crowd detection loops in background threads
    for source in VIDEO_SOURCES:
        t = threading.Thread(
            target=process_video, 
            args=(source["path"], source["spot"], source["capacity"]), 
            daemon=True
        )
        t.start()
        print(f"🤖 Crowd detection started for {source['spot']}")

    # Start Flask web server (main thread)
    port = int(os.environ.get('PORT', 5000))
    print(f"🌐 Flask running at http://127.0.0.1:{port}\n")
    app.run(host='0.0.0.0', port=port, debug=False)
