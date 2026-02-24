from flask import Flask, send_from_directory, request, jsonify
from flask_cors import CORS
import os
import traceback
import cv2
import time
import threading
import gc
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
from ultralytics import YOLO
from supabase import create_client, Client
from dotenv import load_dotenv

# ─── Load env ────────────────────────────────────────────────────────────────
load_dotenv(override=True)

# ─── Flask app ───────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ML Cache to avoid re-training on every request
ml_cache = {}

def get_ml_data(city_name):
    city_key = city_name.lower()
    if city_key in ml_cache:
        return ml_cache[city_key]

    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Ml')
    city_path = os.path.join(base_dir, city_key)
    
    if not os.path.exists(city_path):
        return None

    try:
        # Prevent pandas downcasting warning
        pd.set_option('future.no_silent_downcasting', True)
        
        all_files = os.listdir(city_path)
        train_files = [f for f in all_files if f.lower().endswith('.csv') and '20' in f and 'percent' not in f.lower()]
        # Handle the inconsistent naming in their folders
        pct_files = [f for f in all_files if 'percent' in f.lower() and f.lower().endswith('.csv')]
        
        if not train_files or not pct_files:
            return None

        # Load and combine training data
        data_list = []
        for file in train_files:
            df = pd.read_csv(os.path.join(city_path, file))
            data_list.append(df)
        all_data = pd.concat(data_list, ignore_index=True)
        all_data = all_data.dropna(subset=['Year', 'Month', 'Avg_Weekday_Crowd', 'Avg_Weekend_Crowd'])

        month_map = {
            'January': 1, 'February': 2, 'March': 3, 'April': 4,
            'May': 5, 'June': 6, 'July': 7, 'August': 8,
            'September': 9, 'October': 10, 'November': 11, 'December': 12
        }
        all_data['Month_Num'] = all_data['Month'].replace(month_map).infer_objects(copy=False)
        all_data['Month_Num'] = pd.to_numeric(all_data['Month_Num'], errors='coerce').fillna(0).astype(int)

        X = all_data[['Year', 'Month_Num']]
        y_wkday = all_data['Avg_Weekday_Crowd']
        y_wknd = all_data['Avg_Weekend_Crowd']

        # Use few estimators to save memory
        model_wkday = RandomForestRegressor(n_estimators=30, random_state=42)
        model_wknd = RandomForestRegressor(n_estimators=30, random_state=42)
        model_wkday.fit(X, y_wkday)
        model_wknd.fit(X, y_wknd)
        
        crowd_pct = pd.read_csv(os.path.join(city_path, pct_files[0]))
        
        ml_cache[city_key] = (model_wkday, model_wknd, crowd_pct)
        return ml_cache[city_key]
    except Exception as e:
        print(f"ML Error for {city_name}: {e}")
        return None

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    city = data.get('city', '')
    date_str = data.get('date', '') # YYYY-MM-DD
    place = data.get('place', '')

    if not city or not date_str or not place:
        return jsonify({"error": "Missing parameters"}), 400

    ml_data = get_ml_data(city)
    if not ml_data:
        return jsonify({"error": f"No data found for city: {city}"}), 404

    model_wkday, model_wknd, crowd_pct = ml_data

    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        year = date_obj.year
        month_num = date_obj.month
        is_weekend = date_obj.weekday() >= 5

        # Fuzzy matching for place name
        place_clean = place.lower().replace("'", "").replace("’", "").strip()
        # Copy to avoid warnings on the cached dataframe
        temp_pct = crowd_pct.copy()
        temp_pct['Place_Clean'] = temp_pct['Place'].str.lower().str.replace("'", "").str.replace("’", "").str.strip()
        
        matched_row = temp_pct[temp_pct['Place_Clean'] == place_clean]
        if matched_row.empty:
            return jsonify({"error": f"Place '{place}' not found in city data"}), 404

        input_df = pd.DataFrame([[year, month_num]], columns=['Year', 'Month_Num'])
        if is_weekend:
            pred_base = model_wknd.predict(input_df)[0]
        else:
            pred_base = model_wkday.predict(input_df)[0]

        pct = matched_row.iloc[0]['Crowd_Percentage']
        visitors = int(round(pred_base * pct / 100))

        return jsonify({
            "city": city,
            "place": place,
            "date": date_str,
            "predicted_visitors": visitors
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    data = request.json
    city = data.get('city', '')
    date_str = data.get('date', '')
    places = data.get('places', [])

    if not city or not date_str or not places:
        return jsonify({"error": "Missing parameters"}), 400

    ml_data = get_ml_data(city)
    if not ml_data:
        return jsonify({"error": f"No data found for city: {city}"}), 404

    model_wkday, model_wknd, crowd_pct = ml_data

    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        year = date_obj.year
        month_num = date_obj.month
        is_weekend = date_obj.weekday() >= 5

        input_df = pd.DataFrame([[year, month_num]], columns=['Year', 'Month_Num'])
        pred_base = model_wknd.predict(input_df)[0] if is_weekend else model_wkday.predict(input_df)[0]

        temp_pct = crowd_pct.copy()
        temp_pct['Place_Clean'] = temp_pct['Place'].str.lower().str.replace("'", "").str.replace("\u2019", "").str.strip()

        predictions = []
        for place in places:
            place_clean = place.lower().replace("'", "").replace("\u2019", "").strip()
            matched_row = temp_pct[temp_pct['Place_Clean'] == place_clean]
            if not matched_row.empty:
                pct = matched_row.iloc[0]['Crowd_Percentage']
                visitors = int(round(pred_base * pct / 100))
                predictions.append({"place": place, "predicted_visitors": visitors})
            else:
                predictions.append({"place": place, "predicted_visitors": None, "error": "Place not found"})

        return jsonify({
            "city": city,
            "date": date_str,
            "predictions": predictions
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/plan', methods=['POST'])
def plan_trip():
    data = request.json
    city = data.get('city', '')
    date_str = data.get('date', '')
    place = data.get('place', '') # Optional locked spot
    duration = data.get('duration', '3')

    if not city or not date_str:
        return jsonify({"error": "Missing parameters"}), 400

    try:
        print(f"[PLAN] city='{city}', date='{date_str}', place='{place}', duration='{duration}'")
        sb_response = supabase.table('crowd_data').select('spot_name').ilike('city_name', city).execute()
        all_city_spots = list(set([row['spot_name'] for row in sb_response.data]))
        print(f"[PLAN] Found {len(all_city_spots)} spots from Supabase: {all_city_spots}")
        if not all_city_spots:
            return jsonify({"error": f"No spots found for city: {city}"}), 404
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Supabase error: {str(e)}"}), 500

    ml_data = get_ml_data(city)
    if not ml_data:
        return jsonify({"error": f"No ML data found for {city}"}), 404

    model_wkday, model_wknd, crowd_pct = ml_data

    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        year, month_num = date_obj.year, date_obj.month
        is_weekend = date_obj.weekday() >= 5
        input_df = pd.DataFrame([[year, month_num]], columns=['Year', 'Month_Num'])
        pred_base = model_wknd.predict(input_df)[0] if is_weekend else model_wkday.predict(input_df)[0]

        temp_pct = crowd_pct.copy()
        temp_pct['Place_Clean'] = temp_pct['Place'].str.lower().str.replace("'", "").str.replace("\u2019", "").str.strip()

        # Build list of all available spots with predictions
        scored_spots = []
        for s in all_city_spots:
            s_clean = s.lower().replace("'", "").replace("\u2019", "").strip()
            row = temp_pct[temp_pct['Place_Clean'] == s_clean]
            if not row.empty:
                pct = float(row.iloc[0]['Crowd_Percentage'])
                visitors = int(round(pred_base * pct / 100))
                status = "Low" if pct <= 20 else ("Moderate" if pct <= 40 else "High")
                scored_spots.append({"spot": s, "prediction": visitors, "status": status, "pct": pct})

        # Logic to lock the selected place
        final_itinerary = []
        if place:
            place_clean = place.lower().replace("'", "").replace("\u2019", "").strip()
            locked_item = next((x for x in scored_spots if x['spot'].lower().replace("'", "").replace("\u2019", "").strip() == place_clean), None)
            if locked_item:
                final_itinerary.append(locked_item)
                # Remove from pool to avoid duplicates
                scored_spots = [x for x in scored_spots if x['spot'] != locked_item['spot']]

        # Fill remaining spots based on duration
        limit = 2 if duration == '3' else (4 if duration == '6' else len(scored_spots) + len(final_itinerary))
        remaining_needed = limit - len(final_itinerary)
        
        if remaining_needed > 0:
            # Add top remaining spots
            final_itinerary.extend(scored_spots[:remaining_needed])

        return jsonify({
            "city": city,
            "date": date_str,
            "duration": duration,
            "locked_spot": place,
            "itinerary": final_itinerary
        })
    except Exception as e:
        traceback.print_exc() # Added traceback
        return jsonify({"error": str(e)}), 500

@app.route('/')
@app.route('/index.html')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/list')
@app.route('/list.html')
def list_page():
    return send_from_directory('.', 'list.html')

@app.route('/plan_result.html')
def plan_result():
    return send_from_directory('.', 'plan_result.html')

# ─── Supabase setup ───────────────────────────────────────────────────────────
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip().strip('"').strip("'")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "").strip().strip('"').strip("'")

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    print("CRITICAL ERROR: Supabase credentials not found in environment!")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# ─── YOLO setup ───────────────────────────────────────────────────────────────
# Global lock for thread-safe inference
inference_lock = threading.Lock()
model = YOLO('yolov8n.pt')

# Pre-initialize model
print("Initializing YOLO model...")
import numpy as np
dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
with inference_lock:
    model(dummy_frame, verbose=False)
print("YOLO model initialized and fused.")

UPDATE_INTERVAL = 10  # Increased to save database/CPU

# Video sources configuration
VIDEO_SOURCES = [
    {"path": "Wax Museum.mp4", "spot": "Wax Museum", "capacity": 200},
    {"path": "Deer Park.mp4", "spot": "Deer Park", "capacity": 100},
]

def count_people_in_frame(frame):
    # Use global lock to prevent concurrent inference
    with inference_lock:
        try:
            # Standard image size for consistent results
            results = model(frame, conf=0.25, imgsz=640, verbose=False)
            person_count = 0
            for result in results:
                for box in result.boxes:
                    if int(box.cls) == 0:  # class 0 = person in COCO
                        person_count += 1
            return person_count
        except Exception as e:
            print(f"⚠️ Inference error: {e}")
            return 0

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
        print(f"✗ Update error for {spot_name}: {e}")
        return False

def process_video(video_path, spot_name, capacity):
    print(f"🎥 Tracking: {spot_name}")
    
    while True:
        # Verify file exists before opening
        if not os.path.exists(video_path):
            print(f"⚠️ Warning: Video file '{video_path}' missing for {spot_name}. Retrying in 30s...")
            time.sleep(30)
            continue

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Error: Cannot open video file '{video_path}' for {spot_name}")
            time.sleep(10)
            continue

        frame_count = 0
        person_counts = []
        last_update_time = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    if person_counts:
                        avg_count = int(sum(person_counts) / len(person_counts))
                        update_supabase(avg_count, spot_name, capacity)
                    break

                # Process every 15th frame - CRITICAL for memory
                if frame_count % 15 == 0:
                    try:
                        # Resize frame IMMEDIATELY to reduce memory footprint
                        # Using 640 as max dimension for YOLO
                        h, w = frame.shape[:2]
                        if w > 640:
                            scale = 640 / w
                            frame = cv2.resize(frame, (640, int(h * scale)))
                        
                        person_count = count_people_in_frame(frame)
                        person_counts.append(person_count)
                        
                        # Aggressive memory cleanup
                        del frame
                    except Exception: # Catch any error during frame processing/inference
                        print(f"🚨 Error during frame processing in {spot_name}. Restarting video stream.")
                        break # Exit inner loop to release and restart

                frame_count += 1

                current_time = time.time()
                if current_time - last_update_time >= UPDATE_INTERVAL:
                    if person_counts:
                        avg_count = int(sum(person_counts) / len(person_counts))
                        update_supabase(avg_count, spot_name, capacity)
                    person_counts = []
                    last_update_time = current_time
                    gc.collect() # Frequent GC

                # Yield to other threads
                time.sleep(0.02)
        except Exception as e:
            print(f"🔥 Unexpected error processing {spot_name}: {e}")
        finally:
            cap.release()
            print(f"🔄 Restarting video loop for {spot_name}")
            # Explicit garbage collection
            gc.collect()
            time.sleep(5)

# ─── Entry point ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # Stagger thread starts to prevent OOM peak
    def start_threads():
        for source in VIDEO_SOURCES:
            t = threading.Thread(
                target=process_video, 
                args=(source["path"], source["spot"], source["capacity"]), 
                daemon=True
            )
            t.start()
            print(f"🤖 Monitoring {source['spot']}")
            time.sleep(5) # Stagger

    threading.Thread(target=start_threads, daemon=True).start()

    # Start Flask web server (main thread)
    port = int(os.environ.get('PORT', 5000))
    print(f"🌐 Flask running at http://127.0.0.1:{port}\n")
    app.run(host='0.0.0.0', port=port, debug=False)
