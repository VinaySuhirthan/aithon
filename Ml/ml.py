import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
import os
import sys

def load_and_train_for_city(city_name):
    # Determine the path based on the city name
    city_folder = city_name
    base_dir = os.path.dirname(os.path.abspath(__file__))
    city_path = os.path.join(base_dir, city_folder)
    
    if not os.path.exists(city_path):
        print(f"Error: Could not find data folder for '{city_name}' at {city_path}")
        return None, None, None
        
    print(f"\nLoading data from folder: '{city_folder}'...")
    
    # Find all training data files (e.g. ooty_2015.csv) and crowd percentage
    try:
        all_files = os.listdir(city_path)
    except Exception as e:
        print(f"Error reading directory {city_path}: {e}")
        return None, None, None
        
    # We filter for the training CSVs and the percentage CSV
    train_files = [f for f in all_files if f.lower().endswith('.csv') and '20' in f and 'percent' not in f.lower()]
    pct_files = [f for f in all_files if 'percent' in f.lower() and f.lower().endswith('.csv')]
    
    if not train_files:
        print(f"Error: No training CSVs found for {city_name}.")
        return None, None, None
        
    if not pct_files:
        print(f"Error: No crowd percentage CSV found for {city_name}.")
        return None, None, None

    # Load and combine training data
    data_list = []
    for file in train_files:
        full_path = os.path.join(city_path, file)
        df = pd.read_csv(full_path)
        data_list.append(df)

    all_data = pd.concat(data_list, ignore_index=True)

    # 2. Preprocessing
    # Handle NaN values to prevent ValueError during data processing
    all_data = all_data.dropna(subset=['Year', 'Month', 'Avg_Weekday_Crowd', 'Avg_Weekend_Crowd'])

    month_map = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4,
        'May': 5, 'June': 6, 'July': 7, 'August': 8,
        'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    # Map month names to numbers, leaving existing numerical months intact
    all_data['Month_Num'] = all_data['Month'].replace(month_map).astype(int)

    # Feature Selection
    X = all_data[['Year', 'Month_Num']]
    y_weekday = all_data['Avg_Weekday_Crowd']
    y_weekend = all_data['Avg_Weekend_Crowd']

    # 3. Model Training
    print("Training models...")
    model_weekday = RandomForestRegressor(n_estimators=100, random_state=42)
    model_weekend = RandomForestRegressor(n_estimators=100, random_state=42)

    model_weekday.fit(X, y_weekday)
    model_weekend.fit(X, y_weekend)
    print("Models trained successfully!")
    
    # Load Percentage Data
    crowd_pct_path = os.path.join(city_path, pct_files[0])
    crowd_pct = pd.read_csv(crowd_pct_path)
    
    return model_weekday, model_weekend, crowd_pct

# Single date prediction based on user input
def run_interactive_prediction():
    print("\n" + "=" * 50)
    print("      TOURIST CROWD PREDICTOR")
    print("=" * 50)
    
    city_name = input("Enter the city: ").strip()
    if not city_name:
        print("City is required!")
        return
        
    model_weekday, model_weekend, crowd_pct = load_and_train_for_city(city_name)
    
    if model_weekday is None:
        return
        
    print(f"\n--- {city_name} Crowd Predictor ---")
    print("Format: DD/MM/YYYY")
    
    try:
        user_input = input("Enter the date: ").strip()
        date_obj = datetime.strptime(user_input, "%d/%m/%Y")
        
        # Loop until a valid place is entered
        while True:
            place_input = input("Enter the place: ").strip()
            if not place_input:
                print("Place is mandatory. Please enter a valid place.")
                continue
                
            matched_row = crowd_pct[crowd_pct['Place'] == place_input]
            if not matched_row.empty:
                break
            else:
                available_places = ', '.join(crowd_pct['Place'].tolist())
                print(f"Place '{place_input}' not found.\nAvailable places: {available_places}\nPlease try again.")
        
        year = date_obj.year
        month_num = date_obj.month
        is_weekend = date_obj.weekday() >= 5 # 5 is Saturday, 6 is Sunday
        
        # Prepare input for model
        input_data = pd.DataFrame([[year, month_num]], columns=['Year', 'Month_Num'])
        
        if is_weekend:
            pred_crowd = model_weekend.predict(input_data)[0]
            day_type = "Weekend"
        else:
            pred_crowd = model_weekday.predict(input_data)[0]
            day_type = "Weekday"
        
        total_crowd = round(pred_crowd)
        
        pct = matched_row.iloc[0]['Crowd_Percentage']
        visitors = round(total_crowd * pct / 100)
        
        # Display results for the specific place only
        print("\n" + "=" * 50)
        print(f"  City  : {city_name}")
        print(f"  Date  : {date_obj.strftime('%d %B %Y')} ({date_obj.strftime('%A')})")
        print(f"  Place : {matched_row.iloc[0]['Place']}")
        print(f"  Type  : {day_type}")
        print(f"  PREDICTED CROWD FOR {matched_row.iloc[0]['Place'].upper()} : {visitors}")
        print("=" * 50)
        
    except ValueError:
        print("Invalid format! Please use DD/MM/YYYY")

if __name__ == "__main__":
    run_interactive_prediction()
