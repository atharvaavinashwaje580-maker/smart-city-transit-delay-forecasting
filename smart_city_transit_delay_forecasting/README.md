# Smart City Public Transit Delay Forecasting

## Project title
**Smart City Public Transit Delay Forecasting for Urban and Rural Areas Using Supervised Machine Learning**

### Problem statement
Public transportation delays affect passenger waiting time, route reliability, fleet planning and access to employment, education and healthcare. The project develops a supervised machine-learning prototype that predicts a **delay category** (On Time/Minor, Moderate, Major) from operational, temporal, environmental and area-level features.

### Objectives
- Analyse factors associated with transit delays.
- Compare delay patterns between urban and rural settings.
- Build a supervised classification model in Python.
- Evaluate model performance using accuracy, precision, recall, F1-score and a confusion matrix.
- Create a reusable workflow that can later ingest public GTFS/GTFS-Realtime data.

### Features used
`area_type`, `route_type`, `hour`, `day_of_week`, `weather`, `rainfall_mm`, `traffic_index`, `passenger_load_pct`, `scheduled_headway_min`, `stop_count`, `distance_km`, `road_condition`, `previous_delay_min`, `special_event`

### Target
`delay_category`
- On Time / Minor
- Moderate Delay
- Major Delay

## Machine-learning method
1. Data cleaning
2. Train/test split with stratification
3. Missing-value handling
4. One-hot encoding for categorical features
5. Standardisation of numeric features
6. Logistic Regression baseline
7. Random Forest classifier
8. Accuracy and classification report
9. Confusion matrix and visual analysis

## Repository structure
```text
smart_city_transit_delay_forecasting/
├── data/
│   ├── demo_transit_delay.csv
│   └── README.md
├── notebooks/
│   └── Transit_Delay_Forecasting.ipynb
├── outputs/
│   ├── model_comparison.csv
│   ├── delay_category_distribution.png
│   ├── urban_rural_comparison.png
│   ├── traffic_vs_delay.png
│   └── confusion_matrix.png
├── reports/
│   ├── Smart_City_Transit_Delay_Report.docx
│   └── Smart_City_Transit_Delay_Report.pdf
├── src/
│   ├── generate_demo_data.py
│   ├── train_model.py
│   └── predict.py
├── requirements.txt
└── README.md
```

## Run locally
```bash
pip install -r requirements.txt
python src/generate_demo_data.py
python src/train_model.py
```

## Data-source note
The included CSV is **synthetic demonstration data**, generated deterministically so the GitHub repository runs immediately. It should not be described as real transit observations.

For a final college submission using secondary public data, use a documented public GTFS/GTFS-Realtime source. Useful starting points include the Pune GTFS case-study repository and public Delhi bus transit data. GTFS-Realtime Trip Updates can contain delay/schedule-deviation values, making it suitable for constructing a real delay target.

## Suggested research extension
For a stronger final version, join:
- GTFS schedule: route, trip, stop, scheduled arrival/departure
- GTFS-Realtime: observed/predicted delay
- weather: rainfall, temperature, visibility
- road/traffic: congestion index or speed
- calendar: weekday/weekend/holiday
- geography: urban/rural classification

Then define:
`delay_minutes = actual_or_predicted_arrival - scheduled_arrival`

and classify it into delay bands.

## Academic integrity
This repository is a reproducible project template. Clearly identify the source, licensing terms, date range, preprocessing and whether any variables are simulated or derived.
