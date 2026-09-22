import pandas as pd
import joblib

def predict_delay(input_row, model_path="outputs/transit_delay_model.joblib"):
    model = joblib.load(model_path)
    return model.predict(pd.DataFrame([input_row]))[0]

# Example:
# row = {
#   "area_type":"Urban","route_type":"Local Bus","hour":18,"day_of_week":2,
#   "weather":"Rain","rainfall_mm":8.5,"traffic_index":72,
#   "passenger_load_pct":80,"scheduled_headway_min":10,"stop_count":20,
#   "distance_km":12,"road_condition":"Average","previous_delay_min":7,
#   "special_event":0
# }
# print(predict_delay(row))
