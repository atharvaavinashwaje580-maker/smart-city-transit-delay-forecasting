import numpy as np
import pandas as pd

def generate(path="data/demo_transit_delay.csv", n=6000, seed=42):
    rng = np.random.default_rng(seed)
    area = rng.choice(["Urban", "Rural"], n, p=[0.68, 0.32])
    route_type = rng.choice(["Local Bus", "Express Bus", "Feeder Bus"], n, p=[0.55, 0.25, 0.20])
    hour = rng.integers(5, 23, n)
    dow = rng.integers(0, 7, n)
    weather = rng.choice(["Clear", "Cloudy", "Rain", "Heavy Rain"], n, p=[0.50, 0.20, 0.22, 0.08])
    road = rng.choice(["Good", "Average", "Poor"], n, p=[0.45, 0.40, 0.15])
    traffic = np.clip(rng.normal(48, 23, n), 0, 100)
    rain = np.clip(rng.gamma(1.2, 4.0, n), 0, 35)
    passenger = np.clip(rng.normal(58, 20, n), 10, 100)
    headway = np.clip(rng.normal(14, 6, n), 3, 35)
    stops = rng.integers(5, 35, n)
    distance = np.clip(rng.normal(11, 5, n), 2, 30)
    prev_delay = np.clip(rng.normal(3.5, 5, n), 0, 35)
    event = rng.binomial(1, 0.10, n)
    peak = (((hour >= 7) & (hour <= 10)) | ((hour >= 17) & (hour <= 20))).astype(int)
    delay = (0.045*traffic + 0.055*rain + 0.045*passenger + 0.22*prev_delay
             + 1.8*peak + (area=="Urban")*2.5 + (area=="Rural")*1.5
             + np.where(route_type=="Local Bus",2.5,np.where(route_type=="Feeder Bus",1.5,0))
             + np.select([weather=="Cloudy",weather=="Rain",weather=="Heavy Rain"],[0.5,3,7],default=0)
             + np.select([road=="Average",road=="Poor"],[1.5,4],default=0)
             + 2.5*event + 0.10*stops + 0.08*distance - 0.06*headway
             + rng.normal(0,3.5,n))
    delay = np.clip(delay,0,45)
    category = pd.cut(delay,[-0.1,5,15,100],labels=["On Time / Minor","Moderate Delay","Major Delay"]).astype(str)
    out = pd.DataFrame({
        "area_type":area,"route_type":route_type,"hour":hour,"day_of_week":dow,
        "weather":weather,"rainfall_mm":np.round(rain,2),"traffic_index":np.round(traffic,2),
        "passenger_load_pct":np.round(passenger,2),"scheduled_headway_min":np.round(headway,2),
        "stop_count":stops,"distance_km":np.round(distance,2),"road_condition":road,
        "previous_delay_min":np.round(prev_delay,2),"special_event":event,
        "delay_minutes":np.round(delay,2),"delay_category":category})
    out.to_csv(path,index=False)

if __name__ == "__main__":
    generate()
