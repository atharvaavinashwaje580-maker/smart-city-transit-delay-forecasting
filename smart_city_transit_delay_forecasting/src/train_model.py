import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pathlib import Path

DATA = Path("data/demo_transit_delay.csv")
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA)
X = df.drop(columns=["delay_minutes","delay_category"])
y = df["delay_category"]

cat_cols = ["area_type","route_type","weather","road_condition"]
num_cols = [c for c in X.columns if c not in cat_cols]

pre = ColumnTransformer([
    ("num", Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())]), num_cols),
    ("cat", Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]), cat_cols)
])

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.20,random_state=42,stratify=y
)

model = Pipeline([
    ("preprocessor",pre),
    ("classifier",RandomForestClassifier(
        n_estimators=250,max_depth=14,class_weight="balanced",
        random_state=42,n_jobs=-1
    ))
])
model.fit(X_train,y_train)
pred = model.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test,pred),4))
print(classification_report(y_test,pred))
print("Confusion matrix:")
print(confusion_matrix(y_test,pred))
