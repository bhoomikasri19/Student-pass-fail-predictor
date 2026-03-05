from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,classification_report
import pandas as pd
import os 
import joblib

os.makedirs("model",exist_ok=True)

#loading
df=pd.read_csv("/Users/bhoomikasrivastava19/Documents/student_predictor/student-mat.csv")

print(df.columns.tolist())
print(df.shape)
df["pass"] = (df["G3"] >= 10).astype(int)

X=df[["studytime","failures","absences", "health","famrel", "goout"]]
y=df["pass"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=Pipeline([
    ("scaler",StandardScaler()),
    ("classifier",LogisticRegression(class_weight="balanced"))
])

model.fit(X_train,y_train)

print("Train accuracy:",round(model.score(X_train,y_train),3))
print("Test accuracy:",round(model.score(X_test,y_test),3))
print()
print(classification_report(y_test,model.predict(X_test)))

joblib.dump(model,"model/student_model.pkl")
print("Model saved!")