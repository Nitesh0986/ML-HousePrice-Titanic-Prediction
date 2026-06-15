import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Sample Titanic Dataset
data = {
    "Age": [22,38,26,35,35,28,2,54,27,14],
    "Sex": [0,1,1,1,0,0,0,1,1,0],  # 0=Male,1=Female
    "Pclass": [3,1,3,1,3,2,3,1,2,3],
    "Survived": [0,1,1,1,0,1,1,0,1,0]
}

df = pd.DataFrame(data)

X = df[["Age","Sex","Pclass"]]
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, pred))