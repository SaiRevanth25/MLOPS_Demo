from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import joblib

def train_model():
    iris = load_iris()
    model = RandomForestClassifier()
    model.fit(iris.data, iris.target)
    joblib.dump(model, "model.pkl")
    print("Model trained and saved!")

if __name__ == "__main__":
    train_model()