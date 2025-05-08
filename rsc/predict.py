import joblib

def predict(sample):
    model = joblib.load("model.pkl")
    return model.predict(sample)

if __name__ == "__main__":
    print(f"Predicted class: {predict(sample=[[5.1,3.5, 1.4, 0.2]])}")