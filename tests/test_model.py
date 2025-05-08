import pytest

from rsc.train import train_model
from rsc.predict import predict

def load_data():
    from sklearn.datasets import load_iris
    data = load_iris()
    assert data.shape == (150, 4)

def test_training():
    train_model()  # Simple smoke test
    
def test_prediction():
    sample = [[5.1, 3.5, 1.4, 0.2]]
    result = predict(sample)
    assert result in [0, 1, 2]