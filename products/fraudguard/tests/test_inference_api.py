from fastapi.testclient import TestClient

from products.fraudguard.inference.app import main


class DummyModelLoader:
    def is_ready(self) -> bool:
        return True

    def predict_probability(self, dataframe) -> float:
        assert len(dataframe) == 1
        return 0.73


def test_health_endpoint():
    client = TestClient(main.app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_endpoint(monkeypatch):
    monkeypatch.setattr(main, "model_loader", DummyModelLoader())

    client = TestClient(main.app)
    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"ready": True}


def test_predict_endpoint(monkeypatch):
    monkeypatch.setattr(main, "model_loader", DummyModelLoader())

    client = TestClient(main.app)

    response = client.post(
        "/predict",
        json={
            "TransactionAmt": 129.99,
            "ProductCD": "W",
            "card1": 12345,
            "card2": 321,
            "card3": 150,
            "card4": "visa",
            "card5": 226,
            "card6": "credit",
            "addr1": 204,
            "addr2": 87,
            "P_emaildomain": "gmail.com",
            "R_emaildomain": "gmail.com",
            "DeviceType": "desktop",
            "DeviceInfo": "Windows",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["fraud_probability"] == 0.73
    assert body["decision"] == "review"
    assert body["threshold"] == 0.5
    assert body["model_version"] == "local"