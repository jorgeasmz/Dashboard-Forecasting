def test_health_counts_the_ingested_series(client):
    body = client.get("/").json()

    assert body["series"] == 1


def test_series_listing_carries_the_selected_model(client):
    items = client.get("/series").json()

    assert len(items) == 1
    assert items[0]["slug"] == "test-series"
    assert items[0]["best_model"] == "prophet"
    assert items[0]["observations"] == 60


def test_observations_come_back_in_order(client):
    items = client.get("/series/test-series/observations").json()

    timestamps = [item["ts"] for item in items]
    assert timestamps == sorted(timestamps)


def test_evaluation_is_ordered_by_scaled_error(client):
    items = client.get("/series/test-series/evaluation").json()

    assert [item["model"] for item in items] == ["prophet", "seasonal_naive", "sarima"]


def test_forecast_uses_the_selected_model(client):
    body = client.get("/series/test-series/forecast?horizon=3").json()

    assert body["model"] == "prophet"
    assert body["horizon"] == 3
    assert len(body["points"]) == 3


def test_forecast_points_carry_an_interval(client):
    point = client.get("/series/test-series/forecast?horizon=1").json()["points"][0]

    assert point["yhat_lower"] <= point["yhat"] <= point["yhat_upper"]


def test_an_unknown_series_is_a_404(client):
    assert client.get("/series/nope/evaluation").status_code == 404
    assert client.get("/series/nope/observations").status_code == 404
    assert client.get("/series/nope/forecast").status_code == 404


def test_the_horizon_is_bounded(client):
    assert client.get("/series/test-series/forecast?horizon=0").status_code == 422
    assert client.get("/series/test-series/forecast?horizon=999").status_code == 422
