from plotly.graph_objs import Figure

from src.plotting import plot_accuracy, plot_coverage, plot_forecast
from src.selection import aggregate


def test_forecast_figure_draws_history_prediction_and_interval(history, source):
    from src.forecasters import SeasonalNaive

    forecast = SeasonalNaive(12).fit(history).predict(6, "MS")

    figure = plot_forecast(history, forecast, "units")

    assert isinstance(figure, Figure)
    assert [trace.name for trace in figure.data] == [
        "Upper bound", "80% interval", "Forecast", "Observed"
    ]


def test_accuracy_figure_marks_the_baseline(session, populated):
    figure = plot_accuracy(aggregate(session, populated.slug))

    assert isinstance(figure, Figure)
    assert any(shape.y0 == 1.0 for shape in figure.layout.shapes)


def test_coverage_figure_marks_the_nominal_level(session, populated):
    figure = plot_coverage(aggregate(session, populated.slug))

    assert isinstance(figure, Figure)
    assert any(shape.y0 == 0.80 for shape in figure.layout.shapes)
