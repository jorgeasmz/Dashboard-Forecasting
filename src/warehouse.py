"""
Reads a series from the energy platform's warehouse.

The warehouse is a source like the published CSVs: it is read once when a series
is ingested and the observations are stored here. Nothing the API or the
dashboard serves reaches it, so a demo stays up while the platform's database is
asleep, and a backtest refits without a network round trip per fold.
"""

import pandas as pd
import psycopg

from src.config import WarehouseColumn

# The measurement columns marts.fct_system_hourly declares under its contract. A
# name reaches the query as text, so it is checked against that list rather than
# trusted.
COLUMNS = frozenset({"spot_price", "real_demand", "commercial_demand", "generation"})

GRAINS = (None, "day")

HOURLY = """
    select measured_at as ds, {column}::double precision as y
    from marts.fct_system_hourly
    where {column} is not null
    order by measured_at
"""

# A day missing an hour would average over fewer values than the rest, so it is
# left out rather than compared against complete ones.
DAILY = """
    select market_date::timestamp as ds, avg({column})::double precision as y
    from marts.fct_system_hourly
    where {column} is not null
    group by market_date
    having count({column}) = 24
    order by market_date
"""


def query(origin: WarehouseColumn) -> str:
    """The statement one origin reads, with its column checked against the mart."""
    if origin.column not in COLUMNS:
        raise ValueError(
            f"'{origin.column}' is not a measurement column of marts.fct_system_hourly."
        )
    if origin.grain not in GRAINS:
        raise ValueError(f"Unsupported grain '{origin.grain}'.")

    template = HOURLY if origin.grain is None else DAILY
    return template.format(column=origin.column)


def read(origin: WarehouseColumn, url: str) -> pd.DataFrame:
    """One column of the mart as a ds/y frame, aggregated in the warehouse."""
    if not url:
        raise RuntimeError(
            "WAREHOUSE_URL is unset, so the series it feeds cannot be ingested. "
            "The other series do not need it."
        )

    with psycopg.connect(url) as connection, connection.cursor() as cursor:
        cursor.execute(query(origin))
        rows = cursor.fetchall()

    return pd.DataFrame(rows, columns=["ds", "y"])
