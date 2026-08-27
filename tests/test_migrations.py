from alembic.config import Config
from sqlalchemy import create_engine, inspect

from alembic import command
from src import config


def test_migrations_build_the_schema_from_empty(tmp_path, monkeypatch):
    """Guards against the schema drifting away from the models."""
    url = f"sqlite:///{tmp_path / 'migrated.db'}"
    monkeypatch.setattr(config, "DATABASE_URL", url)

    alembic_config = Config(str(config.PROJECT_ROOT / "alembic.ini"))
    alembic_config.set_main_option("script_location", str(config.PROJECT_ROOT / "alembic"))
    command.upgrade(alembic_config, "head")

    inspector = inspect(create_engine(url))
    assert {"series", "observations", "evaluations"} <= set(inspector.get_table_names())
