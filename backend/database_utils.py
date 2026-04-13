import pandas as pd
from sqlalchemy import create_engine, inspect
import urllib.parse
from typing import List

class DatabaseManager:
    def __init__(self):
        self.engines = {}

    def build_uri(self, db_type: str, host: str, port: str, user: str, password: str, db_name: str) -> str:
        safe_pass = urllib.parse.quote_plus(password) if password else ""
        if db_type.lower() == "postgresql":
            return f"postgresql+psycopg2://{user}:{safe_pass}@{host}:{port}/{db_name}"
        elif db_type.lower() in ["mysql", "mariadb"]:
            return f"mysql+pymysql://{user}:{safe_pass}@{host}:{port}/{db_name}"
        elif db_type.lower() == "sqlite":
            return f"sqlite:///{host}"  # for sqlite, host is the filepath
        else:
            raise ValueError(f"Unsupported database type: {db_type}")

    def connect(self, connection_id: str, db_type: str, host: str, port: str, user: str, password: str, db_name: str) -> bool:
        try:
            uri = self.build_uri(db_type, host, port, user, password, db_name)
            engine = create_engine(uri)
            # Test connection
            with engine.connect() as conn:
                pass
            self.engines[connection_id] = engine
            return True
        except Exception as e:
            raise RuntimeError(f"Failed to connect: {str(e)}")

    def get_tables(self, connection_id: str) -> List[str]:
        if connection_id not in self.engines:
            raise ValueError("Connection not found.")
        try:
            inspector = inspect(self.engines[connection_id])
            return inspector.get_table_names()
        except Exception as e:
            raise RuntimeError(f"Failed to get tables: {str(e)}")

    def import_table(self, connection_id: str, table_name: str) -> pd.DataFrame:
        if connection_id not in self.engines:
            raise ValueError("Connection not found.")
        try:
            query = f"SELECT * FROM {table_name}"
            df = pd.read_sql(query, self.engines[connection_id])
            return df
        except Exception as e:
            raise RuntimeError(f"Failed to import table {table_name}: {str(e)}")

    def sync_table(self, connection_id: str, table_name: str, df: pd.DataFrame, if_exists: str = "replace") -> bool:
        if connection_id not in self.engines:
            raise ValueError("Connection not found.")
        try:
            df.to_sql(table_name, self.engines[connection_id], if_exists=if_exists, index=False)
            return True
        except Exception as e:
            raise RuntimeError(f"Failed to sync table {table_name}: {str(e)}")
