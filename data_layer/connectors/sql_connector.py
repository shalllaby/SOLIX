import pandas as pd
from sqlalchemy import create_engine, inspect
import sqlalchemy.exc
from typing import List, Optional

class SQLConnectionError(Exception):
    """Exception raised for database connection and operation errors."""
    pass

class SQLConnector:
    def __init__(self, db_type: str, host: str = None, port: int = None, 
                 user: str = None, password: str = None, database: str = None,
                 sqlite_path: str = None):
        """
        Initialize the SQL Connector with database credentials.
        
        Args:
            db_type (str): 'sqlite', 'mysql', or 'postgresql'.
            host (str): Database host (for mysql/postgresql).
            port (int): Database port (for mysql/postgresql).
            user (str): Database user (for mysql/postgresql).
            password (str): Database password (for mysql/postgresql).
            database (str): Database name (for mysql/postgresql).
            sqlite_path (str): File path for SQLite database.
        """
        self.db_type = db_type.lower()
        self.engine = None
        
        if self.db_type == 'sqlite':
            if not sqlite_path:
                raise ValueError("sqlite_path is required for SQLite connections.")
            self.connection_string = f"sqlite:///{sqlite_path}"
        elif self.db_type in ['mysql', 'postgresql']:
            if not all([host, user, password, database]):
                raise ValueError(f"host, user, password, and database are required for {self.db_type}.")
            
            # Using standard drivers: pymysql for MySQL, psycopg2 for PostgreSQL
            if self.db_type == 'mysql':
                self.connection_string = f"mysql+pymysql://{user}:{password}@{host}:{port or 3306}/{database}"
            else:
                self.connection_string = f"postgresql://{user}:{password}@{host}:{port or 5432}/{database}"
        else:
            raise ValueError(f"Unsupported database type: {db_type}. Use 'sqlite', 'mysql', or 'postgresql'.")
            
    def connect(self):
        """Establish connection securely to the database."""
        try:
            self.engine = create_engine(self.connection_string)
            # Test the connection proactively
            with self.engine.connect() as conn:
                pass
        except sqlalchemy.exc.OperationalError as e:
            raise SQLConnectionError(f"Failed to connect to {self.db_type} database (check credentials/host): {e}")
        except Exception as e:
            raise SQLConnectionError(f"Unexpected connection error: {e}")
            
    def get_tables(self) -> List[str]:
        """List all available tables in the connected database."""
        if not self.engine:
            self.connect()
        try:
            inspector = inspect(self.engine)
            return inspector.get_table_names()
        except Exception as e:
            raise SQLConnectionError(f"Failed to list tables: {e}")
            
    def load_table(self, table_name: str) -> pd.DataFrame:
        """Fetch a specific table into a Pandas DataFrame."""
        if not self.engine:
            self.connect()
        
        tables = self.get_tables()
        if table_name not in tables:
            raise ValueError(f"Table '{table_name}' does not exist in the database.")
            
        try:
            return pd.read_sql_table(table_name, con=self.engine)
        except sqlalchemy.exc.ProgrammingError as e:
             raise SQLConnectionError(f"Permission or schema error when loading table '{table_name}': {e}")
        except Exception as e:
            raise SQLConnectionError(f"Failed to load table '{table_name}': {e}")
            
    def save_data(self, df: pd.DataFrame, table_name: str, 
                  if_exists: str = "replace", 
                  export_format: Optional[str] = None, 
                  export_path: Optional[str] = None) -> str:
        """
        Saves the DataFrame back to either the SQL database or exports to a file.
        
        Args:
            df (pd.DataFrame): The cleaned DataFrame to save.
            table_name (str): Original table name.
            if_exists (str): 'replace', 'append', or 'fail' (When updating database).
            export_format (str): Optional. If provided ('csv', 'excel', etc.), exports to a file instead.
            export_path (str): Optional. Custom path for exporting file.
            
        Returns:
            str: Path to the exported file or a success message for database updates.
        """
        if export_format:
            fmt = export_format.lower()
            if not export_path:
                 export_path = f"cleaned_{table_name}.{fmt}"
                 
            try:
                if fmt == 'csv':
                    df.to_csv(export_path, index=False)
                elif fmt in ['excel', 'xlsx']:
                    df.to_excel(export_path, index=False)
                elif fmt == 'json':
                    df.to_json(export_path, orient='records', indent=4)
                elif fmt == 'parquet':
                    df.to_parquet(export_path, index=False)
                else:
                    raise ValueError(f"Unsupported export format: {export_format}")
                return export_path
            except Exception as e:
                 raise SQLConnectionError(f"Failed to export data to {export_format}: {e}")
        else:
            # Option 1: Update original database table
            if not self.engine:
                self.connect()
            try:
                df.to_sql(table_name, con=self.engine, if_exists=if_exists, index=False)
                return f"Successfully updated table '{table_name}' in database."
            except sqlalchemy.exc.ProgrammingError as e:
                 raise SQLConnectionError(f"Permission error when updating table '{table_name}': {e}")
            except Exception as e:
                raise SQLConnectionError(f"Failed to update table '{table_name}': {e}")
                
    def disconnect(self):
        """Dispose of the database engine properly."""
        if self.engine:
            self.engine.dispose()
            self.engine = None
