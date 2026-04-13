import pytest
import pandas as pd
import os
import sqlite3
from data_layer.connectors.sql_connector import SQLConnector, SQLConnectionError

@pytest.fixture
def mock_df():
    return pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
        "age": [25, 30, 35]
    })

@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "test.db"
    return str(path)

@pytest.fixture
def sqlite_connector(db_path, mock_df):
    conn = sqlite3.connect(db_path)
    mock_df.to_sql("employees", conn, index=False)
    conn.close()
    
    connector = SQLConnector(db_type="sqlite", sqlite_path=db_path)
    yield connector
    connector.disconnect()

def test_sqlite_connection_and_get_tables(sqlite_connector):
    tables = sqlite_connector.get_tables()
    assert "employees" in tables
    assert len(tables) >= 1

def test_load_table(sqlite_connector, mock_df):
    df_loaded = sqlite_connector.load_table("employees")
    pd.testing.assert_frame_equal(df_loaded, mock_df)

def test_save_table_update_db(sqlite_connector, mock_df):
    # Alter dataframe and save back
    df_altered = mock_df.copy()
    df_altered["age"] = [26, 31, 36]
    
    msg = sqlite_connector.save_data(df_altered, "employees", if_exists="replace")
    assert "Successfully updated" in msg
    
    # Reload and verify
    df_reloaded = sqlite_connector.load_table("employees")
    pd.testing.assert_frame_equal(df_reloaded, df_altered)

def test_save_table_export_csv(sqlite_connector, mock_df, tmp_path):
    export_path = str(tmp_path / "exported.csv")
    csv_path = sqlite_connector.save_data(mock_df, "employees", export_format="csv", export_path=export_path)
    
    assert os.path.exists(csv_path)
    df_csv = pd.read_csv(csv_path)
    pd.testing.assert_frame_equal(df_csv, mock_df)

def test_invalid_table_load(sqlite_connector):
    with pytest.raises(ValueError, match="does not exist"):
        sqlite_connector.load_table("nonexistent_table")

def test_initialization_errors():
    with pytest.raises(ValueError, match="sqlite_path is required"):
        SQLConnector(db_type="sqlite")
        
    with pytest.raises(ValueError, match="host, user, password"):
         SQLConnector(db_type="mysql")

def test_invalid_db_type():
    with pytest.raises(ValueError, match="Unsupported database type"):
         SQLConnector(db_type="oracle", host="localhost")

# Note: Fully testing MySQL and PostgreSQL in a CI/CD environment usually requires 
# active database servers or mocking the SQLAlchemy create_engine call. 
# Here we mock the connection strictly for PostgreSQL to verify it strings together correctly.
def test_postgres_connection_string_mock(mocker):
    # Mocking create_engine to prevent actual connections
    mock_engine = mocker.patch("data_layer.connectors.sql_connector.create_engine")
    
    # Setup mock inspector
    mock_inspector = mocker.patch("data_layer.connectors.sql_connector.inspect")
    mock_inspector_instance = mock_inspector.return_value
    mock_inspector_instance.get_table_names.return_value = ["pg_table"]
    
    # Initialize connector
    connector = SQLConnector(db_type="postgresql", host="localhost", port=5432, 
                             user="usr", password="pw", database="db")
    
    # Verify connection string
    assert connector.connection_string == "postgresql://usr:pw@localhost:5432/db"
    
    # Verify get_tables works through mock
    tables = connector.get_tables()
    assert "pg_table" in tables
    
    # Disconnect
    connector.disconnect()
