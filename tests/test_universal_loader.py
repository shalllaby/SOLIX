import os
import io
import pytest
import pandas as pd
from data_layer.loaders.universal_loader import DataLoaderFactory, DataLoadError, CSVLoader

@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
        "age": [25, 30, 35]
    })

def test_csv_loader(sample_dataframe, tmp_path):
    file_path = tmp_path / "data.csv"
    sample_dataframe.to_csv(file_path, index=False)
    
    df = DataLoaderFactory.load_data(str(file_path))
    pd.testing.assert_frame_equal(df, sample_dataframe)

def test_excel_loader(sample_dataframe, tmp_path):
    file_path = tmp_path / "data.xlsx"
    sample_dataframe.to_excel(file_path, index=False)
    
    df = DataLoaderFactory.load_data(str(file_path))
    pd.testing.assert_frame_equal(df, sample_dataframe)

def test_json_loader(sample_dataframe, tmp_path):
    file_path = tmp_path / "data.json"
    sample_dataframe.to_json(file_path, orient="records")
    
    df = DataLoaderFactory.load_data(str(file_path))
    pd.testing.assert_frame_equal(df, sample_dataframe)

def test_parquet_loader(sample_dataframe, tmp_path):
    file_path = tmp_path / "data.parquet"
    sample_dataframe.to_parquet(file_path, index=False)
    
    df = DataLoaderFactory.load_data(str(file_path))
    pd.testing.assert_frame_equal(df, sample_dataframe)

def test_xml_loader(sample_dataframe, tmp_path):
    file_path = tmp_path / "data.xml"
    sample_dataframe.to_xml(file_path, index=False)
    
    df = DataLoaderFactory.load_data(str(file_path))
    pd.testing.assert_frame_equal(df, sample_dataframe)

def test_feather_loader(sample_dataframe, tmp_path):
    file_path = tmp_path / "data.feather"
    sample_dataframe.to_feather(file_path)
    
    df = DataLoaderFactory.load_data(str(file_path))
    pd.testing.assert_frame_equal(df, sample_dataframe)

def test_hdf5_loader(sample_dataframe, tmp_path):
    file_path = tmp_path / "data.h5"
    sample_dataframe.to_hdf(file_path, key='df', mode='w')
    
    df = DataLoaderFactory.load_data(str(file_path))
    pd.testing.assert_frame_equal(df, sample_dataframe)

def test_orc_loader(sample_dataframe, tmp_path):
    file_path = tmp_path / "data.orc"
    sample_dataframe.to_orc(file_path)
    
    df = DataLoaderFactory.load_data(str(file_path))
    pd.testing.assert_frame_equal(df, sample_dataframe)

def test_sql_loader(sample_dataframe, tmp_path):
    import sqlite3
    file_path = tmp_path / "data.db"
    
    import sqlalchemy
    engine = sqlalchemy.create_engine(f"sqlite:///{file_path}")
    sample_dataframe.to_sql("data", engine, index=False, if_exists="replace")
    
    # test by file path
    df = DataLoaderFactory.load_data(str(file_path))
    pd.testing.assert_frame_equal(df, sample_dataframe)
    
    # test by connection string
    conn_str = f"sqlite:///{file_path}"
    df_conn = DataLoaderFactory.load_data(conn_str)
    pd.testing.assert_frame_equal(df_conn, sample_dataframe)

def test_corrupt_files(tmp_path):
    # test bad feather
    bad_feather = tmp_path / "bad.feather"
    bad_feather.write_text("not a feather file")
    with pytest.raises(DataLoadError, match="Failed to load Feather file"):
        DataLoaderFactory.load_data(str(bad_feather))
        
    # test bad hdf5
    bad_hdf = tmp_path / "bad.h5"
    bad_hdf.write_text("not an hdf5 file")
    with pytest.raises(DataLoadError, match="Failed to load HDF5 file"):
        DataLoaderFactory.load_data(str(bad_hdf))
        
    # test bad orc
    bad_orc = tmp_path / "bad.orc"
    bad_orc.write_text("not an orc file")
    with pytest.raises(DataLoadError, match="Failed to load ORC file"):
        DataLoaderFactory.load_data(str(bad_orc))
        
    # test bad sql
    bad_sql = tmp_path / "bad.db"
    bad_sql.write_text("not a sql file")
    with pytest.raises(DataLoadError, match="Failed to load SQL database"):
        DataLoaderFactory.load_data(str(bad_sql))

def test_unsupported_extension():
    with pytest.raises(ValueError, match="Unsupported file format"):
        DataLoaderFactory.load_data("data.unknown")

def test_stream_loading(sample_dataframe):
    csv_buffer = io.StringIO()
    sample_dataframe.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    # Needs file_name to infer extension when using stream
    df = DataLoaderFactory.load_data(csv_buffer, file_name="data.csv")
    pd.testing.assert_frame_equal(df, sample_dataframe)

def test_malformed_file(tmp_path):
    file_path = tmp_path / "bad_data.csv"
    file_path.write_text("id,name,age\n1,Alice,25\n2,Bob,broken,row,here")
    
    with pytest.raises(DataLoadError, match="Failed to load CSV file"):
        # Pandas would return ParseError which we wrap in DataLoadError
        DataLoaderFactory.load_data(str(file_path), on_bad_lines='error')

def test_custom_loader_registration():
    class DummyLoader(CSVLoader):
         pass
         
    DataLoaderFactory.register_loader('.dummy', DummyLoader)
    assert isinstance(DataLoaderFactory.get_loader('.dummy'), DummyLoader)
