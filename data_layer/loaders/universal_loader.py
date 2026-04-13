import os
import io
import pandas as pd
from abc import ABC, abstractmethod
from typing import Union, BinaryIO, Dict, Type

class DataLoadError(Exception):
    """Custom exception for data loading errors."""
    pass

class BaseLoader(ABC):
    """Abstract base class for all file loaders."""
    
    @abstractmethod
    def load(self, source: Union[str, BinaryIO], **kwargs) -> pd.DataFrame:
        """
        Load data from a given source into a Pandas DataFrame.
        
        Args:
            source (Union[str, BinaryIO]): File path or file-like object.
            **kwargs: Additional keyword arguments passed to the underlying Pandas read function.
            
        Returns:
            pd.DataFrame: Cleaned and normalized Pandas DataFrame.
            
        Raises:
            DataLoadError: If the file is corrupt, malformed, or cannot be read.
        """
        pass

class CSVLoader(BaseLoader):
    """Loader for CSV files."""
    def load(self, source: Union[str, BinaryIO], **kwargs) -> pd.DataFrame:
        try:
            return pd.read_csv(source, **kwargs)
        except Exception as e:
            raise DataLoadError(f"Failed to load CSV file: {e}")

class ExcelLoader(BaseLoader):
    """Loader for Excel files (.xlsx, .xls)."""
    def load(self, source: Union[str, BinaryIO], **kwargs) -> pd.DataFrame:
        try:
            return pd.read_excel(source, **kwargs)
        except Exception as e:
            raise DataLoadError(f"Failed to load Excel file: {e}")

class JSONLoader(BaseLoader):
    """Loader for JSON files."""
    def load(self, source: Union[str, BinaryIO], **kwargs) -> pd.DataFrame:
        try:
            return pd.read_json(source, **kwargs)
        except Exception as e:
            raise DataLoadError(f"Failed to load JSON file: {e}")

class ParquetLoader(BaseLoader):
    """Loader for Parquet files."""
    def load(self, source: Union[str, BinaryIO], **kwargs) -> pd.DataFrame:
        try:
            return pd.read_parquet(source, **kwargs)
        except Exception as e:
            raise DataLoadError(f"Failed to load Parquet file: {e}")

class XMLLoader(BaseLoader):
    """Loader for XML files."""
    def load(self, source: Union[str, BinaryIO], **kwargs) -> pd.DataFrame:
        try:
            return pd.read_xml(source, **kwargs)
        except Exception as e:
            raise DataLoadError(f"Failed to load XML file: {e}")

class FeatherLoader(BaseLoader):
    """Loader for Feather files."""
    def load(self, source: Union[str, BinaryIO], **kwargs) -> pd.DataFrame:
        try:
            return pd.read_feather(source, **kwargs)
        except Exception as e:
            raise DataLoadError(f"Failed to load Feather file: {e}")

class HDF5Loader(BaseLoader):
    """Loader for HDF5 files."""
    def load(self, source: Union[str, BinaryIO], **kwargs) -> pd.DataFrame:
        import tempfile
        try:
            if hasattr(source, 'read'):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp:
                    source.seek(0)
                    tmp.write(source.read())
                    tmp_path = tmp.name
                try:
                    return pd.read_hdf(tmp_path, **kwargs)
                finally:
                    os.remove(tmp_path)
            else:
                return pd.read_hdf(source, **kwargs)
        except Exception as e:
            raise DataLoadError(f"Failed to load HDF5 file: {e}")

class ORCLoader(BaseLoader):
    """Loader for ORC files."""
    def load(self, source: Union[str, BinaryIO], **kwargs) -> pd.DataFrame:
        try:
            return pd.read_orc(source, **kwargs)
        except Exception as e:
            raise DataLoadError(f"Failed to load ORC file: {e}")

class SQLLoader(BaseLoader):
    """Loader for SQL Databases (SQLite, PostgreSQL, MySQL) via SQLAlchemy."""
    def load(self, source: Union[str, BinaryIO], **kwargs) -> pd.DataFrame:
        from sqlalchemy import create_engine, inspect
        import tempfile
        try:
            engine = None
            tmp_path = None
            if isinstance(source, str) and "://" in source:
                engine = create_engine(source)
            else:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as tmp:
                    if hasattr(source, 'read'):
                        source.seek(0)
                        tmp.write(source.read())
                    else:
                        with open(source, 'rb') as f:
                            tmp.write(f.read())
                    tmp_path = tmp.name
                engine = create_engine(f"sqlite:///{tmp_path}")

            try:
                query = kwargs.get('sql') or kwargs.get('table_name')
                if not query:
                    inspector = inspect(engine)
                    tables = inspector.get_table_names()
                    if not tables:
                        raise ValueError("No tables found in the database.")
                    query = tables[0]
                return pd.read_sql(query, engine)
            finally:
                if engine is not None:
                    engine.dispose()
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass # Ignore if still locked
        except Exception as e:
            raise DataLoadError(f"Failed to load SQL database: {e}")


class DataLoaderFactory:
    """
    Factory class to dynamically dispatch the appropriate loader based on file extension.
    This pattern makes the system modular and open for extension.
    """
    # Initial supported loaders registry
    _loaders: Dict[str, Type[BaseLoader]] = {
        '.csv': CSVLoader,
        '.xlsx': ExcelLoader,
        '.xls': ExcelLoader,
        '.json': JSONLoader,
        '.parquet': ParquetLoader,
        '.xml': XMLLoader,
        '.feather': FeatherLoader,
        '.h5': HDF5Loader,
        '.hdf5': HDF5Loader,
        '.orc': ORCLoader,
        '.db': SQLLoader,
        '.sqlite': SQLLoader
    }

    @classmethod
    def register_loader(cls, extension: str, loader_class: Type[BaseLoader]):
        """
        Register a new loader for a specific file extension.
        
        Args:
            extension (str): The file extension (e.g., '.yaml').
            loader_class (Type[BaseLoader]): The loader class to handle this extension.
        """
        if not extension.startswith('.'):
            extension = f".{extension}"
        cls._loaders[extension.lower()] = loader_class

    @classmethod
    def get_loader(cls, extension: str) -> BaseLoader:
        """
        Retrieve the appropriate loader based on the file extension.
        
        Args:
            extension (str): File extension.
            
        Returns:
            BaseLoader: An instance of the matching loader.
            
        Raises:
            ValueError: If the extension is not supported.
        """
        loader_class = cls._loaders.get(extension.lower())
        if not loader_class:
             raise ValueError(f"Unsupported file format: '{extension}'. Supported formats: {list(cls._loaders.keys())}")
        return loader_class()

    @classmethod
    def load_data(cls, source: Union[str, BinaryIO], file_name: str = None, **kwargs) -> pd.DataFrame:
        """
        Main entry point to load data automatically detecting the file type.
        
        Args:
            source (Union[str, BinaryIO]): File path or file-like object.
            file_name (str, optional): Overrides the file name if `source` is a stream. 
                                     Required if `source` is a file-like object without a `.name` attribute.
            **kwargs: Additional arguments for Pandas loaders.
            
        Returns:
            pd.DataFrame: A normalized Pandas DataFrame.
        """
        if file_name:
            _, ext = os.path.splitext(file_name)
        elif isinstance(source, str):
            if "://" in source:
                ext = ".db" # Default to SQL loader for connection strings
            else:
                _, ext = os.path.splitext(source)
        elif hasattr(source, 'name'):
            _, ext = os.path.splitext(source.name)
        else:
            raise ValueError("Could not determine file extension. Please provide 'file_name' explicitly.")
            
        if not ext:
            raise ValueError(f"File name '{file_name or source}' does not have an extension.")
            
        loader = cls.get_loader(ext)
        return loader.load(source, **kwargs)
