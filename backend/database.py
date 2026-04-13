from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os

# Create backend/data directory if it doesn't exist
os.makedirs(os.path.join(os.path.dirname(__file__), 'data'), exist_ok=True)

SQLALCHEMY_DATABASE_URL = "sqlite:///./backend/data/sol.db"
# If running script from backend folder directly instead of run folder
# keeping the path robust.
db_path = os.path.join(os.path.dirname(__file__), 'data', 'sol.db')
SQLALCHEMY_DATABASE_URL = f"sqlite:///{db_path}"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Dependency for FastAPI
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
