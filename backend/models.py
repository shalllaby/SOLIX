from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, JSON, Boolean, Float
from sqlalchemy.sql import func
from datetime import datetime, timezone
from .database import Base

def get_utc_now():
    return datetime.now(timezone.utc)

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    first_name = Column(String, index=True, nullable=True)
    last_name = Column(String, index=True, nullable=True)
    username = Column(String, unique=True, index=True, nullable=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    status = Column(String, default="active") # "pending_verification" or "active"
    job_title = Column(String, nullable=True)
    organization = Column(String, nullable=True)
    avatar_url = Column(String, nullable=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class OTPSession(Base):
    __tablename__ = "otp_sessions"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, index=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=True)
    otp_hash = Column(String, nullable=False) # SHA-256 hash of the 6-digit code
    expires_at = Column(DateTime, nullable=False)
    attempts = Column(Integer, default=0) # Tracks failed verification attempts
    is_verified = Column(Boolean, default=False)
    created_at = Column(DateTime, default=get_utc_now)
    updated_at = Column(DateTime, default=get_utc_now, onupdate=get_utc_now)

class AuthLog(Base):
    __tablename__ = "auth_logs"

    id = Column(Integer, primary_key=True, index=True)
    ip_address = Column(String, nullable=True)
    email = Column(String, nullable=True, index=True)
    user_agent = Column(String, nullable=True)
    action = Column(String, nullable=False) # e.g., "register", "login_success", etc.
    status = Column(String, nullable=False, default="info")
    details = Column(String, nullable=True)
    timestamp = Column(DateTime, default=get_utc_now)


class Form(Base):
    __tablename__ = "forms"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=True)
    title = Column(String, index=True)
    description = Column(Text)
    questions = Column(JSON) # Store array of dicts: type, question, options
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class FormResponse(Base):
    __tablename__ = "responses"

    id = Column(Integer, primary_key=True, index=True)
    form_id = Column(Integer, ForeignKey("forms.id"))
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=True)
    answers = Column(JSON) # JSON obj Mapping question to answer
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

class Notification(Base):
    __tablename__ = "notifications"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    title = Column(String, nullable=False)
    message = Column(Text, nullable=False)
    type = Column(String, default="info") # info, warning, error, success
    is_read = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class JobRecord(Base):
    __tablename__ = "job_records"

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(String, unique=True, index=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    task_type = Column(String, nullable=False) # "cleaning", "synthetic", "ocr", "automl"
    filename = Column(String, nullable=True)
    file_size_bytes = Column(Integer, default=0)
    row_count = Column(Integer, nullable=True)
    col_count = Column(Integer, nullable=True)
    strategy = Column(String, nullable=True) # e.g. "Alpha", "Beta", "Gamma"
    status = Column(String, default="pending") # "pending", "processing", "completed", "failed"
    accuracy_rate = Column(Float, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=get_utc_now)
    updated_at = Column(DateTime, default=get_utc_now, onupdate=get_utc_now)

class TokenUsageRecord(Base):
    __tablename__ = "token_usage_records"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String, index=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=True)
    prompt_tokens = Column(Integer, default=0)
    completion_tokens = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    module_name = Column(String, index=True, nullable=True) # e.g. "cleaning", "narrator", "advisor", "semantic", "synthetic"
    timestamp = Column(DateTime, default=get_utc_now)

class Project(Base):
    __tablename__ = "projects"

    id = Column(String, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=True)
    title = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=get_utc_now)
    updated_at = Column(DateTime, default=get_utc_now, onupdate=get_utc_now)

class Task(Base):
    __tablename__ = "tasks"

    id = Column(String, primary_key=True, index=True)
    project_id = Column(String, ForeignKey("projects.id", ondelete="SET NULL"), nullable=True)
    name = Column(String, nullable=False)
    status = Column(String, default="PENDING")  # PENDING, RUNNING, COMPLETED, FAILED, PAUSED
    progress_percentage = Column(Integer, default=0)
    state_data = Column(JSON, nullable=True)
    created_by = Column(Integer, ForeignKey("users.id"), nullable=False)
    assigned_worker_id = Column(String, nullable=True)
    version_id = Column(Integer, default=1)
    created_at = Column(DateTime, default=get_utc_now)
    updated_at = Column(DateTime, default=get_utc_now, onupdate=get_utc_now)

class TaskRun(Base):
    __tablename__ = "task_runs"

    id = Column(String, primary_key=True, index=True)
    task_id = Column(String, ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=True)
    started_at = Column(DateTime, default=get_utc_now)
    finished_at = Column(DateTime, nullable=True)
    error_log = Column(Text, nullable=True)
    result_metadata = Column(JSON, nullable=True)


class Feedback(Base):
    __tablename__ = "feedbacks"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, nullable=False)
    phone = Column(String, nullable=True)
    message = Column(Text, nullable=False)
    created_at = Column(DateTime, default=get_utc_now)


class UserSettings(Base):
    __tablename__ = "user_settings"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False)
    
    # Store encrypted credentials as strings
    _kaggle_username = Column("kaggle_username", String, nullable=True)
    _kaggle_key = Column("kaggle_key", String, nullable=True)
    _groq_api_key = Column("groq_api_key", String, nullable=True)
    _elevenlabs_api_key = Column("elevenlabs_api_key", String, nullable=True)
    _elevenlabs_id = Column("elevenlabs_id", String, nullable=True)

    @property
    def kaggle_username(self) -> str:
        from .utils.security import decrypt_value
        return decrypt_value(self._kaggle_username)

    @kaggle_username.setter
    def kaggle_username(self, value: str):
        from .utils.security import encrypt_value
        self._kaggle_username = encrypt_value(value)

    @property
    def kaggle_key(self) -> str:
        from .utils.security import decrypt_value
        return decrypt_value(self._kaggle_key)

    @kaggle_key.setter
    def kaggle_key(self, value: str):
        from .utils.security import encrypt_value
        self._kaggle_key = encrypt_value(value)

    @property
    def groq_api_key(self) -> str:
        from .utils.security import decrypt_value
        return decrypt_value(self._groq_api_key)

    @groq_api_key.setter
    def groq_api_key(self, value: str):
        from .utils.security import encrypt_value
        self._groq_api_key = encrypt_value(value)

    @property
    def elevenlabs_api_key(self) -> str:
        from .utils.security import decrypt_value
        return decrypt_value(self._elevenlabs_api_key)

    @elevenlabs_api_key.setter
    def elevenlabs_api_key(self, value: str):
        from .utils.security import encrypt_value
        self._elevenlabs_api_key = encrypt_value(value)

    @property
    def elevenlabs_id(self) -> str:
        from .utils.security import decrypt_value
        return decrypt_value(self._elevenlabs_id)

    @elevenlabs_id.setter
    def elevenlabs_id(self, value: str):
        from .utils.security import encrypt_value
        self._elevenlabs_id = encrypt_value(value)


class DashboardPreferences(Base):
    __tablename__ = "dashboard_preferences"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False)
    layout = Column(JSON, nullable=True) # JSON configuration for layout widgets
    theme_preference = Column(String, default="dark") # UI theme
    created_at = Column(DateTime, default=get_utc_now)
    updated_at = Column(DateTime, default=get_utc_now, onupdate=get_utc_now)


import uuid

class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(String(255), default="New Chat")
    created_at = Column(DateTime, default=get_utc_now)
    updated_at = Column(DateTime, default=get_utc_now, onupdate=get_utc_now)


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(36), ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String(50), nullable=False)  # "user" or "bot"
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=get_utc_now)


from sqlalchemy import event, insert, select

@event.listens_for(User, "after_insert")
def receive_after_insert(mapper, connection, target):
    # Auto-provision empty/default settings and preferences upon user creation
    # Check if they exist first to handle potential duplicate listener registration
    s_exists = connection.execute(
        select(1).select_from(UserSettings.__table__).where(UserSettings.__table__.c.user_id == target.id)
    ).scalar()
    if not s_exists:
        connection.execute(
            insert(UserSettings.__table__).values(
                user_id=target.id,
                kaggle_username=None,
                kaggle_key=None,
                groq_api_key=None,
                elevenlabs_api_key=None,
                elevenlabs_id=None
            )
        )
    
    p_exists = connection.execute(
        select(1).select_from(DashboardPreferences.__table__).where(DashboardPreferences.__table__.c.user_id == target.id)
    ).scalar()
    if not p_exists:
        connection.execute(
            insert(DashboardPreferences.__table__).values(
                user_id=target.id,
                layout="{}",
                theme_preference="dark"
            )
        )


