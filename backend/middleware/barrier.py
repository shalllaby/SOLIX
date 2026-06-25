from fastapi import Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from backend.auth import get_current_user
from backend.database import get_db
from backend.models import User, UserSettings

class CredentialsBarrier:
    def __init__(self, credentials_list: list[str]):
        self.credentials_list = credentials_list

    async def __call__(self, request: Request, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
        if not current_user:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated.")
            
        settings = db.query(UserSettings).filter_by(user_id=current_user.id).first()
        
        prop_map = {
            "groq_api_key": "groq_api_key",
            "GROQ_API_KEY": "groq_api_key",
            "kaggle_username": "kaggle_username",
            "KAGGLE_USERNAME": "kaggle_username",
            "kaggle_key": "kaggle_key",
            "KAGGLE_KEY": "kaggle_key",
            "KAGGLE_API_TOKEN": "kaggle_key",
            "elevenlabs_api_key": "elevenlabs_api_key",
            "elevenlabs_id": "elevenlabs_id"
        }

        for cred in self.credentials_list:
            prop_name = prop_map.get(cred, cred)
            val = None
            if settings:
                val = getattr(settings, prop_name, None)
                
            if not val or val.strip() == "" or "your_api" in val or val.startswith("gsk_XpIqZ"):
                friendly_name = prop_name.replace("_", " ").upper()
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Remote Credentials Required: Missing or invalid key '{friendly_name}'."
                )
