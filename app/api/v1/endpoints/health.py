from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from app.db.session import get_db

router = APIRouter()

@router.get("/health")
def health():
    return {"status": "ok"}

@router.get("/health/db")
def health_db(db: Session = Depends(get_db)):
    try:
        result = db.execute(text("SELECT current_database()"))
        db_name = result.fetchone()[0]
        return {
            "status": "ok",
            "database": "connected",
            "database_name": db_name
        }
    except SQLAlchemyError as e:
        return {
            "status": "error",
            "database": "connection_failed",
            "error": str(e)
        }
