from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

router = APIRouter()

class SimplificationRequest(BaseModel):
    text: str
    target_level: Optional[str] = "simple"
    language: Optional[str] = "en"

class SimplificationResponse(BaseModel):
    original_text: str
    simplified_text: str
    simplification_level: str

@router.post("/", response_model=SimplificationResponse)
async def simplify_text(request: SimplificationRequest):
    raise HTTPException(
        status_code=501,
        detail="Not implemented yet"
    )
