from fastapi import APIRouter
from app.api.v1.endpoints import health, simplification

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(simplification.router, tags=["simplification"], prefix="/simplify")