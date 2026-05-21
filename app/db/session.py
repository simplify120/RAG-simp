from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from urllib.parse import quote_plus, urlparse, urlunparse
from app.core.config import settings

def encode_password_in_url(url: str) -> str:
    if not url.startswith("postgresql://"):
        return url
    try:
        parsed = urlparse(url)
        if parsed.password:
            encoded_password = quote_plus(parsed.password)
            if encoded_password != parsed.password:
                netloc = f"{parsed.username}:{encoded_password}@{parsed.hostname}"
                if parsed.port:
                    netloc += f":{parsed.port}"
                return urlunparse((
                    parsed.scheme, netloc, parsed.path,
                    parsed.params, parsed.query, parsed.fragment
                ))
    except Exception:
        pass
    return url

if settings.DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        settings.DATABASE_URL,
        connect_args={"check_same_thread": False},
        pool_pre_ping=True
    )
elif settings.DATABASE_URL.startswith("postgresql"):
    encoded_url = encode_password_in_url(settings.DATABASE_URL)
    engine = create_engine(
        encoded_url,
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=20,
        pool_recycle=3600,
        connect_args={"connect_timeout": 30}
    )
else:
    engine = create_engine(
        settings.DATABASE_URL,
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=20
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
