
import logging

from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import database
from app.api.routes import chat, creative


# --------------------------------------------------
# LOGGING
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)


# --------------------------------------------------
# APPLICATION LIFESPAN
# --------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):

    logging.info("Initializing NEXORA AI database...")

    database.init_db()

    logging.info("Database initialized successfully.")

    yield

    logging.info("NEXORA AI application shutting down.")


# --------------------------------------------------
# FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="NEXORA AI",
    description="Advanced Multimodal AI Assistant",
    version="1.0.0",
    lifespan=lifespan
)


# --------------------------------------------------
# CORS CONFIGURATION
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    # allow_origins=[
    #     "http://localhost:5173",
    #     "http://127.0.0.1:5173"
    # ],
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# --------------------------------------------------
# ROUTERS
# --------------------------------------------------

app.include_router(
    chat.router,
    prefix="/api"
)

app.include_router(creative.router, prefix="/api")
# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/")
def health_check():

    return {
        "application": "NEXORA AI",
        "status": "running",
        "version": "1.0.0"
    }
