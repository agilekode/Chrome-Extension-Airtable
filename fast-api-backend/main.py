from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import ask, webhook, search

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(ask.router)
app.include_router(webhook.router)
app.include_router(search.router)
