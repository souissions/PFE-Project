from fastapi import FastAPI, APIRouter, Depends
import os
from helpers.config import get_settings, Settings #get_settings is used to get the settings from configuration such as app name and version



base_router = APIRouter(
    prefix="/api/v1", # Prefix for all routes in this router
    tags=["api_v1"], # Tags to group routes in Swagger UI
)

@base_router.get("/")
async def welcome(app_settings: Settings = Depends(get_settings)):
    """
    Endpoint for a basic welcome message, showing app name and version.
    """

    app_name = app_settings.APP_NAME
    app_version = app_settings.APP_VERSION

    return {
        "app_name": app_name,
        "app_version": app_version,
    }
