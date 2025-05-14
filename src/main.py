from fastapi import FastAPI
from controllers import base, data, nlp
from helpers.config import get_settings
from stores.llm.LLMProviderFactory import LLMProviderFactory
from stores.vectordb.VectorDBProviderFactory import VectorDBProviderFactory
from stores.llm.templates.template_parser import TemplateParser
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import logging

logger = logging.getLogger("uvicorn")

app = FastAPI()

async def startup_span():
    try:
        settings = get_settings()
        logger.info("Starting application with settings loaded")

        # Initialize database connection
        postgres_conn = f"postgresql+asyncpg://{settings.POSTGRES_USERNAME}:{settings.POSTGRES_PASSWORD}@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_MAIN_DATABASE}"
        logger.info(f"Connecting to database at {settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}")

        app.db_engine = create_async_engine(postgres_conn)
        app.db_client = sessionmaker(
            app.db_engine, class_=AsyncSession, expire_on_commit=False
        )

        # Initialize LLM providers
        logger.info(f"Initializing LLM provider: {settings.GENERATION_BACKEND}")
        llm_provider_factory = LLMProviderFactory(settings)
        app.generation_client = llm_provider_factory.create(provider=settings.GENERATION_BACKEND)
        if not app.generation_client:
            raise ValueError(f"Failed to create LLM provider: {settings.GENERATION_BACKEND}")
        app.generation_client.set_generation_model(model_id=settings.GENERATION_MODEL_ID)
        logger.info(f"LLM provider initialized with model: {settings.GENERATION_MODEL_ID}")

        # Initialize embedding provider
        logger.info(f"Initializing embedding provider: {settings.EMBEDDING_BACKEND}")
        app.embedding_client = llm_provider_factory.create(provider=settings.EMBEDDING_BACKEND)
        if not app.embedding_client:
            raise ValueError(f"Failed to create embedding provider: {settings.EMBEDDING_BACKEND}")
        app.embedding_client.set_embedding_model(
            model_id=settings.EMBEDDING_MODEL_ID,
            embedding_size=settings.EMBEDDING_MODEL_SIZE
        )
        logger.info(f"Embedding provider initialized with model: {settings.EMBEDDING_MODEL_ID}")

        # Initialize vector DB provider
        logger.info(f"Initializing vector DB provider: {settings.VECTOR_DB_BACKEND}")
        vectordb_provider_factory = VectorDBProviderFactory(config=settings, db_client=app.db_client)
        app.vectordb_client = vectordb_provider_factory.create(provider=settings.VECTOR_DB_BACKEND)
        if not app.vectordb_client:
            raise ValueError(f"Failed to create vector DB provider: {settings.VECTOR_DB_BACKEND}")
        await app.vectordb_client.connect()
        logger.info("Vector DB provider initialized and connected")

        # Initialize template parser
        logger.info("Initializing template parser")
        app.template_parser = TemplateParser(
            language=settings.PRIMARY_LANG,
            default_language=settings.DEFAULT_LANG,
        )
        logger.info("Template parser initialized")

        logger.info("Application startup completed successfully")

    except Exception as e:
        logger.error(f"Error during application startup: {str(e)}")
        raise

async def shutdown_span():
    try:
        logger.info("Starting application shutdown")
        if hasattr(app, 'db_engine'):
            app.db_engine.dispose()
            logger.info("Database connection disposed")
        if hasattr(app, 'vectordb_client'):
            await app.vectordb_client.disconnect()
            logger.info("Vector DB connection closed")
        logger.info("Application shutdown completed")
    except Exception as e:
        logger.error(f"Error during application shutdown: {str(e)}")
        raise

app.on_event("startup")(startup_span)
app.on_event("shutdown")(shutdown_span)

app.include_router(base.base_router)
app.include_router(data.data_router)
app.include_router(nlp.nlp_router)
