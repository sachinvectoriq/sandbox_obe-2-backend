"""FastAPI application entry point for Knowledge Assistant Agentic RAG."""

import os
from contextlib import asynccontextmanager
from sys import exc_info

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from azure.cosmos import PartitionKey

# Application Insights / OpenTelemetry telemetry is DISABLED in this non-prod build.

from app.core.container import Container
from app.api.routes import chat, health, pipeline

from app.utils import audit
from app.utils import feedback
from app.utils import saml
from app.utils import report
from app.utils import report_access
from app.utils.footer_metadata import router as footer_router
from app.utils.image_extraction_utils import router as image_extraction_router
from app.utils.split_skill import split_skill_router
from app.utils.azure_enrichment_api import router as enrichment_router
from app.utils.tag_log import router as tag_log_router

# Initialize dependency injection container (singleton)
container = Container()
logger = container.logger()

__all__ = ["app", "container", "logger"]


async def _ensure_cosmos_resources() -> None:
    cosmos_options = container.cosmos_db_options()
    if not (cosmos_options.connection_string or cosmos_options.endpoint):
        logger.warning("Cosmos DB not configured - skipping conversation history setup")
        return

    try:
        logger.info("Ensuring Cosmos DB resources exist...")
        cosmos_client = container.cosmos_client()

        db = await cosmos_client.create_database_if_not_exists(
            id=cosmos_options.database_name
        )
        logger.info(f"[Cosmos] Database ready: {cosmos_options.database_name}")

        container_kwargs = {}
        if getattr(cosmos_options, "enable_ttl", False):
            default_ttl_days = int(getattr(cosmos_options, "default_ttl_days", 30))
            container_kwargs["default_ttl"] = default_ttl_days * 24 * 60 * 60

        indexing_policy = {
            "indexingMode": "consistent",
            "automatic": True,
            "includedPaths": [{"path": "/*"}],
            "excludedPaths": [{"path": "/serialized_message/?"}],
        }

        hpk = PartitionKey(
            path=["/user_id", "/session_id"],
            kind="MultiHash",
            version=2,
        )

        await db.create_container_if_not_exists(
            id=cosmos_options.container_name,
            partition_key=hpk,
            indexing_policy=indexing_policy,
            **container_kwargs,
        )

        logger.info(
            f"[Cosmos] Container ready: {cosmos_options.container_name} "
            f"(pk=[/user_id, /session_id] HPK)"
        )

    except Exception as e:
        logger.error(f"Failed to provision Cosmos DB resources: {e}")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Knowledge Assistant API starting up...")

    try:
        settings = container.config()
        logger.info("Configuration loaded successfully:")
        logger.info(f"  Environment: {os.getenv('ENVIRONMENT', 'development')}")
        logger.info(f"  Search Service: {settings.search_service.endpoint}")
        logger.info(f"  Search Index: {settings.search_service.index_name}")
        logger.info(f"  Azure OpenAI: {settings.azure_openai.endpoint}")
        logger.info(f"  Chat Model Deployment Name: {settings.azure_openai.deployment_name}")
        logger.info(f"  Embedding Model Deployment Name: {settings.azure_openai.embedding_deployment_name}")
        logger.info(
            f"  Cosmos DB: {'Configured' if settings.cosmos_db.endpoint or settings.cosmos_db.connection_string else 'Not configured'}"
        )
        logger.info(
            f"  Blob Storage: {settings.blob_storage.resource_id if hasattr(settings.blob_storage, 'resource_id') else 'Configured'}"
        )
        logger.info(f"  Use Managed Identity: {settings.use_managed_identity}")

        await _ensure_cosmos_resources()

    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise

    yield

    logger.info("Knowledge Assistant API shutting down...")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Knowledge Assistant Agentic RAG",
        description="FastAPI backend for multimodal document search with agentic RAG",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Telemetry/FastAPI instrumentation disabled in non-prod build.

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(chat.router, prefix="/api")
    app.include_router(pipeline.router, prefix="/api")
    app.include_router(audit.router, prefix="/api")
    app.include_router(feedback.router, prefix="/api")
    app.include_router(saml.router)
    app.include_router(report.router, prefix="/api")
    app.include_router(report_access.router, prefix="/api")
    app.include_router(footer_router, prefix="/api")
    app.include_router(image_extraction_router, prefix="/api")
    app.include_router(split_skill_router, prefix="/api")
    app.include_router(enrichment_router, prefix="/api")
    app.include_router(tag_log_router, prefix="/api")
    
    return app


app = create_app()
