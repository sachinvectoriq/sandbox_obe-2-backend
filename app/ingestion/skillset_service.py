"""Skillset service for Azure AI Search."""

import httpx
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from datetime import timedelta

from azure.search.documents.indexes.aio import SearchIndexerClient
from azure.search.documents.indexes.models import (
    AIServicesAccountIdentity,
    AzureOpenAIEmbeddingSkill,
    DocumentIntelligenceLayoutSkill,
    DocumentIntelligenceLayoutSkillChunkingProperties,
    IndexProjectionMode,
    InputFieldMappingEntry,
    OutputFieldMappingEntry,
    SearchIndexerIndexProjection,
    SearchIndexerIndexProjectionSelector,
    SearchIndexerIndexProjectionsParameters,
    SearchIndexerKnowledgeStore,
    SearchIndexerKnowledgeStoreObjectProjectionSelector,
    SearchIndexerKnowledgeStoreProjection,
    SearchIndexerSkillset,
    ShaperSkill,
    SplitSkill,
    WebApiSkill,
)

from app.models.config_options import (
    AIServicesOptions,
    AzureOpenAIOptions,
    BlobStorageOptions,
    SearchServiceOptions,
)
from app.prompts.templates import IngestionPrompts


class ISkillsetService(ABC):
    async def create_skillset_using_sdk_async(
        self, skillset_name: str, index_name: str
    ) -> None:
        pass

    async def create_skillset_using_rest_async(
        self, skillset_name: str, index_name: str
    ) -> None:
        pass


class SkillsetService(ISkillsetService):
    IMAGE_VERBALIZATION_LITERAL = IngestionPrompts.as_search_string_literal(
        IngestionPrompts.IMAGE_VERBALIZATION_SYSTEM_MESSAGE
    )
    OPCO_EXTRACTION_LITERAL = IngestionPrompts.as_search_string_literal(
        IngestionPrompts.OPCO_EXTRACTION_SYSTEM_MESSAGE
    )
    PERSONA_EXTRACTION_LITERAL = IngestionPrompts.as_search_string_literal(
        IngestionPrompts.PERSONA_EXTRACTION_SYSTEM_MESSAGE
    )
    OPCO_NORMALIZATION_LITERAL = IngestionPrompts.as_search_string_literal(
        IngestionPrompts.OPCO_VALUE_NORMALIZATION_SYSTEM_MESSAGE
    )
    PERSONA_NORMALIZATION_LITERAL = IngestionPrompts.as_search_string_literal(
        IngestionPrompts.PERSONA_VALUE_NORMALIZATION_SYSTEM_MESSAGE
    )

    def __init__(
        self,
        search_indexer_client: SearchIndexerClient,
        search_options: SearchServiceOptions,
        openai_options: AzureOpenAIOptions,
        ai_services_options: AIServicesOptions,
        blob_options: BlobStorageOptions,
        logger,
    ) -> None:
        self._search_indexer_client: SearchIndexerClient = search_indexer_client
        self._search_options: SearchServiceOptions = search_options
        self._openai_options: AzureOpenAIOptions = openai_options
        self._ai_services_options: AIServicesOptions = ai_services_options
        self._blob_options: BlobStorageOptions = blob_options
        self.logger = logger

    async def create_skillset_using_sdk_async(
        self, skillset_name: str, index_name: str
    ) -> None:
        skills: List = [
            DocumentIntelligenceLayoutSkill(
                name="document-intelligence-layout-skill",
                description="Extract text and images with layout from documents using Document Intelligence",
                context="/document",
                output_mode="oneToMany",
                output_format="text",
                markdown_header_depth=None,
                extraction_options=["images", "locationMetadata"],
                chunking_properties=DocumentIntelligenceLayoutSkillChunkingProperties(
                    unit="characters",
                    maximum_length=3000,
                    overlap_length=500,
                ),
                inputs=[
                    InputFieldMappingEntry(name="file_data", source="/document/file_data")
                ],
                outputs=[
                    OutputFieldMappingEntry(name="text_sections", target_name="text_sections"),
                    OutputFieldMappingEntry(name="normalized_images", target_name="normalized_images"),
                ],
            ),
            AzureOpenAIEmbeddingSkill(
                name="text-chunk-embedding-skill",
                description="Generate embeddings for text chunks using Azure OpenAI",
                context="/document/text_sections/*",
                resource_url=self._openai_options.resource_uri,
                deployment_name=self._openai_options.text_embedding_model,
                dimensions=3072,
                model_name=self._openai_options.text_embedding_model,
                inputs=[
                    InputFieldMappingEntry(
                        name="text",
                        source="/document/text_sections/*/content",
                    )
                ],
                outputs=[
                    OutputFieldMappingEntry(name="embedding", target_name="text_vector")
                ],
            ),
            WebApiSkill(
                name="footer-metadata-skill",
                description="Extract and normalize Operating Companies and Persona Categories from footer text",
                context="/document",
                uri="https://app-ka-sandbox-001.azurewebsites.net/api/footer-metadata",
                http_method="POST",
                timeout=timedelta(minutes=3, seconds=50),
                batch_size=1,
                http_headers={},
                inputs=[
                    InputFieldMappingEntry(
                        name="metadata_storage_name",
                        source="/document/metadata_storage_name",
                    ),
                    InputFieldMappingEntry(
                        name="metadata_storage_container",
                        source=f"='{self._blob_options.container_name}'",
                    ),
                ],
                outputs=[
                    OutputFieldMappingEntry(
                        name="opco_values_array",
                        target_name="opco_values_array",
                    ),
                    OutputFieldMappingEntry(
                        name="persona_values_array",
                        target_name="persona_values_array",
                    ),
                ],
            ),
            WebApiSkill(
                name="image-verbalization-skill",
                description="Generate text descriptions of images using GPT vision (via custom WebApiSkill)",
                context="/document/normalized_images/*",
                uri="https://app-ka-sandbox-001.azurewebsites.net/api/image-verbalization",
                http_method="POST",
                timeout=timedelta(minutes=3, seconds=50),
                batch_size=1,
                http_headers={},
                inputs=[
                    InputFieldMappingEntry(
                        name="systemMessage",
                        source=self.IMAGE_VERBALIZATION_LITERAL,
                    ),
                    InputFieldMappingEntry(
                        name="userMessage",
                        source="='Please describe this image.'",
                    ),
                    InputFieldMappingEntry(
                        name="image",
                        source="/document/normalized_images/*/data",
                    ),
                ],
                outputs=[
                    OutputFieldMappingEntry(
                        name="response",
                        target_name="verbalizedImage"
                    ),
                ],
            ),
            AzureOpenAIEmbeddingSkill(
                name="image-description-embedding-skill",
                description="Generate embeddings for image descriptions using Azure OpenAI",
                context="/document/normalized_images/*",
                resource_url=self._openai_options.resource_uri,
                deployment_name=self._openai_options.text_embedding_model,
                dimensions=3072,
                model_name=self._openai_options.text_embedding_model,
                inputs=[
                    InputFieldMappingEntry(
                        name="text",
                        source="/document/normalized_images/*/verbalizedImage",
                    )
                ],
                outputs=[
                    OutputFieldMappingEntry(
                        name="embedding",
                        target_name="verbalizedImage_vector",
                    )
                ],
            ),
            ShaperSkill(
                name="image-path-shaper-skill",
                context="/document/normalized_images/*",
                inputs=[
                    InputFieldMappingEntry(
                        name="normalized_images",
                        source="/document/normalized_images/*",
                    ),
                    InputFieldMappingEntry(
                        name="imagePath",
                        source=f"='{self._blob_options.images_container_name}/' + $(/document/normalized_images/*/imagePath)",
                    ),
                ],
                outputs=[
                    OutputFieldMappingEntry(name="output", target_name="new_normalized_images")
                ],
            ),
        ]

        projection_selectors = [
            SearchIndexerIndexProjectionSelector(
                target_index_name=index_name,
                parent_key_field_name="text_document_id",
                source_context="/document/text_sections/*",
                mappings=[
                    InputFieldMappingEntry(
                        name="content_embedding",
                        source="/document/text_sections/*/text_vector",
                    ),
                    InputFieldMappingEntry(
                        name="content_text",
                        source="/document/text_sections/*/content",
                    ),
                    InputFieldMappingEntry(
                        name="location_metadata",
                        source="/document/text_sections/*/locationMetadata",
                    ),
                    InputFieldMappingEntry(
                        name="document_title",
                        source="/document/document_title",
                    ),
                    InputFieldMappingEntry(
                        name="opco_values",
                        source="/document/opco_values_array/*",
                    ),
                    InputFieldMappingEntry(
                        name="persona_values",
                        source="/document/persona_values_array/*",
                    ),
                ],
            ),
            SearchIndexerIndexProjectionSelector(
                target_index_name=index_name,
                parent_key_field_name="image_document_id",
                source_context="/document/normalized_images/*",
                mappings=[
                    InputFieldMappingEntry(
                        name="content_text",
                        source="/document/normalized_images/*/verbalizedImage",
                    ),
                    InputFieldMappingEntry(
                        name="content_embedding",
                        source="/document/normalized_images/*/verbalizedImage_vector",
                    ),
                    InputFieldMappingEntry(
                        name="content_path",
                        source="/document/normalized_images/*/new_normalized_images/imagePath",
                    ),
                    InputFieldMappingEntry(
                        name="document_title",
                        source="/document/document_title",
                    ),
                    InputFieldMappingEntry(
                        name="location_metadata",
                        source="/document/normalized_images/*/locationMetadata",
                    ),
                    InputFieldMappingEntry(
                        name="opco_values",
                        source="/document/opco_values_array/*",
                    ),
                    InputFieldMappingEntry(
                        name="persona_values",
                        source="/document/persona_values_array/*",
                    ),
                ],
            ),
        ]

        index_projections = SearchIndexerIndexProjection(
            selectors=projection_selectors,
            parameters=SearchIndexerIndexProjectionsParameters(
                projection_mode=IndexProjectionMode.SKIP_INDEXING_PARENT_DOCUMENTS
            ),
        )

        knowledge_store = SearchIndexerKnowledgeStore(
            storage_connection_string=f"ResourceId={self._blob_options.resource_id}",
            projections=[
                SearchIndexerKnowledgeStoreProjection(
                    objects=[
                        SearchIndexerKnowledgeStoreObjectProjectionSelector(
                            storage_container=self._blob_options.images_container_name,
                            source="/document/normalized_images/*",
                        )
                    ]
                )
            ],
            parameters={"synthesizeGeneratedKeyName": True},
        )

        endpoint_str = str(self._ai_services_options.cognitive_services_endpoint).rstrip("/")

        cognitive_services_account = AIServicesAccountIdentity(
            subdomain_url=endpoint_str,
        )

        skillset = SearchIndexerSkillset(
            name=skillset_name,
            description="A skillset for multimodal document processing with text and image extraction",
            skills=skills,
            cognitive_services_account=cognitive_services_account,
            index_projection=index_projections,
            knowledge_store=knowledge_store,
        )

        await self._search_indexer_client.create_or_update_skillset(skillset)

    async def create_skillset_using_rest_async(
        self,
        skillset_name: str,
        index_name: str,
    ) -> None:
        endpoint_str = str(self._ai_services_options.cognitive_services_endpoint).rstrip("/")

        cognitive_services = {
            "@odata.type": "#Microsoft.Azure.Search.AIServicesByIdentity",
            "subdomainUrl": endpoint_str,
        }

        skills: List[Dict[str, Any]] = [
            {
                "@odata.type": "#Microsoft.Skills.Util.DocumentIntelligenceLayoutSkill",
                "name": "document-intelligence-layout-skill",
                "description": "Extract text and images with layout from documents using Document Intelligence",
                "context": "/document",
                "outputMode": "oneToMany",
                "outputFormat": "text",
                "extractionOptions": ["images", "locationMetadata"],
                "chunkingProperties": {
                    "unit": "characters",
                    "maximumLength": 3000,
                    "overlapLength": 500,
                },
                "inputs": [
                    {"name": "file_data", "source": "/document/file_data"}
                ],
                "outputs": [
                    {"name": "text_sections", "targetName": "text_sections"},
                    {"name": "normalized_images", "targetName": "normalized_images"},
                ],
            },
            {
                "@odata.type": "#Microsoft.Skills.Text.AzureOpenAIEmbeddingSkill",
                "name": "text-chunk-embedding-skill",
                "description": "Generate embeddings for text chunks using Azure OpenAI",
                "context": "/document/text_sections/*",
                "resourceUri": self._openai_options.resource_uri,
                "deploymentId": self._openai_options.text_embedding_model,
                "dimensions": 3072,
                "modelName": self._openai_options.text_embedding_model,
                "inputs": [
                    {"name": "text", "source": "/document/text_sections/*/content"}
                ],
                "outputs": [
                    {"name": "embedding", "targetName": "text_vector"}
                ],
            },
            {
                "@odata.type": "#Microsoft.Skills.Custom.WebApiSkill",
                "name": "footer-metadata-skill",
                "description": "Extract and normalize Operating Companies and Persona Categories from footer text",
                "context": "/document",
                "uri": "https://app-ka-sandbox-001.azurewebsites.net/api/footer-metadata",
                "httpMethod": "POST",
                "timeout": "PT3M50S",
                "batchSize": 1,
                "inputs": [
                    {"name": "metadata_storage_name", "source": "/document/metadata_storage_name"},
                    {
                        "name": "metadata_storage_container",
                        "source": f"='{self._blob_options.container_name}'",
                    },
                ],
                "outputs": [
                    {"name": "opco_values_array", "targetName": "opco_values_array"},
                    {"name": "persona_values_array", "targetName": "persona_values_array"},
                ],
                "httpHeaders": {},
            },
            {
                "@odata.type": "#Microsoft.Skills.Custom.WebApiSkill",
                "name": "image-verbalization-skill",
                "description": "Generate text descriptions of images using GPT vision (via custom WebApiSkill)",
                "context": "/document/normalized_images/*",
                "uri": "https://app-ka-sandbox-001.azurewebsites.net/api/image-verbalization",
                "httpMethod": "POST",
                "timeout": "PT3M50S",
                "batchSize": 1,
                "inputs": [
                    {"name": "systemMessage", "source": self.IMAGE_VERBALIZATION_LITERAL},
                    {"name": "userMessage", "source": "='Please describe this image.'"},
                    {"name": "image", "source": "/document/normalized_images/*/data"},
                ],
                "outputs": [
                    {"name": "response", "targetName": "verbalizedImage"}
                ],
                "httpHeaders": {},
            },
            {
                "@odata.type": "#Microsoft.Skills.Text.AzureOpenAIEmbeddingSkill",
                "name": "image-description-embedding-skill",
                "description": "Generate embeddings for image descriptions using Azure OpenAI",
                "context": "/document/normalized_images/*",
                "resourceUri": self._openai_options.resource_uri,
                "deploymentId": self._openai_options.text_embedding_model,
                "dimensions": 3072,
                "modelName": self._openai_options.text_embedding_model,
                "inputs": [
                    {"name": "text", "source": "/document/normalized_images/*/verbalizedImage"}
                ],
                "outputs": [
                    {"name": "embedding", "targetName": "verbalizedImage_vector"}
                ],
            },
            {
                "@odata.type": "#Microsoft.Skills.Util.ShaperSkill",
                "name": "image-path-shaper-skill",
                "context": "/document/normalized_images/*",
                "inputs": [
                    {"name": "normalized_images", "source": "/document/normalized_images/*"},
                    {
                        "name": "imagePath",
                        "source": f"='{self._blob_options.images_container_name}/' + $(/document/normalized_images/*/imagePath)",
                    },
                ],
                "outputs": [
                    {"name": "output", "targetName": "new_normalized_images"}
                ],
            },
        ]

        selectors: List[Dict[str, Any]] = [
            {
                "targetIndexName": index_name,
                "parentKeyFieldName": "text_document_id",
                "sourceContext": "/document/text_sections/*",
                "mappings": [
                    {"name": "content_embedding", "source": "/document/text_sections/*/text_vector"},
                    {"name": "content_text", "source": "/document/text_sections/*/content"},
                    {"name": "location_metadata", "source": "/document/text_sections/*/locationMetadata"},
                    {"name": "document_title", "source": "/document/document_title"},
                    {"name": "opco_values", "source": "/document/opco_values_array/*"},
                    {"name": "persona_values", "source": "/document/persona_values_array/*"},
                ],
            },
            {
                "targetIndexName": index_name,
                "parentKeyFieldName": "image_document_id",
                "sourceContext": "/document/normalized_images/*",
                "mappings": [
                    {"name": "content_text", "source": "/document/normalized_images/*/verbalizedImage"},
                    {"name": "content_embedding", "source": "/document/normalized_images/*/verbalizedImage_vector"},
                    {"name": "content_path", "source": "/document/normalized_images/*/new_normalized_images/imagePath"},
                    {"name": "document_title", "source": "/document/document_title"},
                    {"name": "location_metadata", "source": "/document/normalized_images/*/locationMetadata"},
                    {"name": "opco_values", "source": "/document/opco_values_array/*"},
                    {"name": "persona_values", "source": "/document/persona_values_array/*"},
                ],
            },
        ]

        index_projections: Dict[str, Any] = {
            "selectors": selectors,
            "parameters": {"projectionMode": "skipIndexingParentDocuments"},
        }

        payload: Dict[str, Any] = {
            "name": skillset_name,
            "description": "A skillset for multimodal document processing with text and image extraction",
            "cognitiveServices": cognitive_services,
            "skills": skills,
            "indexProjections": index_projections,
        }

        storage_arm_id = self._blob_options.resource_id
        if isinstance(storage_arm_id, str) and storage_arm_id.startswith("ResourceId="):
            storage_arm_id = storage_arm_id.split("ResourceId=", 1)[1]

        payload["knowledgeStore"] = {
            "storageConnectionString": f"ResourceId={storage_arm_id}",
            "projections": [
                {
                    "objects": [
                        {
                            "storageContainer": self._blob_options.images_container_name,
                            "source": "/document/normalized_images/*",
                        }
                    ]
                }
            ],
            "parameters": {"synthesizeGeneratedKeyName": True},
        }

        endpoint = str(self._search_options.endpoint).rstrip("/")
        url = f"{endpoint}/skillsets/{skillset_name}"
        params = {"api-version": self._search_options.skillset_api_version}

        headers = {
            "api-key": self._search_options.api_key,
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
            resp = await client.put(url, params=params, headers=headers, json=payload)

        if 200 <= resp.status_code < 300:
            return

        raise Exception(
            f"Error creating skillset '{skillset_name}' via REST: {resp.status_code} - {resp.text}"
        )
