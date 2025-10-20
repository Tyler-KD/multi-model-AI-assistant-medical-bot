"""
Configuration file for the Multi-Agent Medical Chatbot

This file contains all the configuration parameters for the project.
"""

import os
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI

# Load environment variables from .env file
load_dotenv()

class ModelConfig:
    def __init__(self):
        self.conversation_model = "gpt-4o"
        self.decision_model = "gpt-4o"
        self.vision_model = "gpt-4o"
        self.default_temperature = 0.1
        self.rag_temperature = 0.0
        self.conversation_temperature = 0.7
        self.confidence_threshold = 0.85
        self.medical_confidence_threshold = 0.95

class RAGConfig:
    def __init__(self):
        self.vector_db_type = "qdrant"
        self.vector_db_path = "data/vector_db"
        self.collection_name = "medical_knowledge"
        # self.embedding_model = "text-embedding-3-large"
        # Initialize Azure OpenAI Embeddings
        self.embedding_model = AzureOpenAIEmbeddings(
            deployment = os.getenv("embedding_deployment_name"),  # Replace with your Azure deployment name
            model = os.getenv("embedding_model_name"),  # Replace with your Azure model name
            azure_endpoint = os.getenv("embedding_azure_endpoint"),  # Replace with your Azure endpoint
            openai_api_key = os.getenv("embedding_openai_api_key"),  # Replace with your Azure OpenAI API key
            openai_api_version = os.getenv("embedding_openai_api_version")  # Ensure this matches your API version
        )
        self.llm = AzureChatOpenAI(
            deployment_name = os.getenv("deployment_name"),  # Replace with your Azure deployment name
            model_name = os.getenv("model_name"),  # Replace with your Azure model name
            azure_endpoint = os.getenv("azure_endpoint"),  # Replace with your Azure endpoint
            openai_api_key = os.getenv("openai_api_key"),  # Replace with your Azure OpenAI API key
            openai_api_version = os.getenv("openai_api_version"),  # Ensure this matches your API version
            temperature=0.3  # Slightly creative but factual
        )
        self.top_k = 5
        self.similarity_threshold = 0.75
        self.huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
        self.embedding_dim = 1536  # Add the embedding dimension here
        self.distance_metric = "Cosine"  # Add this with a default value
        self.use_local = True  # Add this with a default value
        self.local_path = "./data/qdrant_db"  # Add this with a default value
        self.url = os.getenv("QDRANT_URL")
        self.api_key = os.getenv("QDRANT_API_KEY")
        self.collection_name = "medical_assistance_rag"  # Ensure a valid name
        self.chunk_size = 512  # Set a default value
        self.chunk_overlap = 50  # If you use overlap, set it too
        self.processed_docs_dir = "./data/processed"  # Set a default value

        self.reranker_model = "cross-encoder/ms-marco-TinyBERT-L-6"
        self.reranker_top_k = 5

        self.max_context_length = 1024  # ADD THIS LINE (Change based on your need)
        self.response_format_instructions = "Provide a well-structured response based on retrieved knowledge."  # ADD THIS LINE
        self.include_sources = True  # ADD THIS LINE
        self.metrics_save_path = "./logs/rag_metrics.json"  # ADD THIS LINE

        self.min_retrieval_confidence = 0.4

class MedicalCVConfig:
    def __init__(self):
        self.brain_tumor_model_path = "models/brain_tumor_segmentation.onnx"
        self.chest_xray_model_path = "models/chest_xray_detection.onnx"
        self.skin_lesion_model_path = "models/skin_lesion_classification.onnx"
        self.brain_image_size = (240, 240, 155)
        self.chest_xray_image_size = (512, 512)
        self.skin_lesion_image_size = (224, 224)
        self.llm = AzureChatOpenAI(
            deployment_name = os.getenv("deployment_name"),  # Replace with your Azure deployment name
            model_name = os.getenv("model_name"),  # Replace with your Azure model name
            azure_endpoint = os.getenv("azure_endpoint"),  # Replace with your Azure endpoint
            openai_api_key = os.getenv("openai_api_key"),  # Replace with your Azure OpenAI API key
            openai_api_version = os.getenv("openai_api_version")  # Ensure this matches your API version
        )

class APIConfig:
    def __init__(self):
        self.host = "0.0.0.0"
        self.port = 8000
        self.debug = True
        self.rate_limit = 10
        self.max_image_upload_size = 10

class SpeechConfig:
    def __init__(self):
        self.tts_voice_id = "EXAVITQu4vr4xnSDxMaL"
        self.tts_stability = 0.5
        self.tts_similarity_boost = 0.8
        self.stt_model = "whisper-1"
        self.stt_language = "en"

class ValidationConfig:
    def __init__(self):
        self.require_validation = {
            "CONVERSATION_AGENT": False,
            "RAG_AGENT": False,
            "WEB_SEARCH_AGENT": False,
            "BRAIN_TUMOR_AGENT": True,
            "CHEST_XRAY_AGENT": True,
            "SKIN_LESION_AGENT": True
        }
        self.validation_timeout = 300
        self.default_action = "reject"

class UIConfig:
    def __init__(self):
        self.theme = "light"
        self.max_chat_history = 50
        self.enable_speech = True
        self.enable_image_upload = True

class Config:
    def __init__(self):
        self.model = ModelConfig()
        self.rag = RAGConfig()
        self.medical_cv = MedicalCVConfig()
        self.api = APIConfig()
        self.speech = SpeechConfig()
        self.validation = ValidationConfig()
        self.ui = UIConfig()
        self.eleven_labs_api_key = os.getenv("ELEVEN_LABS_API_KEY")
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        self.max_conversation_history = 10

# # Example usage
# config = Config()