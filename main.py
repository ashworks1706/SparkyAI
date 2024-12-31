# %% [markdown]
# # Sparky Discord Bot: AI-Powered ASU Information Assistant
# 
# ## Key Features
# - **Intelligent Q&A**: Powered by Vertex AI (Gemini) for answering ASU-related student queries
# - **Advanced Search**: Integrates Hugging Face models, Qdrant vector database, and multiple retrieval methods
# - **Multi-step Reasoning**: Supports complex information retrieval across multiple sources
# - **Dynamic User Interaction**: Provides context-aware responses with citation tracking
# - **Optimized Retrieval**: Implements RAPTOR, MIPS, and ScaNN for efficient information retrieval
# - **Cross-Encoder Reranking**: Enhances result relevance through advanced reranking techniques
# 
# 

# %% [markdown]
# ## Description
# 
# This code implements a sophisticated Discord bot designed to assist Arizona State University (ASU) students. The bot utilizes Vertex AI (Gemini) agents for intelligent query processing and leverages advanced technologies like Retrieval-Augmented Generation (RAG) for accurate information retrieval. It integrates with Hugging Face Hub for pre-trained models and embeddings, and uses Qdrant vector database for efficient semantic search.
# 
# The bot supports multi-step reasoning, allowing it to answer complex queries by synthesizing information from multiple sources. It implements various features such as web scraping, AI-driven summarization, and dynamic access to ASU resources like clubs, events, and scholarships. The code also includes user verification mechanisms, custom Discord commands, and extensive error handling.
# 
# Advanced retrieval methods such as RAPTOR (Retrieval Augmented Prompt Tree Optimization and Refinement), Maximum Inner Product Search (MIPS), and ScaNN (Scalable Nearest Neighbors) are implemented to enhance search efficiency and accuracy. Cross-encoder reranking further improves the relevance of retrieved results.
# 
# Additionally, it incorporates analytics capabilities for tracking bot interactions and user activity, making it a comprehensive solution for ASU-related inquiries within a Discord environment[1].
# 

# %% [markdown]
# ## Components
# - **AI Agent**: Gemini-powered intelligent response generation
# - **Vector Database**: Qdrant for efficient semantic search
# - **Embedding Model**: Hugging Face transformers for text representation
# - **Discord Integration**: Real-time interaction and information retrieval
# - **Retrieval Methods**: RAPTOR, MIPS, and ScaNN for optimized search
# - **Reranking**: Cross-encoder model for improved result relevance
# 

# %% [markdown]
# ## Downloading dependencies

# %%
# # Upgrade pip to ensure we have the latest version for package management
# %pip install --upgrade pip

# # AI/ML packages
# # For using pre-tained NLP models and pipelines
# %pip install transformers -U  

# # Google's Generative AI libraries
# # type 1 - Official Google Generatie AI library
# %pip install google-generativeai
# # type 2 - Alternative Google Genrative AI library
# %pip install google-genai

# # LangChain and related packages for building applications with LLMs
# %pip install langchain
# # Integration with Huggng Face models
# %pip install langchain-huggingface  
# # For vector store integration
# %pip install langchain-qdrant  
# # Access to Hugging Face model hub
# %pip install huggingface_hub  
# # Community extensions for Langhain
# %pip install -U langchain-community  

# # Database and vector stores for efficient similarity search
# # Qdrant vector database client
# %pip install qdrant-client  
# %pip install scikit-learn numpy

# # ChromaDB for document storae and retrieval
# %pip install chromadb  
# # Facebook AI SimilaritySearch (GPU version)
# %pip install faiss-gpu  

# # Web and utilities
# # For Discord bot fuctionality
# %pip install discord.py
# # For making HTTP requess  
# %pip install requests  
# # For web scraping
# %pip install beautiulsoup4
#  # Asynchronous HTTP client/erver  
# %pip install aiohttp 
# # For retrying operatins
# %pip install tenacity 
# # Interactive widgets fr Jupyter notebooks
# %pip install ipywidgets 
# # For unit testing
# %pip install pytest 
# # For asynchronous proramming
# %pip install asyncio  
# # To allow asyncio to wrk in Jupyter notebooks
# %pip install nest_asyncio  
# # For web browser automation
# %pip install selenium 
# # For managing SeleniumWebDriver
# %pip install webdriver-manager  
# %pip install urllib
# # Cryptography libray, required for voice in Discord
# %pip install PyNaCl 
# # For ASCII transliteations of Unicode text
# %pip install unidecode  
# # For converting HTML to lain text
# %pip install html2text  
# # Google Cloud AI Platfor
# %pip install --upgrade googe-cloud-aiplatform  
# %pip install sentence-transformers hnswlib 


# # Google API related packages
# %pip install google-auth-oauthib google-auth-httplib2 google-api-python-client
# %pip install google-cloud-firestore
# %pip install firebase_admin


# # Natural Language Toolkit for text processing
# %pip install nltk


# %% [markdown]
# ## Importing Libraries
# 

# %%
# Standard library imports
import os  # For operating system related operations
import json  # For JSON data handling
import time  # For time-related functions
import random  # For generating random numbers
import uuid  # For generating unique identifiers
import re  # For regular expressions
import smtplib  # For sending emails
import asyncio  # For asynchronous programming
import traceback  # For exception handling and stack traces
import concurrent.futures  # For parallel execution of tasks
import tracemalloc  # For tracking memory allocations
from datetime import datetime  # For date and time operations
from email.mime.text import MIMEText  # For creating email messages
from typing import Dict, Any, Callable, Optional, List, Union, Tuple  # For type hinting
from dataclasses import dataclass  # For creating data classes
from urllib.parse import quote_plus, urlparse, parse_qs  # For URL parsing and encoding
import urllib.parse


# Third-party library imports
import requests  # For making HTTP requests
import aiohttp  # For asynchronous HTTP requests
import discord  # For Discord bot functionality
from discord import app_commands
from discord.ui import Modal, TextInput
import nest_asyncio  # For nested asyncio support
import nltk  # Natural Language Toolkit
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup  # For web scraping
from unidecode import unidecode  # For Unicode to ASCII conversion

# Lang Chain Imports
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_qdrant import  QdrantVectorStore 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import ScaNN
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams, OptimizersConfigDiff
from huggingface_hub import login
import hnswlib
from sentence_transformers import CrossEncoder
import numpy as np
from sklearn.cluster import KMeans
from qdrant_client.http.models import Filter, FieldCondition, MatchAny

## Google APIs
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import Flow
import firebase_admin
from firebase_admin import credentials, firestore


# Selenium Imports
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import ElementClickInterceptedException, TimeoutException

# Webscrape Utilities
from html2markdown import convert
import html2text

# Logging
import logging
from itertools import cycle

# Google AI imports
import google.generativeai as genai
from google.ai.generativelanguage_v1beta.types import content
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from google import genai as genai2
from google.genai import types
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch




# %% [markdown]
# ## Setting up Environment 
# 
# This AppConfig class is the heart of our configuration management. It's designed to load and provide easy access to various configuration settings that our bot needs. We're using a JSON file (appConfig.json by default) to store these settings, which makes it easy to modify without changing the code.
# 

# %%
class AppConfig:
    def __init__(self, config_file='appConfig.json'):
        with open(config_file, 'r') as file:
            config_data = json.load(file)
        
        os.environ['NUMEXPR_MAX_THREADS'] = config_data.get('NUMEXPR_MAX_THREADS', '16')
        os.environ['HUGGINGFACEHUB_API_TOKEN'] = config_data.get('HUGGINGFACEHUB_API_TOKEN', '')
        self._api_key = config_data.get('API_KEY', '')
        self.handshake_user = config_data.get('HANDSHAKE_USER', '')
        self.handshake_pass = config_data.get('HANDSHAKE_PASS', '')
        self.gmail = config_data.get('GMAIL', '')
        self.gmail_pass = config_data.get('GMAIL_PASS', '')
        self.spreadsheet_id = config_data.get('SPREADSHEET_ID', '')
        
        self.main_agent_prompt = config_data.get('MAIN_AGENT_PROMPT', '')
        
        self.live_status_agent_prompt = config_data.get('LIVE_STATUS_AGENT_PROMPT', '')
        self.live_status_agent_instruction = config_data.get('LIVE_STATUS_AGENT_INSTRUCTION', '')
        
        self.discord_agent_prompt = config_data.get('DISCORD_AGENT_PROMPT', '')
        self.discord_agent_instruction = config_data.get('DISCORD_AGENT_INSTRUCTION', '')
        
        self.search_agent_prompt = config_data.get('SEARCH_AGENT_PROMPT', '')
        self.search_agent_instruction = config_data.get('SEARCH_AGENT_INSTRUCTION', '')
        
        self.action_agent_prompt = config_data.get('ACTION_AGENT_PROMPT', '')
        self.action_agent_instruction = config_data.get('ACTION_AGENT_INSTRUCTION', '')
        
        
        self.search_agent_prompt = config_data.get('SEARCH_AGENT_PROMPT', '')
        self.search_agent_instruction = config_data.get('SEARCH_AGENT_INSTRUCTION', '')
        
        self.discord_agent_prompt = config_data.get('DISCORD_AGENT_PROMPT', '')
        self.discord_agent_instruction = config_data.get('DISCORD_AGENT_INSTRUCTION', '')
        
        self.google_agent_prompt = config_data.get('GOOGLE_AGENT_PROMPT', '')
        self.google_agent_instruction = config_data.get('GOOGLE_AGENT_INSTRUCTION', '')
        self.data_agent_prompt = config_data.get('DATA_AGENT_PROMPT', '')
        self.discord_bot_token = config_data.get('DISCORD_BOT_TOKEN', '')
        self.kubernetes_api_key = config_data.get('KUBERNETES_SECRET', '')
        self.qdrant_api_key = config_data.get('QDRANT_API_KEY', '')

    def get_numexpr_max_threads(self):
        return os.environ['NUMEXPR_MAX_THREADS']

    def get_huggingfacehub_api_token(self):
        return os.environ['HUGGINGFACEHUB_API_TOKEN']

    def get_qdrant_api_key(self):
        return self.qdrant_api_key
    
    def get_discord_bot_token(self):
        return self.discord_bot_token

    def get_kubernetes_api_key(self):
        return self.kubernetes_api_key
    
    def get_data_agent_prompt(self):
        return self.data_agent_prompt
    
    
    def get_live_status_agent_prompt(self):
        return self.live_status_agent_prompt
    def get_live_status_agent_instruction(self):
        return self.live_status_agent_instruction
    
    def get_discord_agent_prompt(self):
        return self.discord_agent_prompt
    def get_discord_agent_instruction(self):
        return self.discord_agent_instruction
    
    def get_search_agent_prompt(self):
        return self.search_agent_prompt
    def get_search_agent_instruction(self):
        return self.search_agent_instruction
    
    
    def get_action_agent_prompt(self):
        return self.action_agent_prompt
    def get_action_agent_instruction(self):
        return self.action_agent_instruction
    
    def get_google_agent_prompt(self):
        return self.google_agent_prompt
    def get_google_agent_instruction(self):
        return self.google_agent_instruction
    
    def get_api_key(self):
        return self._api_key
    def get_handshake_user(self):
        return self.handshake_user
    def get_handshake_pass(self):
        return self.handshake_pass
    def get_gmail(self):
        return self.gmail
    def get_spreadsheet_id(self):
        return self.spreadsheet_id
    def get_gmail_pass(self):
        return self.gmail_pass


# %% [markdown]
# ```python
#         self.action_agent_prompt = config_data.get('ACTION_AGENT_PROMPT', '')
#         self.action_agent_instruction = config_data.get('ACTION_AGENT_INSTRUCTION', '')
#         # ... (similar lines for other agent prompts and instructions)
# ```
# These are the prompts and instructions for our various AI agents. Each agent (main, live status, discord, search, action, deep search, google, data) has its own prompt and sometimes an instruction. This modular approach allows us to fine-tune the behavior of each agent independently.
# 
# ```python
#     def get_numexpr_max_threads(self):
#         return os.environ['NUMEXPR_MAX_THREADS']
# 
#     def get_huggingfacehub_api_token(self):
#         return os.environ['HUGGINGFACEHUB_API_TOKEN']
# 
#     # ... (other getter methods)
# ```
# 
# These getter methods provide a clean interface to access our configuration values. They're particularly useful for values that might change during runtime or need some processing before being used.

# %% [markdown]
# 
# ```python
#         os.environ['NUMEXPR_MAX_THREADS'] = config_data.get('NUMEXPR_MAX_THREADS', '16')
#         os.environ['HUGGINGFACEHUB_API_TOKEN'] = config_data.get('HUGGINGFACEHUB_API_TOKEN', '')
# ```
# 
# Here, we're setting some environment variables. The `NUMEXPR_MAX_THREADS` is used to control parallel processing, while `HUGGINGFACEHUB_API_TOKEN` is crucial for accessing Hugging Face's models and services.
# 
# ```python
#         self._api_key = config_data.get('API_KEY', '')
#         self.handshake_user = config_data.get('HANDSHAKE_USER', '')
#         self.handshake_pass = config_data.get('HANDSHAKE_PASS', '')
#         self.gmail = config_data.get('GMAIL', '')
#         self.gmail_pass = config_data.get('GMAIL_PASS', '')
#         self.spreadsheet_id = config_data.get('SPREADSHEET_ID', '')
# ```
# 
# These lines are loading various API keys and credentials. We're using them for different services like email notifications and Google Sheets integration. The `get` method with default empty strings ensures our code doesn't crash if a key is missing.
# 
# 
# 
# 
# 
# 
# Here, we're creating a global instance of our `AppConfig` class. This allows us to access our configuration from anywhere in the code simply by importing `app_config`.
# 

# %%
app_config = AppConfig()

# %% [markdown]
# Finally, we set up logging. This is crucial for debugging and monitoring our bot's behavior. We're logging to both a file and the console, which helps in development and production environments. The `tracemalloc.start()` line enables memory allocation tracking, which can be super helpful for optimizing our bot's performance.
# 
# This configuration setup allows us to easily manage and update various settings, API keys, and agent behaviors without diving into the core code. It's a crucial part of making our bot flexible and maintainable.

# %%
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_processor.log'),
        logging.StreamHandler()
    ]
)
tracemalloc.start()
logger = logging.getLogger(__name__)



# %% [markdown]
# #### Discord State Variables

# %% [markdown]
# This setup allows us to maintain a consistent state for our Discord bot and provides easy access to our AI model throughout the application. It's designed to be flexible and easy to manage, which is crucial for a complex bot like ours that interacts with both Discord and AI services.
# 
# ```python
# class DiscordState:
#     def __init__(self):
#         nest_asyncio.apply()
#         self.intents = discord.Intents.default()
#         self.intents.message_content = True
#         self.intents.members = True
#         # ... (other attribute initializations)
# ```
# 
# The `DiscordState` class is crucial for managing the state of our Discord bot. Here's what's happening:
# 
# - We apply `nest_asyncio` to allow nested event loops, which can be necessary in some environments.
# - We set up Discord intents, which define what events our bot can receive. We're enabling message content and member information access.
# - We initialize various attributes to track the bot's state, such as the current user, guild, and voice channel information.
# 
# ```python
#     def update(self, **kwargs):
#         for key, value in kwargs.items():
#             if hasattr(self, key):
#                 setattr(self, key, value)
#             else:
#                 raise AttributeError(f"DiscordState has no attribute '{key}'")
# ```
# 
# This `update` method allows us to update multiple attributes of our Discord state at once. It's a convenient way to keep our state in sync with the current Discord context.
# 
# 
# 
# 

# %%


class DiscordState:
    def __init__(self):
        nest_asyncio.apply()
        self.intents = discord.Intents.default()
        self.intents.message_content = True
        self.intents.members = True
        self.user = False
        self.target_guild = None
        self.user_id = None
        self.user_has_mod_role = False
        self.user_in_voice_channel = False
        self.request_in_dm = False
        self.guild_user= None
        self.user_voice_channel_id = None
        self.discord_client = discord.Client(intents=self.intents)
        self.task_message = None

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"DiscordState has no attribute '{key}'")

    def get(self, attr):
        if hasattr(self, attr):
            return getattr(self, attr)
        else:
            raise AttributeError(f"DiscordState has no attribute '{attr}'")

    def __str__(self):
        return "\n".join([f"{attr}: {getattr(self, attr)}" for attr in vars(self) if not attr.startswith('__')])


# %% [markdown]
# ```python
#     def get(self, attr):
#         if hasattr(self, attr):
#             return getattr(self, attr)
#         else:
#             raise AttributeError(f"DiscordState has no attribute '{attr}'")
# ```
# 
# The `get` method provides a safe way to retrieve attribute values. It raises an error if we try to access a non-existent attribute, which helps catch bugs early.
# 
# ```python
#     def __str__(self):
#         return "\n".join([f"{attr}: {getattr(self, attr)}" for attr in vars(self) if not attr.startswith('__')])
# ```
# 
# This `__str__` method gives us a nice string representation of our Discord state, which is super helpful for debugging and logging.

# %% [markdown]
# 
# 
# ```python
# discord_state = DiscordState()
# ```
# 
# Here, we're creating a global instance of our `DiscordState`. This allows us to access and update the Discord state from anywhere in our code.

# %%
discord_state = DiscordState()

# %% [markdown]
# #### Model and Database Global Variables

# %% [markdown]
# 
# Now, let's look at the model initialization:
# 
# ```python
# genai.configure(api_key=app_config.get_api_key())
# dir_Model = genai.GenerativeModel('gemini-1.5-flash')
# ```
# 1. We're configuring the Google Generative AI (genai) with our API key, which we fetch from our AppConfig.
# 2. We're initializing our main AI model, specifically the 'gemini-1.5-flash' model. This is a powerful model that we'll use for generating responses and processing queries.
# 3. Finally, we log a success message to confirm that our global variables for the model and database (if applicable) have been initialized correctly.
# 

# %%

genai.configure(api_key=app_config.get_api_key())
dir_Model = genai.GenerativeModel('gemini-1.5-flash')

logger.info("\nSuccessfully initialized global variables for model and database")

# %% [markdown]
# ## Qdrant Vector Storage
# 
# 
# This class is designed for efficient vector storage and retrieval, with features like automatic version control, duplicate handling, and optimized semantic search capabilities. It's built to be robust and performant, suitable for large-scale document processing and storage operations. The addition of MIPS search and HNSWlib integration further enhances its search capabilities and performance for real-time applications.

# %% [markdown]
# ### Initialization
# - The `__init__` method sets up the vector store with configurable parameters like host, port, collection name, and embedding model.
# - It initializes connections to Qdrant and sets up the embedding model.
# - Error handling and logging are implemented for initialization failures.
# 
# ### Embedding Model
# - The `_initialize_embedding_model` method sets up a HuggingFace embedding model (default: "BAAI/bge-large-en-v1.5").
# - It determines the vector size based on a test embedding.
# 
# ### Qdrant Client
# - `_create_qdrant_client` establishes a connection to the Qdrant server with retry logic.
# - It attempts to connect multiple times before raising an error.
# 
# ### Collection Management
# - `_setup_collection` checks for existing collections and creates a new one if needed.
# - `_verify_collection_dimensions` ensures the existing collection matches the current model's vector size.
# - `_create_collection` sets up a new Qdrant collection with specified parameters.
# 
# ### Document Storage
# - `store_to_vector_db` is an asynchronous method for storing documents in batches.
# - It includes logic for skipping duplicates and handling errors during storage.
# 
# ### Document Processing
# - `_should_store_document` checks if a document should be stored based on existing data and timestamps.
# - It implements version control by replacing outdated documents.
# 
# ### Vector Store Initialization
# - `_initialize_vector_store` sets up the QdrantVectorStore with the configured client and embedding model.
# 
# ### Error Handling and Logging
# - Comprehensive error logging is implemented throughout the class.
# - `_log_detailed_error` provides in-depth error diagnostics.
# 
# ### MIPS Search
# - The `mips_search` method performs Maximum Inner Product Search on the vector database.
# - It returns formatted results with metadata and similarity scores.
# 
# ### HNSWlib Integration
# - The `build_hnsw_index` method constructs an HNSW index for efficient similarity search.
# - It uses all documents in the collection to build the index.
# 
# ### Document Retrieval
# - The `get_all_documents` method retrieves all documents from the Qdrant collection.
# - The `get_embeddings` method generates embeddings for given documents
# 

# %%



class VectorStore:
    
    """A class to manage vector storage operations using Qdrant with enhanced logging and performance."""
    
    def __init__(self, 
                 force_recreate: bool = False,
                 host: str = "10.10.0.9",
                 port: int = 6333,
                 collection_name: str = "asu_docs",
                 model_name: str = "BAAI/bge-large-en-v1.5",
                 batch_size: int = 100,
                 max_retry_attempts: int = 3,
                 retry_delay: int = 2):
        """
        Initialize the VectorStore with specified parameters and enhanced error handling.
        
        Args:
            force_recreate (bool): Whether to recreate the collection if it exists
            host (str): Qdrant server host
            port (int): Qdrant server port
            collection_name (str): Name of the collection
            model_name (str): Name of the embedding model
            batch_size (int): Size of batches for document processing
            max_retry_attempts (int): Maximum number of retry attempts for operations
            retry_delay (int): Delay between retry attempts in seconds
        """
        self.vector_store: Optional[QdrantVectorStore] = None
        self.collection_name = collection_name
        self.batch_size = batch_size
        self.max_retry_attempts = max_retry_attempts
        self.retry_delay = retry_delay
        self.corpus = []
        self.hnsw_index = None


        
        logger.info(f"Initializing VectorStore with collection: {collection_name}")
        logger.info(f"Configuration: host={host}, port={port}, model={model_name}")
        
        try:
            self.client = self._create_qdrant_client(host, port)
            self._initialize_embedding_model(model_name)
            self._setup_collection(force_recreate)
            self._initialize_vector_store()
            
        except Exception as e:
            logger.error(f"Critical VectorStore initialization error: {str(e)}", exc_info=True)
            self._log_detailed_error(e)
            raise RuntimeError(f"VectorStore initialization failed: {str(e)}")
    
    def mips_search(self, query_vector: List[float], top_k: int = 5):
        try:
            if not self.vector_store:
                logger.error("Vector store not initialized.")
                raise ValueError("Vector store not properly initialized.")
            
            if self.hnsw_index is None:
                self.build_hnsw_index()
            
            labels, distances = self.hnsw_index.knn_query(query_vector, k=top_k)
            results = []
            for label, distance in zip(labels[0], distances[0]):
                doc = self.get_document_by_id(int(label))
                results.append({
                    "id": doc.metadata.get('id'),
                    "score": 1 - distance,  # Convert distance to similarity score
                    "payload": {
                        "page_content": doc.page_content,
                        "metadata": doc.metadata
                    }
                })
            
            logger.info(f"MIPS search retrieved {len(results)} results.")
            return results
        except Exception as e:
            logger.error(f"Error during MIPS search: {str(e)}")
            return []
        
    def get_all_documents(self):
        # Implement method to retrieve all documents from Qdrant
        results = self.client.scroll(
            collection_name=self.collection_name,
            limit=10000  # Adjust as needed
        )
        return [Document(page_content=item.payload["page_content"], metadata=item.payload["metadata"]) for item in results[0]]

    def get_embeddings(self, documents):
        return [self.embedding_model.embed_query(self.get_document_content(doc)) for doc in documents]

    def get_document_content(self, doc):
        if isinstance(doc, str):
            try:
                doc_dict = json.loads(doc)
                return doc_dict.get('page_content', '')
            except json.JSONDecodeError:
                return doc
        else:
            return getattr(doc, 'page_content', str(doc))
        
    def build_hnsw_index(self):
        all_docs = self.get_all_documents()
        all_embeddings = self.get_embeddings(all_docs)
        
        self.hnsw_index = hnswlib.Index(space='cosine', dim=self.vector_size)
        self.hnsw_index.init_index(max_elements=len(all_docs), ef_construction=200, M=16)
        self.hnsw_index.add_items(all_embeddings, np.arange(len(all_docs)))
        self.hnsw_index.set_ef(50)  # Adjust for speed/accuracy trade-off

    def _initialize_embedding_model(self, model_name: str) -> None:
        """Initialize the embedding model."""
        logger.info(f"Initializing embedding model: {model_name}")
        try:
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': 'cpu'}
            )
            self.vector_size = len(self.embedding_model.embed_query("test"))
            logger.info(f"Embedding model initialized with vector size: {self.vector_size}")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {str(e)}", exc_info=True)
            raise
    
    def _create_qdrant_client(self, host: str, port: int) -> QdrantClient:
        for attempt in range(self.max_retry_attempts):
            try:
                api_key = app_config.get_qdrant_api_key()  # Use the Kubernetes API key for authentication
                client = QdrantClient(
                url="https://4bfefa3a-9337-4325-9836-5f054c1de8d8.us-east-1-0.aws.cloud.qdrant.io",
                api_key=api_key,
                prefer_grpc=False
                )

                logger.info(f"Successfully connected to Qdrant at {host}:{port} (Attempt {attempt + 1})")
                return client
            except Exception as e:
                if attempt == self.max_retry_attempts - 1:
                    logger.error(f"Failed to connect to Qdrant after {self.max_retry_attempts} attempts")
                    raise
                logger.warning(f"Qdrant connection attempt {attempt + 1} failed: {str(e)}")
                time.sleep(self.retry_delay)
    
    def _verify_collection_dimensions(self) -> None:
        """Verify that existing collection dimensions match the model."""
        collection_info = self.client.get_collection(self.collection_name)
        existing_size = collection_info.config.params.vectors.size
        
        if existing_size != self.vector_size:
            error_msg = (f"Dimension mismatch: Collection has {existing_size}, "
                        f"model requires {self.vector_size}")
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info(f"Verified collection dimensions: {existing_size}")
    
    def _log_detailed_error(self, exception: Exception) -> None:
        """
        Log detailed error information for diagnostics.
        
        Args:
            exception (Exception): The exception to log details for
        """
        logger.error("Detailed Error Diagnostics:")
        logger.error(f"Error Type: {type(exception).__name__}")
        logger.error(f"Error Message: {str(exception)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
    
    def _setup_collection(self, force_recreate: bool) -> None:
        """Set up the Qdrant collection."""
        try:
            collections = self.client.get_collections().collections
            collection_exists = any(c.name == self.collection_name for c in collections)
            
            if collection_exists:
                if force_recreate:
                    logger.info(f"Force recreating collection: {self.collection_name}")
                    self.client.delete_collection(self.collection_name)
                    collection_exists = False
                else:
                    self._verify_collection_dimensions()
            
            if not collection_exists:
                self._create_collection()
        except Exception as e:
            logger.error(f"Failed to setup collection: {str(e)}", exc_info=True)
    
    def _create_collection(self) -> None:
        """Create a new Qdrant collection."""
        logger.info(f"Creating new collection: {self.collection_name}")
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_size,
                distance=Distance.COSINE
            ),
            optimizers_config=OptimizersConfigDiff(
                default_segment_number=2,
                memmap_threshold=20000
            )
        )
        logger.info("\nCollection created successfully")
    
    def queue_documents(self, docs: List[Document]) -> None:
        """Queue documents for storage."""
        self.corpus.extend(docs)
        logger.info("Queued Processed Documents")
        return True
  
    async def store_to_vector_db(self) -> bool:
        if self.vector_store is None and self.corpus is None:
            logger.critical("Vector store not initialized - cannot proceed")
            raise ValueError("Vector store not properly initialized")

        total_docs = len(self.corpus)
        logger.info(f"Document storage initiated: {total_docs} documents to process")
        performance_start = time.time()
        processed_count = 0
        skipped_count = 0
        error_count = 0
        try:
            for doc in self.corpus:
                logger.debug(f"Processing document {processed_count + 1}/{total_docs}")
                try:
                    should_store = self._should_store_document(doc)
                    if should_store:
                        await self.vector_store.aadd_documents([doc])
                        processed_count += 1
                        logger.info(f"Successfully stored document {processed_count}")
                    else:
                        skipped_count += 1
                except Exception as e:
                    logger.error(f"Error processing document {doc.metadata.get('url', 'Unknown')}: {str(e)}")
                    error_count += 1

            performance_end = time.time()
            logger.info(f"Total Documents: {total_docs}")
            logger.info(f"Processed Documents: {processed_count}")
            logger.info(f"Skipped Documents: {skipped_count}")
            logger.info(f"Error Documents: {error_count}")
            logger.info(f"Total Processing Time: {performance_end - performance_start:.2f} seconds")
            return True
        except Exception as e:
            logger.error(f"Catastrophic document storage failure: {str(e)}", exc_info=True)
            self._log_detailed_error(e)
            raise

    def _initialize_vector_store(self) -> None:
        """Initialize the QdrantVectorStore."""
        logger.info("\nInitializing QdrantVectorStore")
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embedding_model,
            content_payload_key="page_content",
            metadata_payload_key="metadata",
            distance=Distance.COSINE
        )
        
    
    def get_vector_store(self):
        return self.vector_store

    def _should_store_document(self, doc: Document) -> bool:
        try:
            urls = doc.metadata['url'] if isinstance(doc.metadata['url'], list) else [doc.metadata['url']]
            existing_docs = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.url", 
                            match=models.MatchAny(any=urls)
                        )
                    ]
                )
            )[0]
            if existing_docs:
                logger.info("Found existing Docs\n")
                logger.info(existing_docs)
                new_timestamp = doc.metadata.get('timestamp')                
                for existing_doc in existing_docs:
                    existing_timestamp = existing_doc.payload.get('metadata', {}).get('timestamp')
                    # Ensure both timestamps are datetime OBJECTs
                    if isinstance(new_timestamp, str):
                        new_timestamp = datetime.fromisoformat(new_timestamp)
                    if isinstance(existing_timestamp, str):
                        existing_timestamp = datetime.fromisoformat(existing_timestamp)
                    # Enhanced timestamp comparison with configurable threshold
                    if (not existing_timestamp or 
                        (new_timestamp and 
                        (new_timestamp - existing_timestamp).total_seconds() >= 60 * 60)):
                        # Delete outdated document
                        self.client.delete(
                            collection_name=self.collection_name,
                            points_selector=models.PointIdsList(points=[existing_doc.id])
                        )
                        logger.info(f"Replaced document: {urls} due to significant changes")
                    
                    logger.debug(f"Skipping document with minimal time difference: {urls}")
            return True
        except Exception as e:
            logger.error(f"Document evaluation error: {str(e)}")
            raise
    
    

# %%
asu_store = VectorStore(force_recreate=False)


# %% [markdown]
# ## Raptor Cluster Implementation
# 
# The `RaptorRetriever` class implements a hierarchical clustering approach for efficient document retrieval. Here's a detailed explanation of its implementation:

# %% [markdown]
# 
# 
# ### Initialization
# ```python
# def __init__(self, vector_store, num_levels=3, branching_factor=5):
# ```
# - Initializes with a vector store, number of hierarchical levels, and branching factor.
# - Builds the RAPTOR tree upon initialization.
# 
# ### Building the RAPTOR Tree
# ```python
# def build_raptor_tree(self):
# ```
# - Creates a hierarchical tree structure for efficient retrieval.
# - Uses K-means clustering at each level to group documents.
# - Generates summaries for each cluster.
# 
# Key steps:
# 1. Retrieves all documents and their embeddings from the vector store.
# 2. For each level:
#    - Performs K-means clustering with `branching_factor^(level+1)` clusters.
#    - Groups documents into clusters based on K-means labels.
#    - Generates summaries for each cluster.
#    - Stores cluster information and summaries in the tree.
#    - Uses cluster summaries as documents for the next level.
# 
# ### Summary Generation
# ```python
# def generate_summary(self, documents):
# ```
# - Placeholder for summary generation logic.
# - Currently concatenates the first 50 characters of each document.
# 
# ### Document Retrieval
# ```python
# def retrieve(self, query, top_k=5):
# ```
# - Implements the RAPTOR retrieval algorithm.
# - Traverses the tree from top to bottom, selecting the best cluster at each level.
# - At the lowest level, performs a similarity search within the best cluster.
# - Applies reranking to the initial results.
# 
# Key steps:
# 1. Embeds the query.
# 2. Starts at the top level of the tree.
# 3. At each level, selects the best cluster based on cosine similarity with summaries.
# 4. At the lowest level, performs a similarity search within the selected cluster.
# 5. Reranks the initial results using a cross-encoder.
# 
# ### Result Reranking
# ```python
# def rerank_results(self, query, initial_results, top_k=5):
# ```
# - Uses a cross-encoder model to rerank the initial retrieval results.
# - Improves the relevance of the top results.
# 
# This implementation combines hierarchical clustering for efficient search space reduction with cross-encoder reranking for improved result relevance, making it suitable for large-scale document retrieval tasks.
# 

# %%


class RaptorRetriever:
    def __init__(self, vector_store=None, num_levels=3, branching_factor=5):
        self.vector_store = asu_store
        self.num_levels = num_levels
        self.branching_factor = branching_factor
        self.tree = self.build_raptor_tree()
        logger.info("RAPTOR Retriever initialized.")

    def build_raptor_tree(self):
        tree = {}
        logger.info("Building RAPTOR tree...")
        all_docs = asu_store.get_all_documents()
        logger.info(f"Retrieved {len(all_docs)} documents for tree construction")
        all_embeddings = self.vector_store.get_embeddings(all_docs)

        if not all_embeddings:
            logger.warning("No embeddings found. The vector store may be empty.")
            return tree

        all_embeddings = np.array(all_embeddings)

        for level in range(self.num_levels):
            n_clusters = min(self.branching_factor ** (level + 1), len(all_embeddings))
            kmeans = KMeans(n_clusters=n_clusters)
            cluster_labels = kmeans.fit_predict(all_embeddings)
            
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(all_docs[i])
            
            summaries = {label: self.generate_summary(docs) for label, docs in clusters.items()}
            logger.info(f"Generated summaries for level {level + 1}")
            tree[f"level_{level}"] = {
                "clusters": clusters,
                "summaries": summaries
            }
            logger.info(f"Built tree for level {level + 1}")
            
            all_docs = list(summaries.values())
            logger.info(f"Retrieved {len(all_docs)} documents for next level")
            all_embeddings = self.vector_store.get_embeddings(all_docs)
            logger.info(f"Retrieved embeddings for next level")
            
            if not all_embeddings:
                logger.warning(f"No embeddings found for level {level + 1}. Stopping tree construction.")
                break

            all_embeddings = np.array(all_embeddings)
        
        logger.info("Building RAPTOR tree completed.")
        return tree

    def generate_summary(self, documents):
        summaries = []
        for doc in documents:
            if isinstance(doc, str):
                try:
                    doc_dict = json.loads(doc)
                    content = doc_dict.get('page_content', '')
                except json.JSONDecodeError:
                    content = doc
            else:
                content = getattr(doc, 'page_content', str(doc))
            summaries.append(content[:50])
            logger.info("Generated summary for document.")
        return " ".join(summaries)[:200]

    def retrieve(self, query, top_k=5):
        query_embedding = self.vector_store.embedding_model.embed_query(query)
        current_level = self.num_levels - 1
        current_node = self.tree[f"level_{current_level}"]
        
        while current_level >= 0:
            summaries = current_node["summaries"]
            summary_embeddings = self.vector_store.get_embeddings(list(summaries.values()))
            best_cluster = max(summaries.keys(), key=lambda x: np.dot(query_embedding, summary_embeddings[x]))
             
            if current_level == 0:
                initial_results = self.vector_store.similarity_search(query, filter={"cluster": best_cluster}, k=top_k)
                return self.rerank_results(query, initial_results, top_k)
            
            current_level -= 1
            current_node = self.tree[f"level_{current_level}"]
            current_node = {k: v for k, v in current_node.items() if k in current_node["clusters"][best_cluster]}
       
        logger.info("No results found in RAPTOR tree.")
        return []

    def rerank_results(self, query, initial_results, top_k=5):
        cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        pairs = [[query, self.get_document_content(doc)] for doc in initial_results]
        scores = cross_encoder.predict(pairs)
        reranked_results = [doc for _, doc in sorted(zip(scores, initial_results), key=lambda x: x[0], reverse=True)]
        logger.info("Reranked results.")
        return reranked_results[:top_k]

    def get_document_content(self, doc):
        if isinstance(doc, str):
            try:
                doc_dict = json.loads(doc)
                logger.info("Retrieved document content.")
                return doc_dict.get('page_content', '')
            except json.JSONDecodeError:
                return doc
        else:
            return getattr(doc, 'page_content', str(doc))


# %% [markdown]
# ## Google Sheet Database
# 
# This class, `GoogleSheet`, is designed to manage google sheet database storage operations using Google API, with enhanced logging and performance features. 
# 

# %%
class UpdateTask:
    def __init__(self, user_id: str, column: str, value: Any):
        self.user_id = user_id
        self.column = column
        self.value = value

class GoogleSheet:
    def __init__(self, credentials_file: str, spreadsheet_id: str) -> None:
        self.credentials = Credentials.from_service_account_file(
            credentials_file,
            scopes=['https://www.googleapis.com/auth/spreadsheets']
        )
        self.spreadsheet_id = spreadsheet_id
        self.service = build('sheets', 'v4', credentials=self.credentials, cache_discovery=False)
        self.sheet = self.service.spreadsheets()
        self.logger = logging.getLogger(__name__)
        self.update_tasks: List[UpdateTask] = []
        self.user_row_cache: Dict[str, int] = {}

    async def get_all_users(self, range_name: str = 'SparkyVerify!A:C'):
        try:
            result = self.sheet.values().get(
                spreadsheetId=self.spreadsheet_id,
                range=range_name
            ).execute()
            return result.get('values', [])
        except Exception as e:
            self.logger.error(f"Error getting all users: {str(e)}")
            return []

    async def increment_function_call(self, user_id: str, column: str):
        self.update_tasks.append(UpdateTask(user_id, column, None))

    async def update_user_column(self, user_id: str, column: str, value):
        self.update_tasks.append(UpdateTask(user_id, column, value))

    async def perform_updates(self):
        try:
            batch_update_data = []
            for task in self.update_tasks:
                row = await self.get_user_row(task.user_id)
                if row:
                    range_name = f'SparkyVerify!{task.column}{row}'
                    if task.value is None:  # Increment function call
                        current_value = await self.get_cell_value(range_name)
                        new_value = int(current_value) + 1 if current_value else 1
                    else:
                        new_value = task.value
                    batch_update_data.append({
                        'range': range_name,
                        'values': [[new_value]]
                    })
                else:
                    self.logger.warning(f"User {task.user_id} not found for updating")

            if batch_update_data:
                body = {
                    'valueInputOption': 'USER_ENTERED',
                    'data': batch_update_data
                }
                self.sheet.values().batchUpdate(
                    spreadsheetId=self.spreadsheet_id,
                    body=body
                ).execute()
                self.logger.info(f"Batch update completed for {len(batch_update_data)} tasks")
            
            self.update_tasks.clear()
        except Exception as e:
            self.logger.error(f"Error performing batch updates: {str(e)}")

    async def get_user_row(self, user_id: str):
        if user_id in self.user_row_cache:
            return self.user_row_cache[user_id]

        try:
            range_name = 'SparkyVerify!A:A'
            result = self.sheet.values().get(
                spreadsheetId=self.spreadsheet_id,
                range=range_name
            ).execute()
            rows = result.get('values', [])
            for idx, row in enumerate(rows):
                if row and row[0] == str(user_id):
                    row_number = idx + 1
                    self.user_row_cache[user_id] = row_number
                    return row_number
            return None
        except Exception as e:
            self.logger.error(f"Error finding user row: {str(e)}")
            return None

    async def get_cell_value(self, range_name: str):
        try:
            result = self.sheet.values().get(
                spreadsheetId=self.spreadsheet_id,
                range=range_name
            ).execute()
            return result.get('values', [[0]])[0][0]
        except Exception as e:
            self.logger.error(f"Error getting cell value: {str(e)}")
            return 0

    async def add_new_user(self, user, email):
        user_data = [str(user.id), user.name, email, datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        existing_row = await self.get_user_row(user.id)
        
        try:
            if existing_row:
                range_name = f'SparkyVerify!A{existing_row}:Z{existing_row}'
                body = {'values': [user_data]}
                self.sheet.values().update(
                    spreadsheetId=self.spreadsheet_id,
                    range=range_name,
                    valueInputOption='USER_ENTERED',
                    body=body
                ).execute()
                self.logger.info(f"Updated existing user: {user.id}")
            else:
                range_name = 'SparkyVerify!A:Z'
                body = {'values': [user_data]}
                self.sheet.values().append(
                    spreadsheetId=self.spreadsheet_id,
                    range=range_name,
                    valueInputOption='USER_ENTERED',
                    body=body
                ).execute()
                self.logger.info(f"Added new user: {user.id}")
            
            self.user_row_cache[user.id] = await self.get_user_row(user.id)
        except Exception as e:
            self.logger.error(f"Error adding/updating user: {str(e)}")


# %% [markdown]
# ### Initializing Global Instance

# %%
google_sheet = GoogleSheet('client_secret.json', app_config.get_spreadsheet_id())

# %% [markdown]
# ## Firestore Chat Database

# %% [markdown]
# We're using firestore to store chats from user's in different channels of servers and direct messages

# %%
class Firestore:
    def __init__(self):
        if not firebase_admin._apps:
            cred = credentials.Certificate("firebase_secret.json")
            firebase_admin.initialize_app(cred)
        self.db = firestore.client()
        self.collection = None
        self.document = {
            "action_agent_message": [],
            "discord_agent_message": [],
            "google_agent_message": [],
            "live_status_agent_message": [],
            "search_agent_message": [],
            "user_id": "",
            "user_message": "",
            "time": "",
            "category": []
        }

    def update_collection(self, collection):
        self.collection = collection
    
    def update_message(self, property, value):
        if property in self.document:
            if isinstance(self.document[property], list):
                self.document[property].append(f"{value}")
            else:
                self.document[property] = f"{value}"
        else:
            raise ValueError(f"Invalid property: {property}")
    
    async def push_message(self):
        if not self.collection:
            raise ValueError("Collection not set. Use update_collection() first.")
        
        self.document["user_id"] = discord_state.get('user_id')
        self.document["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        doc_ref = self.db.collection(self.collection).document()
        doc_ref.set(self.document)
        
        return doc_ref.id  # Return the document ID for reference


# %% [markdown]
# ### Creating Global Instance

# %%
firestore = Firestore()

# %% [markdown]
# ## Utils
# 
# This `Utils` class is designed to manage various utility functions for an AI assistant, particularly focusing on task tracking, animation handling, and search operations.

# %% [markdown]
# 
# ### Initialization
# - The `__init__` method initializes task tracking, content management, and connections to the vector store and RaptorRetriever.
# - It includes error handling and logging for initialization failures.
# 
# ### Animation and Task Management
# - `start_animation`: Initiates a loading animation using Discord's thinking indicator.
# - `update_text`: Dynamically updates the displayed text while maintaining a task history.
# - `stop_animation`: Finalizes the animation and resets the internal state.
# 
# ### Search Result Formatting
# - `format_search_results`: Converts raw search results into a readable, formatted string.
# - It handles various metadata fields and formats the content for easy reading.
# 
# ### Web Search Operations
# - `perform_web_search`: Executes a web search using a provided URL and optional query.
# - It processes the search results and stores them in the vector database.
# 
# ### Vector Store Operations
# - `perform_similarity_search`: Conducts a similarity search in the vector store.
# - It supports category-based filtering and returns formatted results.
# 
# ### MIPS Search
# - `perform_mips_search`: Performs Maximum Inner Product Search (MIPS) on the vector database.
# - It supports category-based filtering and returns formatted results.
# 
# ### Database Search
# - `perform_database_search`: Combines RAPTOR, similarity, and MIPS search methods.
# - It merges and deduplicates results from different search methods.
# - Updates cached document IDs and ground sources for future reference.
# 
# ### Result Merging
# - `merge_search_results`: Combines and deduplicates results from different search methods.
# - Sorts results based on relevance scores.
# 
# ### Source Management
# - `update_ground_sources`, `get_ground_sources`, and `clear_ground_sources`: Manage and retrieve a list of unique source URLs.
# 
# This class is designed for efficient task management, multi-method search operations, and user interaction in an AI assistant context. It provides robust error handling, detailed logging, and flexible search capabilities, making it suitable for complex, interactive AI applications in the ASU Discord bot environment.
# 

# %%
class Utils:
    def __init__(self):
        """Initialize the Utils class with task tracking and logging."""
        try:
            self.tasks = []
            self.current_content = "Understanding your question"
            self.message = None
            self.cached_doc_ids = []
            self.ground_sources =[]
            # self.scann_store = None
            self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
            self.cached_queries=[]
            self.vector_store = asu_store.get_vector_store()
            self.raptor_retriever = RaptorRetriever()
            logger.info("\nUtils instance initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Utils: {e}")

    async def start_animation(self, message):
        """Start the loading animation using Discord's built-in thinking indicator"""
        try:
            self.message = message
            logger.info(f"Animation started for message: {message.id}")
        except Exception as e:
            logger.error(f"Failed to start animation: {e}")

    async def update_text(self, new_content):
        """Update text while maintaining task history"""
        try:
            # Append previous content to tasks
            if self.current_content:
                self.tasks.append(self.current_content)
                logger.debug(f"Added task to history: {self.current_content}")

            # Update current content
            self.current_content = new_content
            logger.debug(f"Updated current content to: {new_content}")

            # Format and display all tasks
            display_lines = []

            # Display completed tasks
            if self.tasks:
                display_lines.extend(f"✓ {task}" for task in self.tasks)

            # Add the current task with a different symbol
            display_lines.append(f"⋯ {new_content}")
            content="\n".join(display_lines)
            # Update the message content
            await self.message.edit(content=content)
            logger.info(f"Message updated with {len(display_lines)} tasks")

        except Exception as e:
            logger.error(f"Failed to update text: {e}")
            # Optionally, you could re-raise the exception or handle it differently

    async def stop_animation(self, message=None, final_content=None,View=None):
        """Stop animation and display final content"""
        try:
            # Edit message with final content if provided
            if message and final_content:
                if View:
                    await message.edit(content=final_content,view=View)
                await message.edit(content=final_content)
                logger.info(f"Final content set: {final_content}")

            # Reset internal state
            self.tasks = []
            self.current_content = ""
            self.message = None
            logger.info("\nAnimation stopped and state reset")

        except Exception as e:
            logger.error(f"Error stopping animation: {e}")

    def format_search_results(self, engine_context):
        """Format search results into a readable string."""
        if not engine_context:
            return "No search results found."
        try:
            formatted_results = "\n\n"
            if isinstance(engine_context, str):
                # If engine_context is already formatted, return it as is
                return engine_context

            for i, result in enumerate(engine_context, 1):
                formatted_results += f"## Document {i}\n"

                # Safely access metadata dictionary
                metadata = result.get('metadata', {})
                if not isinstance(metadata, dict):
                    metadata = {}

                # Safely get values with defaults
                title = metadata.get('title', 'No title')
                category = metadata.get('category', 'Uncategorized')
                timestamp = metadata.get('timestamp', 'timestamp')
                url = metadata.get('url', 'No URL')
                content = result.get('content', 'No content available')

                # Build formatted string
                formatted_results += f"**Title:** {title}\n"
                formatted_results += f"**Category:** {category}\n"
                formatted_results += f"**Last Updated:** {timestamp}\n"
                formatted_results += f"""**Source:** {url}\n"""
                formatted_results += "\n**Content:**\n"
                formatted_results += f"{content}\n\n"
                formatted_results += "---\n\n"

            return formatted_results

        except Exception as e:
            logger.error(f"Error formatting search results: {str(e)}")
            return "Error formatting search results."

    async def perform_web_search(self,search_url:str =None,  optional_query : str = None, doc_title : str =None, doc_category : str = None):
        try:
            # Initial search
            logger.info("\nPerforming Web Search")

            documents = await asu_scraper.engine_search(search_url, optional_query)

            if not documents:
                raise ValueError("No documents found matching the query")
                return "No results found on web"
            
            logger.info(documents)
            global action_command
            logger.info("\nPreprocessing documents...")
            
            processed_docs = await asu_data_processor.process_documents(
                documents=documents, 
                search_context=action_command,
                title = doc_title, category = doc_category
            )

            store = asu_store.queue_documents(processed_docs)

            results = []
            extracted_urls=[]
            for doc in processed_docs:
                doc_info = {
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'timestamp': doc.metadata.get('timestamp'),
                    'url': doc.metadata.get('url')
                }
                sources = doc.metadata.get('url')
                extracted_urls.extend(sources)

                results.append(doc_info)

            self.update_ground_sources(extracted_urls)

            results = utils.format_search_results(results)

            return results

        except Exception as e:
            logger.error(f"Error in web search: {str(e)}")
            return "No results found on web"

    async def perform_similarity_search(self, query: str, categories: list):
        try:
            logger.info(f"Action Model: Performing similarity search with query: {query}")
            self.vector_store = asu_store.get_vector_store()
            if not self.vector_store:
                logger.info("\nVector Store not initialized")
                raise ValueError("Vector store not properly initialized")

            # Correct filter construction using Qdrant's Filter class
            filter_conditions = None
            if categories and len(categories) > 0:
                

                filter_conditions = Filter(
                    must=[
                        FieldCondition(
                            key="metadata.category", 
                            match=MatchAny(any=categories)
                        )
                    ]
                )

            # Perform similarity search with optional filtering
            results = self.vector_store.similarity_search(
                query, 
                filter=filter_conditions
            )

            # Check if results are empty
            if not results:
                logger.info("\nNo documents found in vector store")
                return None

            documents = []
            for doc in results:
                doc_info = {
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'timestamp': doc.metadata.get('timestamp'),
                    'url': doc.metadata.get('url'),
                    'category': doc.metadata.get('category')
                }
                documents.append(doc_info)

            logger.info(f"Retrieved {len(documents)} documents from vector store")
            return documents

        except Exception as e:
            logger.error(f"Error during similarity search: {str(e)}")
            return None

    async def perform_mips_search(self, query: str, categories: list):
        try:
            logger.info(f"Action Model: Performing MIPS search with query: {query}")
            query_vector = self.vector_store.embedding_model.embed_query(query)
            
            filter_conditions = None
            if categories and len(categories) > 0:
                filter_conditions = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.category",
                            match=models.MatchAny(any=categories)
                        )
                    ]
                )
            
            results = self.vector_store.mips_search(query_vector, filter=filter_conditions)
            
            documents = []
            for doc in results:
                doc_info = {
                    'content': doc['payload']['page_content'],
                    'metadata': doc['payload']['metadata'],
                    'score': doc['score']
                }
                documents.append(doc_info)
            
            logger.info(f"Retrieved {len(documents)} documents from MIPS search")
            return documents
        except Exception as e:
            logger.error(f"Error during MIPS search: {str(e)}")
            return None
    
    def merge_search_results(self, raptor_results, similarity_results, mips_results):
        combined_results = raptor_results + similarity_results + mips_results
        
        deduplicated_results = []
        seen_urls = set()
        
        for result in combined_results:
            url = result.get('metadata', {}).get('url')
            if url not in seen_urls:
                seen_urls.add(url)
                deduplicated_results.append(result)
        
        # Sort by relevance score (assuming higher is better)
        sorted_results = sorted(deduplicated_results, key=lambda x: x.get('score', 0), reverse=True)
        
        return sorted_results[:10]  # Return top 10 unique results

    async def perform_database_search(self, query: str, categories: list):
        self.cached_queries.append(query)
        
        # Perform RAPTOR search
        raptor_results = await self.raptor_retriever.retrieve(query, top_k=5)
        
        # Perform similarity search
        similarity_results = await self.perform_similarity_search(query, categories)
        
        # Perform MIPS search
        mips_results = await self.perform_mips_search(query, categories)
        query_embedding = self.vector_store.embedding_model.embed_query(query)



        # Combine and deduplicate results
        combined_results = self.merge_search_results(raptor_results, similarity_results, mips_results)
        
        # Process results
        extracted_urls = []
        self.cached_doc_ids.clear()
        
        for doc in combined_results[:5]:
            doc_id = doc.get('metadata', {}).get('id')
            if doc_id:
                self.cached_doc_ids.append(doc_id)
            sources = doc.get('metadata', {}).get('url', [])
            extracted_urls.extend(sources)
        
        self.update_ground_sources(extracted_urls)
        formatted_context = self.format_search_results(combined_results[:5])
        return formatted_context

    def update_ground_sources(self,extracted_urls:[]):
        self.ground_sources.extend(extracted_urls)
        self.ground_sources = list(set(self.ground_sources))
    
    def get_ground_sources(self):
        return self.ground_sources
    
    def clear_ground_sources(self):
        self.ground_sources = []
        return True

# %%
utils = Utils()

# %% [markdown]
# ## Data Preprocessor 
# 
# This `DataPreprocessor` is designed for efficient, context-aware document processing, making it ideal for RAG (Retrieval-Augmented Generation) applications. It ensures clean, structured text output with comprehensive metadata, enhancing downstream NLP tasks.

# %% [markdown]
# 
# 
# 
# ### Initialization
# - The `__init__` method sets up the preprocessor with configurable parameters for text splitting and retry mechanisms.
# - It initializes NLTK resources for advanced text processing.
# - Regular expressions are compiled for efficient text cleaning.
# 
# ### Document Processing
# - The `process_documents` method is the core function for handling multiple documents.
# - It implements a retry mechanism to ensure robust processing.
# - The method consolidates documents, refines content, and splits them into manageable chunks.
# 
# ### Text Cleaning
# - `clean_and_structure_text` performs comprehensive text cleaning:
#   - Removes HTML tags and links
#   - Converts text to ASCII and lowercase
#   - Replaces special characters and normalizes whitespace
#   - Applies tokenization and lemmatization for advanced text normalization
# 
# ### Content Refinement
# - `_refine_content` attempts to improve document content using an AI agent.
# - It handles potential errors during the refinement process.
# 
# ### Document Creation and Splitting
# - `_create_processed_document` generates a document with rich metadata.
# - `_split_and_annotate_document` divides the document into chunks and adds unique identifiers.
# 
# ### Error Handling
# - The class implements a fallback mechanism to handle processing failures.
# - `_generate_fallback_document` creates a basic document when all processing attempts fail.
# 
# 

# %%

class DataPreprocessor:
    def __init__(self, 
                 chunk_size: int = 1024, 
                 chunk_overlap: int = 200, 
                 max_processing_attempts: int = 3):
        """Initialize DataPreprocessor with configurable text splitting and retry mechanism."""
        self.max_processing_attempts = max_processing_attempts
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        self.doc_title = None
        self.doc_category = None
        nltk.download('punkt_tab', quiet=True)
        nltk.download('wordnet', quiet=True)
        self.lemmatizer = WordNetLemmatizer()
        self.WHITESPACE_PATTERN = re.compile(r'\s+')
        self.LINK_PATTERN = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.SPECIAL_CHAR_PATTERN = re.compile(r'[^a-zA-Z0-9\s.,!?;:()\-"\'$]')



        logger.info(f"DataPreprocessor initialized with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")

    async def process_documents(self, 
                                documents: List[Dict[str, str]], 
                                search_context: str, 
                                title: str = None, 
                                category: str = None) -> List[Document]:
        """Process documents with advanced error handling and multiple retry mechanism."""
        await utils.update_text("Understanding Results...")
        logger.info(f"Processing documents with title={title} category={category}")
        self.doc_title = title
        self.doc_category = category

        for attempt in range(self.max_processing_attempts):
            try:
                start_time = time.time()
                logger.info(f"Starting document processing for {len(documents)} documents")
                try:
                    consolidated_text = await self._consolidate_documents(documents)
                except Exception as e:
                    logger.error(f"Document consolidation failed: {str(e)}")
                    raise
                if not self.doc_title:
                    try:
                        self.doc_title = await self._refine_content(search_context, consolidated_text)
                    except Exception as e:
                        logger.error(f"Title refinement failed: {str(e)}")
                        raise
                try:
                    document = self._create_processed_document(consolidated_text, documents)
                except Exception as e:
                    logger.error(f"Document creation failed: {str(e)}")
                    raise
                try:
                    processed_documents = self._split_and_annotate_document(document)
                except  Exception as e:
                    logger.error(f"Document splitting failed: {str(e)}")
                    raise
                
                processing_time = time.time() - start_time
                logger.info(f"Document processing completed in {processing_time:.2f} seconds. "
                            f"Generated {len(processed_documents)} document chunks.")

                return processed_documents

            except Exception as e:
                logger.error(f"Document processing attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.max_processing_attempts - 1:
                    return await self._generate_fallback_document(documents, e)

    def clean_and_structure_text(self, text: str) -> str:
        """Enhanced text cleaning for RAG applications."""
        # Remove HTML tags
        text = BeautifulSoup(text, "html.parser").get_text()

        # Remove links
        text = self.LINK_PATTERN.sub('', text)

        # Convert to ASCII and lowercase
        text = unidecode(text.lower())

        # Replace $ with USD
        text = text.replace('$', 'USD ')

        # Remove special characters while preserving important punctuation
        text = self.SPECIAL_CHAR_PATTERN.sub(' ', text)

        # Tokenize
        tokens = word_tokenize(text)

        # Lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]

        # Rejoin tokens and normalize whitespace
        text = ' '.join(tokens)
        text = self.WHITESPACE_PATTERN.sub(' ', text).strip()

        return text

    async def _consolidate_documents(self, documents: List[Dict[str, str]]) -> str:
        """Consolidate and clean documents into a single text corpus."""
        return '\n\n'.join([
            self.clean_and_structure_text(doc['content']) 
            for doc in documents
        ]).strip()

    async def _refine_content(self, search_context: str, consolidated_text: str) -> Optional[str]:
        """Attempt to refine document content with error handling."""
        try:
            return await asu_data_agent.refine(search_context, consolidated_text)
        except Exception as e:
            logger.warning(f"Content refinement agent failed: {str(e)}")
            return None

    def _create_processed_document(self, 
                                   consolidated_text: str, 
                                   documents: List[Dict[str, str]]) -> Document:
        """Create a processed document with comprehensive metadata."""
        return Document(
            page_content=consolidated_text,
            metadata={
                'title': self.doc_title or 'Untitled',
                'category': self.doc_category or "google_results",
                'url': [doc['metadata']["url"] for doc in documents[:5]],
                'timestamp': datetime.now(),
                'total_source_documents': len(documents),
                'cluster': None
            }
        )

    def _split_and_annotate_document(self, document: Document) -> List[Document]:
        """Split document into chunks and annotate with metadata."""
        splits = self.text_splitter.split_documents([document])

        for i, split in enumerate(splits):
            split.metadata.update({
                'id': str(uuid.uuid4()),
                'chunk_id': str(uuid.uuid4()),
                'chunk_index': i,
                'total_chunks': len(splits)
            })

        return splits

    async def _generate_fallback_document(self, 
                                          documents: List[Dict[str, str]], 
                                          error: Exception) -> List[Document]:
        """Generate a fallback document when all processing attempts fail."""
        fallback_doc = Document(
            page_content=' '.join([doc['content'] for doc in documents]),
            metadata={
                'title': self.doc_title or 'Fallback Document',
                'category': self.doc_category,
                'url': [doc['metadata']["url"] for doc in documents],
                'timestamp': datetime.now(),
                'error_message': str(error),
                'total_source_documents': len(documents)
            }
        )

        return [fallback_doc]

# %%
asu_data_processor = DataPreprocessor()



# %% [markdown]
# ## Web Scraper
# 
# This `ASUWebScraper` is designed to efficiently retrieve and process information from various ASU-related web sources, providing a comprehensive data collection tool for the ASU Discord Research Assistant Bot.
# 
# 

# %% [markdown]
# ### Initialization
# - The `__init__` method sets up the scraper with various configurations:
#   - Initializes a Discord client for potential integration
#   - Sets up Chrome options for headless browsing
#   - Configures headers for web requests
#   - Initializes containers for visited URLs and scraped content
# 
# ### Web Interaction
# - Uses Selenium WebDriver for dynamic web page interactions
# - Implements a `__login__` method for authenticating with ASU's Handshake platform
# - Handles various types of web elements including popups and cookies
# 
# ### Content Extraction
# - The `scrape_content` method is the core function for extracting information from web pages
# - Supports both Selenium-based scraping and Jina AI-powered content extraction
# - Implements retry mechanisms for robust scraping
# 
# ### Specialized Scraping
# - Contains methods for scraping specific ASU resources:
#   - Course catalog information
#   - Library resources
#   - Job postings from Handshake
#   - Shuttle status information
# 
# ### Search Functionality
# - The `engine_search` method performs searches across various ASU platforms
# - Handles different types of search results (Google, ASU Campus Labs, social media)
# - Implements URL deduplication to avoid redundant scraping
# 
# ### Error Handling
# - Robust error handling throughout the class
# - Implements logging for debugging and monitoring scraping processes
# 
# ### Customization
# - Allows for customization of scraping behavior through optional parameters
# - Supports different scraping strategies based on the target website
# 
# 

# %%
class ASUWebScraper:
    def __init__(self):
        self.discord_client = discord_state.get('discord_client')
        self.visited_urls = set()
        self.text_content = []
        self.optionalLinks = []
        self.logged_in_driver= None
        self.driver= None
        self.chrome_options = Options()
        self.chrome_options.add_argument('--headless')  
        self.chrome_options.add_argument('--no-sandbox')
        self.chrome_options.add_argument('--disable-dev-shm-usage')
        self.chrome_options.add_argument('--disable-gpu')
        self.chrome_options.add_argument('--window-size=1920,1080')
        self.chrome_options.add_argument('--ignore-certificate-errors')
        self.chrome_options.add_argument('--disable-extensions')
        self.chrome_options.add_argument('--no-first-run')
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        self.popup = False
    
    async def __login__(self, username, password):
        try:
            # Initialize WebDriver with configured Chrome options
            driver = webdriver.Chrome(options=self.chrome_options)
            
            # Navigate to Handshake login page
            driver.get("https://asu.joinhandshake.com/login?ref=app-domain")
            
            # Wait for page to load
            wait = WebDriverWait(driver, 10)
            
            # Find and click "Sign in with your email address" button
            email_signin_button = wait.until(
                EC.element_to_be_clickable((By.XPATH, "//a[@data-bind='click: toggle']"))
            )
            email_signin_button.click()
            
            # Enter email address
            email_input = wait.until(
                EC.presence_of_element_located((By.ID, "non-sso-email-address"))
            )
            email_input.send_keys(username)
            
            # Click "Next" button
            next_button = wait.until(
                EC.element_to_be_clickable((By.XPATH, "//button[@type='submit' and contains(@class, 'button')]"))
            )
            next_button.click()
            
            # Click "Or log in using your Handshake credentials"
            alternate_login = wait.until(
                EC.element_to_be_clickable((By.CLASS_NAME, "alternate-login-link"))
            )
            alternate_login.click()
            
            # Enter password
            password_input = wait.until(
                EC.presence_of_element_located((By.ID, "password"))
            )
            password_input.send_keys(password)
            
            # Submit login
            submit_button = wait.until(
                EC.element_to_be_clickable((By.XPATH, "//button[@aria-label='submit' and contains(@class, 'button default-focus')]"))
            )
            submit_button.click()
            
            
            # Store the logged-in driver state
            self.logged_in_driver = driver
            
            return True
        
        except Exception as e:
            logger.error(f"Login failed: {str(e)}")
            return False
        
    def handle_feedback_popup(self,driver):
                if self.popup:
                    
                    pass
                else:
                    try:
                        logger.info("\nHandling feedback popup")
                        # Wait for the popup to be present
                        popup = WebDriverWait(driver, 5).until(
                            EC.presence_of_element_located((By.CLASS_NAME, "fsrDeclineButton"))
                        )
                        
                        # Click the "No thanks" button
                        popup.click()
                        logger.info("\nFeedback popup clicked")
                        # Optional: Wait for popup to disappear
                        WebDriverWait(driver, 5).until(
                            EC.invisibility_of_element_located((By.ID, "fsrFullScreenContainer"))
                        )
                        
                        self.popup = True
                    except Exception as e:
                        # If popup doesn't appear or can't be clicked, log or handle silently

                        pass
    
    def handle_cookie(self,driver):
                if self.popup:
                    
                    pass
                else:
                    try:
                        logger.info("\nHandling feedback popup")
                        cookie_button = WebDriverWait(driver, 10).until(
                            EC.element_to_be_clickable((By.ID, "rcc-confirm-button"))
                        )
                        cookie_button.click()
                        logger.info("\nSuccessfully clciked on cookie button")
                    except:
                        pass
    
    async def scrape_content(self, url: str, query_type: str = None, max_retries: int = 3, selenium :bool = False, optional_query:str=None) -> bool:
        """Scrape content using Jina.ai"""
        
        logger.info(f"Scraping url : {url} ")
        logger.info(f"query_type : {query_type} ")
        logger.info(f"max_retries : {max_retries} ")
        logger.info(f"selenium required : {selenium} ")
        logger.info(f"optional query : {optional_query} ")
        
        await utils.update_text("Understanding Results...")

        if isinstance(url, dict):
            url = url.get('url', '')
    
        # Ensure url is a string and not empty
        if not isinstance(url, str) or not url:
            logger.error(f"Invalid URL: {url}")
            return False
        if url in self.visited_urls:
            return False
        
        self.visited_urls.add(url)
        
        if not selenium:
            for attempt in range(max_retries):
                try:
                    loader = WebBaseLoader(url)
                    documents = loader.load()
                    
                    if documents and documents[0].page_content and len(documents[0].page_content.strip()) > 50 and not "requires javascript to be enabled" in documents[0].page_content:
                        self.text_content.append({
                                'content': documents[0].page_content,
                                'metadata': {
                                    'url': url,
                                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                    'title': documents[0].metadata.get('title', ''),
                                    'description': documents[0].metadata.get('description', ''),
                                }
                            })
                        return True
                except Exception as e:
                    logger.error(f"Error fetching content from {url}: {str(e)}")
                    await asyncio.sleep(3) 
                    continue  
                else:
                    jina_url = f"https://r.jina.ai/{url}"
                    response = requests.get(jina_url, headers=self.headers, timeout=30)
                    response.raise_for_status()
                    
                    text = response.text
                    if "LOADING" in text :
                        logger.warning(f"LOADING response detected for {url}. Retry attempt {attempt + 1}")
                        await asyncio.sleep(3)  # Wait before retrying
                        continue
                    
                    if text and len(text.strip()) > 50:
                        self.text_content.append({
                            'content': text,
                            'metadata': {
                                'url': url,
                                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            }
                        })
                        
                        logger.info(self.text_content[-1])
                        return True
                    
            return False
        
        elif 'postings' in url:

                
            # Navigate to Handshake job postings
            self.logged_in_driver.get(url)
            
            
            # Wait for page to load
            wait = WebDriverWait(self.logged_in_driver, 10)
            
            # Parse optional query parameters
            if optional_query:
                query_params = parse_qs(optional_query)
                
                
                
                driver = self.logged_in_driver
                
                # Search Bar Query
                if 'search_bar_query' in query_params:
                    search_input = wait.until(
                        EC.presence_of_element_located((By.XPATH, "//input[@role='combobox']"))
                    )
                    search_input.send_keys(query_params['search_bar_query'][0])
                    logger.info("\nSuccessfully entered search_bar_query")
                
                
                if 'job_location' in query_params:
                    location_button = wait.until(
                        EC.element_to_be_clickable((By.XPATH, "//button[contains(@class, 'style__pill___3uHDM') and .//span[text()='Location']]"))
                    )
                    location_button.click()

                    location_input = wait.until(
                        EC.presence_of_element_located((By.ID, "locations-filter"))
                    )
                    
                    # Remove list brackets and use the first element directly
                    job_location = query_params['job_location'][0]
                    
                    location_input.clear()
                    location_input.send_keys(job_location)
                    logger.info("\n Successfully entered job_location")

                    try:
                        wait.until(EC.presence_of_element_located((By.CLASS_NAME, "mapbox-autocomplete-results")))
                        time.sleep(1)

                        first_location = wait.until(
                            EC.element_to_be_clickable((
                            By.XPATH, 
                            f"//div[contains(@class, 'mapbox-autocomplete-results')]//label[contains(text(), '{job_location}')]"
                            )))
                        
                        first_location.click()
                        
                        logger.info(f"Selected location: {job_location}")
                    except Exception as e:
                        logger.error(f"Error job location: {e}")

                all_filters_button = wait.until(
                    EC.element_to_be_clickable((
                        By.XPATH, 
                        "//button[contains(@class, 'style__pill___3uHDM') and .//span[text()='All filters']]"
                    ))
                )
                all_filters_button.click()
                
                logger.info("\nClicked on all filters")
                                    
                
                if 'job_type' in query_params:
                    # Get the specific job type from query parameters
                    job_type = query_params['job_type'][0]
                    
                    # Function to force click using JavaScript with multiple attempts
                    def force_click_element(driver, element, max_attempts=3):
                        for attempt in range(max_attempts):
                            logger.info("\nAttempting to force click")
                            try:
                                # Try different click methods
                                driver.execute_script("arguments[0].click();", element)
                                time.sleep(0.5)  # Short pause to allow for potential page changes
                                return True
                            except Exception:
                                # Try alternative click methods
                                try:
                                    element.click()
                                except Exception:
                                    # Last resort: move and click
                                    try:
                                        ActionChains(driver).move_to_element(element).click().perform()
                                    except Exception:
                                        continue
                        return False
                    
                    # Check if the job type is in the first level of buttons (Full-Time, Part-Time)
                    standard_job_types = ['Full-Time', 'Part-Time']
                    
                    if job_type in standard_job_types:
                        # Direct selection for standard job types
                        try:
                            job_type_button = wait.until(
                                EC.presence_of_element_located((
                                    By.XPATH,
                                    f"//button[contains(@class, 'style__pill___3uHDM') and .//div[@data-name='{job_type}' and @tabindex='-1']]"
                                ))
                            )
                            force_click_element(driver, job_type_button)
                            logger.info("\nSelect job type")
                        except Exception:
                            pass
                    else:
                        # For nested job types, click More button first
                        try:
                            more_button = wait.until(
                                EC.presence_of_element_located((
                                    By.XPATH, 
                                    "//button[contains(@class, 'style__pill___') and contains(text(), '+ More')]"
                                ))
                            )
                            force_click_element(driver, more_button)
                            logger.info("\nClicked more button")
                            
                            # Wait and force click the specific job type button from nested options
                            job_type_button = wait.until(
                                EC.presence_of_element_located((
                                    By.XPATH,
                                    f"//button[contains(@class, 'style__pill___3uHDM') and .//div[@data-name='{job_type}' and @tabindex='-1']]"
                                ))
                            )
                            force_click_element(driver, job_type_button)
                            logger.info("\nSelect Job type")
                        except Exception:
                            pass
                    
                
                
                # Wait for the Show results button to be clickable
                show_results_button = wait.until(
                    EC.element_to_be_clickable((
                        By.CLASS_NAME, 
                        "style__clickable___3a6Y8"
                    ))
                )

                # Optional: Add a small delay before clicking to ensure page is ready
                time.sleep(4)

                # Force click the Show results button using JavaScript
                driver.execute_script("arguments[0].click();", show_results_button)





                try:
                    # Wait for job cards to be present using data-hook
                    job_cards = wait.until(
                        EC.presence_of_all_elements_located(
                            (By.CSS_SELECTOR, "[data-hook='jobs-card']")
                        )
                    )
                    text_content = []  # Limit to top 3 jobs
                    
                    for job_card in job_cards[:5]:
                        full_job_link = job_card.get_attribute('href')

                        driver.execute_script("arguments[0].click();", job_card)
                        logger.info("\nClicked Job Card")
                        
                        # Wait for preview panel to load using data-hook
                        wait.until(
                            EC.presence_of_element_located(
                                (By.CSS_SELECTOR, "[data-hook='details-pane']")
                            )
                        )
                        
                        # Find 'More' button using a more robust selector
                        more_button = wait.until(
                            EC.element_to_be_clickable((By.CSS_SELECTOR, "button.view-more-button"))
                        )
                        driver.execute_script("arguments[0].click();", more_button)
                        logger.info("\nClicked 'More' button")

                        
                        time.sleep(1)
                        h = html2text.HTML2Text()
                        time.sleep(2)
                        
                        job_preview_html = driver.find_element(By.CLASS_NAME, "style__details-padding___Y_KHb")
                        
                        soup = BeautifulSoup(job_preview_html.get_attribute('outerHTML'), 'html.parser')
    
                        # Find and remove the specific div with the class
                        unwanted_div = soup.find('div', class_='sc-gwVtdH fXuOWU')
                        if unwanted_div:
                            unwanted_div.decompose()
                        
                        unwanted_div = soup.find('div', class_='sc-dkdNSM eNTbTl')
                        if unwanted_div:
                            unwanted_div.decompose()
                        unwanted_div = soup.find('div', class_='sc-jEYHeb hSVHZy')
                        if unwanted_div:
                            unwanted_div.decompose()
                        unwanted_div = soup.find('div', class_='sc-VJPgA bRBKUF')
                        if unwanted_div:
                            unwanted_div.decompose()
                        

                        markdown_content = h.handle(str(soup))
                        
                        # remove image links
                        markdown_content = re.sub(r'!\[.*?\]\(.*?\)', '', markdown_content)
    
                        # Remove hyperlinks
                        markdown_content = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', markdown_content)

                        markdown_content = markdown_content.replace('\n',' ')
                        
                        markdown_content = markdown_content.replace('[',' ')
                        
                        markdown_content = markdown_content.replace(']',' ')
                        
                        markdown_content = markdown_content.replace('/',' ')
                        
                        self.text_content.append({
                            'content': markdown_content,
                            'metadata': {
                                'url': full_job_link,
                                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            }
                        })
                except Exception as e:
                        logger.error(f"Error html to makrdown conversion :  {e}")
            
            return True
        
        elif 'catalog.apps.asu.edu' in url:
            driver = self.driver
            driver.get(url)
            
            self.handle_cookie(driver)
            course_elements = WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.CLASS_NAME, "class-accordion"))
            )

            detailed_courses = []
            
            for course in course_elements[:7]:
                try:
                    course_title_element = course.find_element(By.CSS_SELECTOR, ".course .bold-hyperlink")
                    course_title = course_title_element.text
                    
                    # Use JavaScript click to handle potential interception
                    driver.execute_script("arguments[0].click();", course_title_element)
                    logger.info("\nSuccessfully clicked on the course")

                    # Wait for dropdown to load
                    details_element = WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.ID, "class-details"))
                    )
                    
                    # Extract additional details
                    course_info = {
                        'title': course_title,
                        'description': details_element.find_element(By.XPATH, ".//h5[contains(text(), 'Course Description')]/following-sibling::p").text,
                        'enrollment_requirements': details_element.find_element(By.XPATH, ".//h5[contains(text(), 'Enrollment Requirements')]/following-sibling::p").text,
                        'location': details_element.find_element(By.XPATH, ".//h5[contains(text(), 'Location')]/following-sibling::p").text,
                        'number': details_element.find_element(By.XPATH, ".//h5[contains(text(), 'Number')]/following-sibling::p").text,
                        'units': details_element.find_element(By.XPATH, ".//h5[contains(text(), 'Units')]/following-sibling::p").text,
                        'dates': details_element.find_element(By.CLASS_NAME, "text-nowrap").text,
                        'offered_by': details_element.find_element(By.XPATH, ".//h5[contains(text(), 'Offered By')]/following-sibling::p").text,
                        'repeatable_for_credit': details_element.find_element(By.XPATH, ".//h5[contains(text(), 'Repeatable for credit')]/following-sibling::p").text,
                        'component': details_element.find_element(By.XPATH, ".//h5[contains(text(), 'Component')]/following-sibling::p").text,
                        'last_day_to_enroll': details_element.find_element(By.XPATH, ".//h5[contains(text(), 'Last day to enroll')]/following-sibling::p").text,
                        'drop_deadline': details_element.find_element(By.XPATH, ".//h5[contains(text(), 'Drop deadline')]/following-sibling::p").text,
                        'course_withdrawal_deadline': details_element.find_element(By.XPATH, ".//h5[contains(text(), 'Course withdrawal deadline')]/following-sibling::p").text,
                        'consent': details_element.find_element(By.XPATH, ".//h5[contains(text(), 'Consent')]/following-sibling::p").text,
                        'course_notes': details_element.find_element(By.XPATH, ".//h5[contains(text(), 'Course Notes')]/following-sibling::p").text,
                        'fees': details_element.find_element(By.XPATH, ".//h5[contains(text(), 'Fees')]/following-sibling::p").text
                    }
                    
                    # Extract reserved seats information
                    try:
                        reserved_seats_table = details_element.find_element(By.CLASS_NAME, "reserved-seats")
                        reserved_groups = []
                        rows = reserved_seats_table.find_elements(By.TAG_NAME, "tr")[1:-1]  # Skip header and last row
                        for row in rows:
                            cols = row.find_elements(By.TAG_NAME, "td")
                            reserved_groups.append({
                                'group': cols[0].text,
                                'available_seats': cols[1].text,
                                'students_enrolled': cols[2].text,
                                'total_seats_reserved': cols[3].text,
                                'reserved_until': cols[4].text
                            })
                        course_info['reserved_seats'] = reserved_groups
                    except:
                        course_info['reserved_seats'] = []
                    
                    detailed_courses.append(course_info)
                    
                except Exception as e:
                    logger.error(f"Error processing course {e}")

                    
                formatted_courses = []
                for course in detailed_courses:
                    course_string = f"Title: {course['title']}\n"
                    course_string += f"Description: {course['description']}\n"
                    course_string += f"Enrollment Requirements: {course['enrollment_requirements']}\n"
                    course_string += f"Instructor: {course['instructor']}\n"
                    course_string += f"Location: {course['location']}\n"
                    course_string += f"Course Number: {course['number']}\n"
                    course_string += f"Units: {course['units']}\n"
                    course_string += f"Dates: {course['dates']}\n"
                    course_string += f"Offered By: {course['offered_by']}\n"
                    course_string += f"Repeatable for Credit: {course['repeatable_for_credit']}\n"
                    course_string += f"Component: {course['component']}\n"
                    course_string += f"Last Day to Enroll: {course['last_day_to_enroll']}\n"
                    course_string += f"Drop Deadline: {course['drop_deadline']}\n"
                    course_string += f"Course Withdrawal Deadline: {course['course_withdrawal_deadline']}\n"
                    course_string += f"Consent: {course['consent']}\n"
                    course_string += f"Course Notes: {course['course_notes']}\n"
                    course_string += f"Fees: {course['fees']}\n"

                    # Add reserved seats information
                    if course.get('reserved_seats'):
                        course_string += "Reserved Seats:\n"
                        for group in course['reserved_seats']:
                            course_string += f"- Group: {group['group']}\n"
                            course_string += f"  Available Seats: {group['available_seats']}\n"
                            course_string += f"  Students Enrolled: {group['students_enrolled']}\n"
                            course_string += f"  Total Reserved Seats: {group['total_seats_reserved']}\n"
                            course_string += f"  Reserved Until: {group['reserved_until']}\n"
                    self.text_content.append({
                            'content': course_string,
                            'metadata': {
                                'url': url,
                                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            }
                        })
                    logger.info(f"Appended {self.text_content[-1]}")
                    formatted_courses.append(course_string)
        
        elif 'search.lib.asu.edu' in url:
            self.driver.get(url)
            time.sleep(1) 
            book_results=[]
            self.handle_feedback_popup(self.driver)
            try:
                # Find and click on the first book title link
                first_book_link = WebDriverWait(self.driver, 10).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "h3.item-title a"))
                )
                first_book_link.click()
                logger.info("\nBook Title Clicked")
            
                try:
                    book_details = WebDriverWait(self.driver, 10).until(
                        EC.presence_of_element_located((By.CLASS_NAME, "full-view-inner-container"))
                    )
                    logger.info("\nBook Details Clicked")

                except:

                    first_book_link = WebDriverWait(self.driver, 10).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, "h3.item-title a"))
                    )
                    first_book_link.click()
                    logger.info("\nBook Title Clicked")
                
                
                self.handle_feedback_popup(self.driver)
                    
                for _ in range(3):
                    # Wait for book details to be present
                    self.handle_feedback_popup(self.driver)
                    book_details = WebDriverWait(self.driver, 10).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, "div.full-view-inner-container.flex"))
                    )
                    logger.info("\nBook Details fetched")
                    
                    

                    
                    # Extract book title
                    author_view = self.driver.find_element(By.CSS_SELECTOR, "div.result-item-text.layout-fill.layout-column.flex")
                    logger.info("\nAuthors fetched")
                    
                    title = author_view.find_element(By.CSS_SELECTOR, "h3.item-title").text.strip()
                    logger.info("\nBook Title fetched")
                    
                    # Extract Authors
                    
                    authors = []
        
                    
                    try:
                        author_div = author_view.find_element(By.XPATH, "//div[contains(@class, 'item-detail') and contains(@class, 'reduce-lines-display')]")
                        
                        
                        # Find all author elements within this div
                        author_elements = author_div.find_elements(By.CSS_SELECTOR, "span[data-field-selector='creator'], span[data-field-selector='contributor']")
                        
                        if len(author_elements)>0:
                            for element in author_elements:
                                author_text = element.text.strip()
                                if author_text and author_text not in authors:
                                    authors.append(author_text)
                        else:
                            author_div = book_details.find_element(By.XPATH, "//div[.//span[@title='Author']]")
                        
                            author_elements = author_div.find_elements(By.CSS_SELECTOR, "a span[ng-bind-html='$ctrl.highlightedText']")
                            
                            if not author_elements:

                                author_elements = book_details.find_elements(By.XPATH, "//div[contains(@class, 'item-details-element')]//a//span[contains(@ng-bind-html, '$ctrl.highlightedText')]")
                            if len(author_elements)>0:
                                for element in author_elements:
                                    author_text = element.text.strip()
                                    if author_text and author_text not in authors:
                                        authors.append(author_text)
                        logger.info("\nAuthors fetched")
                        
                    except Exception as e:
                        
                        author = 'N/A'

                    
                    try:
                        publisher = book_details.find_element(By.CSS_SELECTOR, "span[data-field-selector='publisher']").text.strip()
                        logger.info("\nPublisher fetched")
                    except:
                        logger.info("\nNo Publisher found")
                        publisher = "N/A"
                    
                    # Extract publication year
                    try:
                        year = book_details.find_element(By.CSS_SELECTOR, "span[data-field-selector='creationdate']").text.strip()
                    except:
                        logger.info("\nNo Book details found")
                        year = "N/A"
                    
                    # Extract availability
                    try:
                        location_element = book_details.find_element(By.CSS_SELECTOR, "h6.md-title")
                        availability = location_element.text.strip()
                        logger.info("\nAvailability found with first method")

                    except Exception as e:
                        # Find the first link in the exception block
                        location_element = book_details.find_elements(By.CSS_SELECTOR, "a.item-title.md-primoExplore-theme")
                        if isinstance(location_element,list):
                            availability = location_element[0].get_attribute('href')
                        else:
                            availability = location_element.get_attribute('href')
                        logger.info("\nAvailability found with second method")
                        
                        if availability is None:
                            location_element = book_details.find_elements(By.CSS_SELECTOR, "h6.md-title ng-binding zero-margin")
                            availability = location_element.text.strip()
                            logger.info("\nAvailablility found with third method")
                            

                        
                    try:
                        # Use more flexible locator strategies
                        links = self.driver.find_elements(By.XPATH, "//a[contains(@ui-sref, 'sourceRecord')]")
                        
                        if isinstance(links, list) and len(links) > 0:
                            
                            link = links[0]
                            link = link.get_attribute('href')
                            logger.info("\nFetched Link")
                        else:
                            link = link.get_attribute('href')
                            logger.info("\nFetched Link")
                    except Exception as e:
                        logger.info("\nNo link Found")


                    # Compile book result
                    book_result = {
                        "title": title,
                        "authors": authors,
                        "publisher": publisher,
                        "year": year,
                        "availability": availability,
                        "link": link
                    }
                    
                    book_results.append(book_result)
                    
                    try:
                        next_button = WebDriverWait(self.driver, 5).until(
                            EC.element_to_be_clickable((By.XPATH, "//button[contains(@ng-click, '$ctrl.getNextRecord()')]"))
                        )
                        self.driver.execute_script("arguments[0].click();", next_button)

                        next_button.click()
                        
                        logger.info("\nClciked next button")

                        self.handle_feedback_popup(self.driver)
                        
                    except Exception as e:
                        logger.error(f"Failed to click next button")
                    
                if len(book_results)==0:
                    return False
                
                for book in book_results:
                    book_string = f"Title: {book['title']}\n"
                    book_string += f"Authors: {', '.join(book['authors']) if book['authors'] else 'N/A'}\n"
                    book_string += f"Publisher: {book['publisher']}\n"
                    book_string += f"Publication Year: {book['year']}\n"
                    book_string += f"Availability: {book['availability']}\n"
                    book_string += f"Link: {book['link']}\n"

                    self.text_content.append({
                        'content': book_string,
                        'metadata': {
                            'url': book['link'],
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        }
                    })
                    logger.info("\nAppended book details: %s" % self.text_content[-1])
                    
            except Exception as e:
                logger.info("\nFailed to append book details: %s" % e)
                return False

            return True
        
        elif 'lib.asu.edu' in url:
            def extract_query_parameters(query):
                pattern = r'(\w+)=([^&]*)'
                matches = re.findall(pattern, query)
                parameters = [{param: value} for param, value in matches]
                return parameters

            # Classify the extracted parameters into lists
            library_names = []
            dates = []
            results = []

            # Extract parameters from the query string
            params = extract_query_parameters(optional_query)

            # Populate the lists based on parameter types
            for param in params:
                for key, value in param.items():
                    if key == 'library_names' and value != 'None':
                        library_names.append(value.replace("['", "").replace("']", ""))
                    if key == 'date' and value != 'None':
                        dates.append(value.replace("['","").replace("']",""))
            
            try:
                driver = self.driver
                # Navigate to library hours page
                self.driver.get(url)

                # Wait for page to load
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "s-lc-whw"))
                )
                
                # Handle cookie popup
                cookie_button = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.CLASS_NAME, "accept-btn"))
                )
                cookie_button.click()

                # Map library names to their row identifiers
                library_map = {
                    "Tempe Campus - Noble Library": "Noble Library",
                    "Tempe Campus - Hayden Library": "Hayden Library",
                    "Downtown Phoenix Campus - Fletcher Library": "Downtown campus Library",
                    "West Campus - Library": "Fletcher (West Valley)",
                    "Polytechnic Campus - Library": "Polytechnic"
                }


                # Process each library and date
                for library_name in library_names: 
                    
                    for date in dates:    
                        iterations = 0
                        is_date_present = False

                        while not is_date_present:
                            # Find all date headers in the thead
                            date_headers = self.driver.find_elements(
                                By.XPATH, "//thead/tr/th/span[@class='s-lc-whw-head-date']"
                            )
                            
                            # Extract text from date headers
                            header_dates = [header.text.strip() for header in date_headers]
                            
                            # Remove line breaks and additional whitespace
                            header_dates = [date.lower().split('\n')[0] for date in header_dates]
                                
                            # Check if requested date is in the list of header dates
                            is_date_present = date.lower() in header_dates
                            
                            if not is_date_present:
                                next_button = self.driver.find_element(By.ID, "s-lc-whw-next-0")
                                next_button.click()
                                time.sleep(0.2)  # Allow page to load
                            
                            iterations += 1
                        
                        # Optional: logger.info debug information
                        logger.info(f"Available Dates: {header_dates}")
                        logger.info(f"Requested Date: {date}")
                        logger.info(f"Date Present: {is_date_present}")
                        
                    
                        logger.info("\nhello")
                        mapped_library_names = library_map.get(str(library_name))
                        logger.info(f"Mapped library names: {mapped_library_names}")
                        
                        # Find library row
                        library_row = self.driver.find_element(
                            By.XPATH, f"//tr[contains(., '{mapped_library_names}')]"
                        )
                        
                        
                        logger.info("\nFound library row")

                        # Find date column index
                        date_headers = self.driver.find_elements(By.XPATH, "//thead/tr/th/span[@class='s-lc-whw-head-date']")
                        
                        logger.info(f"Found date_headers")
                        
                        date_column_index = None
                        for index, header in enumerate(date_headers, start=0):
                            logger.info(f"header.text.lower() = {header.text.lower()}")  
                            logger.info(f"date.lower() = {date.lower()}")  
                            if date.lower() == header.text.lower():
                                date_column_index = index+1 if index==0 else index
                                logger.info("\nFound date column index")
                                break

                        if date_column_index is None:
                            logger.info("\nNo date info found")
                            continue  # Skip if date not found
                        
                        logger.info(f"Found date column index {date_column_index}")
                        # Extract status
                        status_cell = library_row.find_elements(By.TAG_NAME, "td")[date_column_index]
                        logger.info(f"Found library row elements : {status_cell}")
                        try:
                            status = status_cell.find_element(By.CSS_SELECTOR, "span").text
                            logger.info(f"Found library status elements : {status}")
                        except Exception as e:
                            logger.info(f"Status cell HTML: {status_cell.get_attribute('outerHTML')}")
                            logger.error(f"Error extracting library status: {e}")
                            raise
                            break

                        # Append to results
                        library_result = {
                            'library': mapped_library_names,
                            'date': date,
                            'status': status
                        }
                        logger.info(f"mapping {library_result}")
                        results.append(library_result)

                # Convert results to formatted string for text_content
                logger.info(f"Results : {results}")
                for library in results:
                    lib_string = f"Library: {library['library']}\n"
                    lib_string += f"Date: {library['date']}\n"
                    lib_string += f"Status: {library['status']}\n"
                    
                    self.text_content.append({
                        'content': lib_string,
                        'metadata': {
                            'url': url,
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        }
                    })
                logger.info(self.text_content)
                
                return True

            except Exception as e:
                return f"Error retrieving library status: {str(e)}" 
                        
        
            
            return False
            
        elif 'asu.libcal.com' in url:
            # Navigate to the URL
            self.driver.get(url)
            
            # Wait for page to load
            
            try:
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CLASS_NAME, 'panel'))
                )
                
                # Parse page source with BeautifulSoup
                soup = BeautifulSoup(self.driver.page_source, 'html.parser')
                
                
                # Reset text_content for new scrape
                self.text_content = []
                
                # Find all study room panels
                study_rooms = soup.find_all('div', class_='panel panel-default')
                
                for room in study_rooms:
                    # Extract study room name
                    study_room_name = room.find('h2', class_='panel-title').text.split('\n')[0].strip()
                    # Extract the date (consistent across all rooms)
                    date = room.find('p').text.strip()  # "Friday, December 6, 2024"
                    
                    # Find all available time slots
                    available_times = []
                    time_slots = room.find_all('div', class_='checkbox')
                    
                    for slot in time_slots:
                        time_text = slot.find('label').text.strip()
                        available_times.append(time_text)
                    
                    # Append to text_content
                    self.text_content.append({
                        'content': f"""Library: {optional_query}\nStudy Room: {study_room_name}\nDate: {date}\nAvailable slots: {', '.join(available_times)}""",
                        'metadata': {
                            'url': url,
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'source_type': 'asu_web_scrape',
                        }
                    })
            except Exception as e:
                logger.error(f"Error extracting study room data: {e}")
                return "No Study Rooms Open Today"
                
            
            return True
        
        elif 'asu-shuttles.rider.peaktransit.com' in url:
            query = optional_query
            # Navigate to the URL
            try:
                # Navigate to the URL
                self.driver.get(url)
                time.sleep(3)
                # Wait for route list to load
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_all_elements_located((By.CSS_SELECTOR, '#route-list .route-block .route-name'))
                )
                # Target the route list container first
                route_list = self.driver.find_element(By.CSS_SELECTOR, "div#route-list.route-block-container")
                
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'div.route-block'))
                )
                route_blocks = route_list.find_elements(By.CSS_SELECTOR, "div.route-block")
                
                iterate_Y = 0
                results=[]
                button_times = 0
                iterate_X=0
                route = None
                for route_block in route_blocks:
                    WebDriverWait(self.driver, 10).until(
                        EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'div.route-name'))
                    )
                    route_name = route_block.find_element(By.CSS_SELECTOR, "div.route-name")
                    logger.info("\nloacted routenames")
                    if "Tempe-Downtown" in route_name.text and  "Tempe-Downtown" in query:
                        button_times =5
                        route = route_name.text
                        route_block.click()
                        logger.info("\nclicked")
                        break
                    elif "Tempe-West" in route_name.text and "Tempe-West" in query:
                        button_times=5
                        route = route_name.text
                        route_block.click()
                        logger.info("\nclicked")
                        break
                    elif "Mercado" in route_name.text and "Mercado" in query:
                        button_times = 2
                        iterate_X = 12
                        iterate_Y = 8
                        route = route_name.text

                        route_block.click()
                        logger.info("\nMercado")
                        break
                    elif "Polytechnic" in route_name.text and "Polytechnic" in query:
                        button_times = 2
                        route_block.click()
                        iterate_X = 10
                        iterate_Y = 17
                        route = route_name.text

                    
                        logger.info("\nPolytechnic")
                        break
                
                time.sleep(2)
                
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_all_elements_located((By.XPATH, "//div[contains(@style, 'z-index: 106')]"))
                )
                
                try:
                    zoom_out_button = WebDriverWait(self.driver, 10).until(
                        EC.element_to_be_clickable((By.XPATH, "//button[@aria-label='Zoom out']"))
                    )
                    
                    for _ in range(button_times):
                        zoom_out_button.click()
                        time.sleep(0.5)  # Short pause between clicks
                
                except Exception as e:
                    logger.info(f"Error clicking zoom out button: {e}")

                map_div = None
                try:
                    # Method 1: JavaScript click
                    map_div = self.driver.find_element(By.CSS_SELECTOR, "div[aria-label='Map']")
                    self.driver.execute_script("arguments[0].click();", map_div)
                    logger.info("\nfirst method worked")
                
                except Exception as first_error:
                    try:
                        # Method 2: ActionChains click
                        map_div = driver.find_element(By.CSS_SELECTOR, "div[aria-label='Map']")
                        actions = ActionChains(self.driver)
                        actions.move_to_element(map_div).click().perform()
                        logger.info("\nsecond method worked")
                    
                    except Exception as second_error:
                        try:
                            # Method 3: Move and click with offset
                            map_div = driver.find_element(By.CSS_SELECTOR, "div[aria-label='Map']")
                            actions = ActionChains(self.driver)
                            actions.move_to_element_with_offset(map_div, 10, 10).click().perform()
                            logger.info("\nthird method worked")
                                 
                        except Exception as third_error:
                            logger.info(f"All click methods failed: {first_error}, {second_error}, {third_error}")

                
                
                actions = ActionChains(self.driver)
                if "Mercado" in query:
                    # Move map to different directions
                    directions_x = [
                        (300, 0), 
                    ]
                    directions_y = [
                        (0, 300),   
                    ]
                    
                    for i in range(0, iterate_X):
                        
                        for dx, dy in directions_x:
                            # Click and hold on map
                            actions.move_to_element(map_div).click_and_hold()
                            
                            # Move by offset
                            actions.move_by_offset(dx, dy)
                            
                            # Release mouse button
                            actions.release()
                            
                            # Perform the action
                            actions.perform()
                            logger.info("\nmoved")
                            # Wait a moment between movements
                    logger.info("\niterating over y")        
                    for i in range(0, iterate_Y):
                        for dx, dy in directions_y:
                            actions.move_to_element(map_div).click_and_hold()
                            actions.move_by_offset(dx, dy)
                            actions.release()
                            actions.perform()
                            logger.info("\nmoved")
                if "Polytechnic" in query:
                    logger.info("\npoly")
                    # Move map to different directions
                    directions_x = [
                        (-300, 0),
                    ]
                    directions_y = [
                        (0, -300),   
                    ]
                    
                    for i in range(0, iterate_X):
                        
                        for dx, dy in directions_x:
                            # Click and hold on map
                            actions.move_to_element(map_div).click_and_hold()
                            
                            # Move by offset
                            actions.move_by_offset(dx, dy)
                            actions.move_by_offset(dx, dy)
                            
                            # Release mouse button
                            actions.release()
                            
                            # Perform the action
                            actions.perform()
                            logger.info("\nmoved")
                            # Wait a moment between movements
                    logger.info("\niterating over y")        
                    for i in range(0, iterate_Y):
                        
                        for dx, dy in directions_y:
                            actions.move_to_element(map_div).click_and_hold()
                            actions.move_by_offset(dx, dy)
                            actions.release()
                            actions.perform()
                            logger.info("\nmoved")
                  
                map_markers = self.driver.find_elements(By.CSS_SELECTOR, 
                    'div[role="button"]  img[src="https://maps.gstatic.com/mapfiles/transparent.png"]')
                
                for marker in map_markers:
                    try:
                        parent_div = marker.find_element(By.XPATH, '..')
                        self.driver.execute_script("arguments[0].click();", parent_div)
                        
                        dialog = WebDriverWait(self.driver, 10).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, 'div[role="dialog"]'))
                        )
                        
                        dialog_html = dialog.get_attribute('outerHTML')
                        soup = BeautifulSoup(dialog_html, 'html.parser')
                        
                        stop_name_elem = soup.find('div', class_='stop-name')
                        if stop_name_elem:
                            stop_name = stop_name_elem.find('h2').get_text(strip=True)
                            routes = soup.find_all('div', class_='route-name')
                            
                            station_routes = []
                            for route in routes:
                                route_name = route.get_text(strip=True)
                                bus_blocks = route.find_next_siblings('div', class_='bus-block')
                                
                                # Safer extraction of bus times
                                try:
                                    next_bus_time = bus_blocks[0].find('div', class_='bus-time').get_text(strip=True) if bus_blocks else 'N/A'
                                    second_bus_time = bus_blocks[1].find('div', class_='bus-time').get_text(strip=True) if len(bus_blocks) > 1 else 'N/A'
                                    
                                    station_routes.append({
                                        'Route': route_name,
                                        'Next Bus': next_bus_time,
                                        'Second Bus': second_bus_time
                                    })
                                except IndexError:
                                    # Skip routes without bus times
                                    continue
                            
                            # Only append if station_routes is not empty
                            if station_routes:
                                parsed_stations = [{
                                    'Station': stop_name,
                                    'Routes': station_routes
                                }]
                                results.extend(parsed_stations)
                        
                        
                        
                    
                    
                    except Exception as e:
                        # Log the error without stopping the entire process
                        logger.info(f"Error processing marker: {e}")
                        continue
                content = [
                            f"Station : {result['Station']}\n"
                            f"Route : {route['Route']}\n"
                            f"Next Bus : {route['Next Bus']}\n"
                            f"Second Bus : {route['Second Bus']}"
                            for result in results
                            for route in result['Routes']
                            if 'mins.' in route['Next Bus'] and 'mins.' in route['Second Bus']
                        ]
                content = set(content)  
                for c in content:
                    self.text_content.append({
                        'content': c,
                        'metadata': {
                            'url': url,
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),

                        }
                    })        
                return True
            
            except Exception as e:
                logger.info(f"Error extracting shuttle status: {e}")
                return False
            
        else:
            logger.error("NO CHOICE FOR SCRAPER!")
            
        return False
    
    async def discord_search(self, query: str, channel_ids: List[int], limit: int = 40) -> List[Dict[str, str]]:
        if not self.discord_client:
            logger.info(f"Could not initialize discord_client {self.discord_client}")
            return []
        
        messages = []
        await utils.update_text("Searching the Sparky Discord Server")
        
        for channel_id in channel_ids:
            channel = self.discord_client.get_channel(channel_id)
            
            if not channel:
                logger.info(f"Could not access channel with ID {channel_id}")
                continue
            
            if isinstance(channel, discord.TextChannel):
                async for message in channel.history(limit=limit):
                    messages.append(self._format_message(message))
            elif isinstance(channel, discord.ForumChannel):
                async for thread in channel.archived_threads(limit=limit):
                    async for message in thread.history(limit=limit):
                        messages.append(self._format_message(message))
            
            if len(messages) >= limit:
                break
            
        print(messages)
        
        for message in messages[:limit]:
            self.text_content.append({
                'content': message['content'],
                'metadata': {
                    'url': message['url'],
                    'timestamp': message['timestamp'],
                }
            })

        
        return True

    def _format_message(self, message: discord.Message) -> Dict[str, str]:
        timestamp = message.created_at.strftime("%Y-%m-%d %H:%M:%S")
        
        formatted_content = (
            f"Sent by: {message.author.name} {timestamp}\n"
            f"Message content: {message.content}"
        )
        
        return {
            'url': message.jump_url,
            'content': formatted_content,
            'timestamp': timestamp
        }
    
    async def engine_search(self, search_url: str =None, optional_query : str = None ) -> List[Dict[str, str]]:
        """Handle both Google search results and ASU Campus Labs pages using Selenium"""
        
        try:
            search_results = []
            await utils.update_text(f"Searching for {search_url}")
            await self.discord_search(query=optional_query, channel_ids=[1323386884554231919,1298772258491203676,1256079393009438770,1256128945318002708], limit=30)
            self.driver = webdriver.Chrome(options=self.chrome_options)
            driver = self.driver
            wait = WebDriverWait(driver, 10)
            if (search_url):
                try:
                    driver.get(search_url)

                    if 'google.com/search' in search_url:
                        # Wait for search results to load
                        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'div.g')))
                        
                        # Find all search result elements
                        results = driver.find_elements(By.CSS_SELECTOR, 'div.g')
                        for result in results[:5]:  # Limit to top 3 results
                            try:
                                link = result.find_element(By.CSS_SELECTOR, 'a')
                                url = link.get_attribute('href')
                                
                                if url and url.startswith('http'):
                                    clean_url = f"{url}"
                                    if clean_url not in search_results:
                                        search_results.append(clean_url)
                            except Exception as e:
                                continue
                        
                        logger.info(f"Found {len(search_results)} Google search results")
                                        # Handle ASU Campus Labs pages
                    
                    if 'asu.campuslabs.com/engage' in search_url:
                        if 'events' in search_url:
                            # Wait for events to load
                            events = wait.until(EC.presence_of_all_elements_located(
                                (By.CSS_SELECTOR, 'a[href*="/engage/event/"]')
                            ))
                            search_results = [
                                event.get_attribute('href') 
                                for event in events[:5]
                            ]
                            
                        elif 'organizations' in search_url:
                            # Wait for organizations to load
                            orgs = wait.until(EC.presence_of_all_elements_located(
                                (By.CSS_SELECTOR, 'a[href*="/engage/organization/"]')
                            ))
                            search_results = [
                                org.get_attribute('href') 
                                for org in orgs[:5]
                            ]
                            
                        elif 'news' in search_url:
                            # Wait for news items to load
                            news = wait.until(EC.presence_of_all_elements_located(
                                (By.CSS_SELECTOR, 'a[href*="/engage/news/"]')
                            ))
                            search_results = [
                                article.get_attribute('href') 
                                for article in news[:5]
                            ]
                        
                        
                        logger.info(f"Found {len(search_results)} ASU Campus Labs results")
                    
                    if 'x.com' in search_url or 'facebook.com' in search_url or "instagram.com" in search_url:
                        if optional_query:
                            logger.info("\nOptional query :: %s" % optional_query)
                            google_search_url = f"https://www.google.com/search?q={urllib.parse.quote(optional_query)} site:{urlparse(search_url).netloc}"
                            google_results = await self.engine_search(search_url=google_search_url)
                            
                            google_filtered_results = [
                                url for url in google_results 
                                if urlparse(url).netloc == urlparse(search_url).netloc
                            ][:5]
                            
                            search_results.extend(google_filtered_results)
                        else:
                            if 'x.com' in search_url or 'twitter.com' in search_url:
                                try:
                                    try:
                                        WebDriverWait(driver, 30).until(
                                            EC.presence_of_all_elements_located((By.TAG_NAME, 'body'))
                                        )
                                    except Exception as e:
                                        logger.warning(f"Timeout waiting for tweets to load {str(e)}")
                                    page_source = driver.page_source
                                    
                                    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                                    time.sleep(3)
                                    
                                    # Define tweet selectors
                                    tweet_selectors = [
                                        'article[data-testid="tweet"]',
                                        'div[data-testid="cellInnerDiv"]',
                                        'div[role="article"]',
                                        
                                    ]
                                    
                                    # Find tweet articles
                                    tweet_articles = []
                                    for selector in tweet_selectors:
                                        tweet_articles = driver.find_elements(By.CSS_SELECTOR, selector)
                                        if tweet_articles:
                                            break
                                    
                                    if not tweet_articles:
                                        logger.error('No tweet articles found')
                                        return []
                                    
                                    # Extract top 3 tweet links
                                    tweet_links = []
                                    for article in tweet_articles[:5]:
                                        try:
                                            link_selectors = [
                                                'a[href*="/status/"]',
                                                'a[dir="ltr"][href*="/status/"]'
                                            ]
                                            
                                            for selector in link_selectors:
                                                try:
                                                    timestamp_link = article.find_element(By.CSS_SELECTOR, selector)
                                                    tweet_url = timestamp_link.get_attribute('href')
                                                    if tweet_url:
                                                        tweet_links.append(tweet_url)
                                                        break
                                                except:
                                                    continue
                                        except Exception as inner_e:
                                            logger.error(f"Error extracting individual tweet link: {str(inner_e)}")
                                    
                                    logger.info(tweet_links)
                                    search_results.extend(tweet_links)
                                    logger.info(f"Found {len(tweet_links)} X (Twitter) links")
                                    
                                except Exception as e:
                                    logger.error(f"X.com tweet link extraction error: {str(e)}")
                                    try:
                                        driver.save_screenshot("x_com_error_screenshot.png")
                                    except:
                                        pass

                            
                            elif 'instagram.com' in search_url:
                                try:
                                    instagram_post_selectors = [
                                        'article[role="presentation"]',
                                        'div[role="presentation"]',
                                        'div[class*="v1Nh3"]'
                                    ]
                                    
                                    instagram_link_selectors = [
                                        'a[href*="/p/"]',
                                        'a[role="link"][href*="/p/"]'
                                    ]
                                    
                                    instagram_articles = []
                                    for selector in instagram_post_selectors:
                                        instagram_articles = driver.find_elements(By.CSS_SELECTOR, selector)
                                        if instagram_articles:
                                            break
                                    
                                    instagram_links = []
                                    for article in instagram_articles[:5]:
                                        for link_selector in instagram_link_selectors:
                                            try:
                                                post_link = article.find_element(By.CSS_SELECTOR, link_selector)
                                                insta_url = post_link.get_attribute('href')
                                                if insta_url and insta_url not in instagram_links:
                                                    instagram_links.append(insta_url)
                                                    break
                                            except Exception as insta_link_error:
                                                continue
                                    
                                    search_results.extend(instagram_links)
                                    logger.info(f"Found {len(instagram_links)} Instagram post links")
                                
                                except Exception as instagram_error:
                                    logger.error(f"Instagram link extraction error: {str(instagram_error)}")
                                    try:
                                        driver.save_screenshot("instagram_error_screenshot.png")
                                    except:
                                        pass

                        logger.info(f"Found {len(search_results)} ASU Social Media results")
                    
                    if 'https://goglobal.asu.edu/scholarship-search' in search_url or 'https://onsa.asu.edu/scholarships'in search_url:
                        try:
                            # Get base domain based on URL
                            base_url = "https://goglobal.asu.edu" if "goglobal" in search_url else "https://onsa.asu.edu"
                            
                            driver.get(search_url)
                            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
                            
                            # Handle cookie consent for goglobal
                            try:
                                cookie_button = WebDriverWait(driver, 5).until(
                                    EC.element_to_be_clickable((By.CSS_SELECTOR, '.accept-btn'))
                                )
                                driver.execute_script("arguments[0].click();", cookie_button)
                                time.sleep(2)
                            except Exception as cookie_error:
                                logger.warning(f"Cookie consent handling failed: {cookie_error}")
                            
                            if optional_query:
                                logger.info("\nOptional query :: %s" % optional_query)
                                
                                # Parse query parameters
                                query_params = dict(param.split('=') for param in optional_query.split('&') if '=' in param)
                                
                                # Define filter mappings based on site
                                filter_mapping = {
                                    'goglobal.asu.edu': {
                                        'academiclevel': '#edit-field-ss-student-type-target-id',
                                        'citizenship_status': '#edit-field-ss-citizenship-status-target-id',
                                        'gpa': '#edit-field-ss-my-gpa-target-id',
                                        # 'college': '#edit-field-college-ss-target-id',
                                    },
                                    'onsa.asu.edu': {
                                        'search_bar_query': 'input[name="combine"]',
                                        'citizenship_status': 'select[name="field_citizenship_status"]',
                                        'eligible_applicants': 'select[name="field_eligible_applicants"]',
                                        'focus': 'select[name="field_focus"]',
                                    }
                                }
                                
                                # Determine which site's filter mapping to use
                                site_filters = filter_mapping['goglobal.asu.edu'] if 'goglobal.asu.edu' in search_url else filter_mapping['onsa.asu.edu']
                                
                                # Apply filters with robust error handling
                                for param, value in query_params.items():
                                    if param in site_filters and value:
                                        try:
                                            filter_element = WebDriverWait(driver, 10).until(
                                                EC.element_to_be_clickable((By.CSS_SELECTOR, site_filters[param]))
                                            )
                                            
                                            # Scroll element into view
                                            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", filter_element)
                                            time.sleep(1)
                                            
                                            # Handle different input types
                                            if filter_element.tag_name == 'select':
                                                Select(filter_element).select_by_visible_text(value)
                                            elif filter_element.tag_name == 'input':
                                                filter_element.clear()
                                                filter_element.send_keys(value)
                                                filter_element.send_keys(Keys.ENTER)
                                            
                                            time.sleep(1)
                                        except Exception as filter_error:
                                            logger.warning(f"Could not apply filter {param}: {filter_error}")
                                
                                # Click search button with multiple retry mechanism
                                search_button_selectors = ['input[type="submit"]', 'button[type="submit"]', '.search-button']
                                for selector in search_button_selectors:
                                    try:
                                        search_button = WebDriverWait(driver, 10).until(
                                            EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                                        )
                                        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", search_button)
                                        time.sleep(1)
                                        driver.execute_script("arguments[0].click();", search_button)
                                        break
                                    except Exception as e:
                                        logger.warning(f"Search button click failed for selector {selector}: {e}")
                            
                            # Extract scholarship links with improved URL construction
                            link_selectors = {
                                'goglobal': 'td[headers="view-title-table-column"] a',
                                'onsa': 'td a'
                            }
                            
                            current_selector = link_selectors['goglobal'] if "goglobal" in search_url else link_selectors['onsa']
                            
                            scholarship_links = WebDriverWait(driver, 10).until(
                                EC.presence_of_all_elements_located((By.CSS_SELECTOR, current_selector))
                            )
                            
                            for link in scholarship_links[:5]:
                                href = link.get_attribute('href')
                                if href:
                                    if href.startswith('/'):
                                        search_results.append(f"{base_url}{href}")
                                    elif href.startswith('http'):
                                        search_results.append(href)
                                    else:
                                        search_results.append(f"{base_url}/{href}")
                            
                            logger.info(f"Found {len(search_results)} scholarship links - ")
                            logger.info(f"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
                            logger.info(f"{search_results}")
                            logger.info(f"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
                            
                        except Exception as e:
                            logger.error(f"Error in scholarship search: {str(e)}")
                            try:
                                driver.save_screenshot(f"error_screenshot_{int(time.time())}.png")
                            except:
                                pass
                    
                    if 'catalog.apps.asu.edu' in search_url or  'search.lib.asu.edu' in search_url :
                        await self.scrape_content(search_url, selenium=True)
                        return self.text_content
                    
                    if 'https://app.joinhandshake.com/stu/postings' in search_url or 'lib.asu.edu' in search_url or "asu.libcal.com" in search_url or "asu-shuttles.rider.peaktransit.com" in search_url:                    
                        await self.scrape_content(search_url, selenium=True, optional_query=optional_query)
                        return self.text_content
                        
                finally:
                    driver.quit()
                
                for url in search_results:
                    await self.scrape_content(url=url)
            
            return self.text_content
                
        except Exception as e:
            logger.error(f"Error in search: {str(e)}")
            return []

# %% [markdown]
# ### Global Instance

# %%
asu_scraper = ASUWebScraper()


logger.info("\nInitialized ASUWebScraper")

# %% [markdown]
# ## Setting up Gemini Agents
# 
# This section outlines the setup for Gemini Agents, focusing on defining function parameters for tool functions and establishing a global instance for action commands.

# %% [markdown]
# 
# 
# ### Types of Function Parameters
# 
# #### String Parameter
# For a simple string input:
# 
# ```python
# "search_bar_query": content.Schema(
#     type=content.Type.STRING,
#     description="Optional search query to filter social media posts"
# ),
# ```
# 
# #### Array Parameter
# For an array of string options:
# 
# ```python
# "class_subject": content.Schema(
#     type=content.Type.ARRAY,
#     items=content.Schema(
#         type=content.Type.STRING,
#         enum=[
#             "@ArizonaState", 
#             "@SunDevilAthletics", 
#             "@SparkySunDevil", 
#             "@SunDevilFootball", 
#             "@ASUFootball", 
#             "@SunDevilFB"
#         ]
#     ),
#     description="Pick from the List of ASU social media account names to search"
# ),
# ```
# 
# #### Enumerated String Parameter
# For a string parameter with predefined options:
# 
# ```python
# "class_subject": content.Schema(
#     type=content.Type.STRING,
#     description="Program Type",
#     enum=[
#         "Exchange",
#         "Faculty-Directed",
#         "Global Intensive Experience",
#         "Partnership",
#     ]
# )
# ```
# 
# 

# %% [markdown]
# when tool function parameter is a string - 
# 
# ```
# "search_bar_query": content.Schema(
#     type=content.Type.STRING,
#     description="Optional search query to filter social media posts"
# ),
# ```
# when tool function parameter is an array - 
# ```
# "class_subject": content.Schema(
#     type=content.Type.ARRAY,
#     items=content.Schema(
#         type=content.Type.STRING,
#         enum=[
#             "@ArizonaState", 
#             "@SunDevilAthletics", 
#             "@SparkySunDevil", 
#             "@SunDevilFootball", 
#             "@ASUFootball", 
#             "@SunDevilFB"
#         ]
#     ),
#     description="Pick from the List of ASU social media account names to search"
# ),
# ```
# when tool function parameter is a string and you want model to pick from something - 
# ```
# "class_subject": content.Schema(
#     type=content.Type.STRING,
#     description="Program Type",
#     enum=[
#         "Exchange",
#         "Faculty-Directed",
#         "Global Intensive Experience",
#         "Partnership",
#     ]
# )
# ```

# %% [markdown]
# ### Global Action Command Instance
# 
# To maintain action commands across different agents:
# 
# 

# %%
global action_command
action_command = None


logger.info("\nInitialized ActionCommands")

# %% [markdown]
# ### Data Model
# 
# This `DataModel` class serves as a crucial component for enhancing the quality and structure of text data, particularly useful in the context of the ASU Discord Research Assistant Bot for refining search results and improving information presentation.
# 

# %% [markdown]
# #### Initialization
# - The class is initialized with an optional `model` parameter, which is expected to be an instance of Google's Gemini model.
# 
# #### Text Refinement
# - The `refine` method is the core function for processing text:
#   - It takes a `search_context` and `text` as input.
#   - Constructs a prompt using a predefined template from `app_config.get_data_agent_prompt()`.
#   - Sends the prompt to the Gemini model for processing.
# 
# #### Response Handling
# - The method attempts to generate content using the Gemini model.
# - If successful, it parses the JSON response using the `parse_json_response` method.
# - Returns the refined title extracted from the parsed response.
# 
# #### Error Handling
# - Implements comprehensive error logging for various scenarios:
#   - Gemini model errors
#   - JSON parsing errors
#   - Missing required fields in the response
# 
# #### JSON Parsing
# - The `parse_json_response` method processes the model's output:
#   - Cleans the response text by removing markdown code block indicators.
#   - Parses the cleaned text as JSON.
#   - Validates the presence of required fields (currently only 'title').
#   - Returns a dictionary with the parsed data or default values if parsing fails.
# 
# 

# %%
class DataModel:
    def __init__(self, model=None):
        self.model = model
  
    def refine(self, search_context: str, text: str) -> tuple[str, str, str]:
        prompt = f"""{app_config.get_data_agent_prompt()}        
        Search Context: {search_context}
        Input Text: {text}

        """

        try:
            logger.info(f"Data Model: Refining Data with context : {search_context} \n and data : {text}")
            response = self.model.generate_content(prompt)
            if response and hasattr(response, 'text'):
                parsed = self.parse_json_response(response.text)
                return (
                    # parsed.get('refined_content', ''),
                    parsed.get('title', ''),
                )
            return None, None, None
        except Exception as e:
            logger.error(f"Gemini refinement error: {str(e)}")
            return None, None, None

    def parse_json_response(self, response_text: str) -> dict:
        """Parse the JSON response into components."""
        try:
            # Remove any potential markdown code block indicators
            cleaned_response = response_text.replace('```json', '').replace('```', '').strip()

            # Parse the JSON string into a dictionary
            parsed_data = json.loads(cleaned_response)

            # Validate required fields
            required_fields = { 'title'}
            if not all(field in parsed_data for field in required_fields):
                logger.error("Missing required fields in JSON response")
                return { 'title': ''}

            return parsed_data

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {str(e)}")
            return { 'title': '', 'category': ''}
        except Exception as e:
            logger.error(f"Unexpected error parsing response: {str(e)}")
            return {'title': '', 'category': ''}

# %% [markdown]
# #### Global Instance 

# %%
asu_data_agent = DataModel(dir_Model)

logger.info("\nInitialized DataModel")


# %% [markdown]
# ### Live_Status Model
# 
# The `Live_Status_Model` class is designed to handle real-time status queries for ASU services, particularly focusing on library and shuttle statuses. 

# %% [markdown]
# #### Initialization
# - Initializes with a model (`live_status_model`), functions (`Live_Status_Model_Functions`), and conversation tracking.
# - Implements rate limiting and request counting.
# 
# #### Function Execution
# - `execute_function` method dynamically calls the appropriate function based on the function name.
# - Supports `get_live_library_status` and `get_live_shuttle_status` functions.
# - Implements error handling and logging for function execution.
# 
# #### Model Initialization
# - `_initialize_model` method sets up the chat model with automatic function calling enabled.
# - Implements rate limiting to prevent excessive requests.
# 
# #### Action Determination
# - `determine_action` method processes user queries and special instructions.
# - Constructs a prompt with current context, user query, and agent instructions.
# - Handles both text responses and function calls from the model.
# 
# #### Response Processing
# - Iterates through model response parts, handling both text and function calls.
# - Executes functions when called and formats the final response.
# 
# #### Error Handling
# - Comprehensive error logging throughout the class.
# - Fallback responses for various error scenarios.
# 
# 

# %%
class Live_Status_Model_Functions:
    def __init__(self):
        self.visited_urls = set()
        self.max_depth = 2
        self.max_links_per_page = 3
                
    async def get_live_library_status(self, status_type : [] = None, date : str = None, library_names: [] = None):
        """
        Retrieve ASU Library Status using ASU Library Search with robust parameter handling.

        Args:
            status_type ([]): Open or Close & Study Room Availability.
            library_names ([], optional): Name of library.
            date (str): Dec 01 (Month_prefix + Date).

        Returns:
            Union[str, dict]: Search results or error message.
            
        Notes:
        Example URL- 
        https://asu.libcal.com/r/accessible/availability?lid=13858&gid=28619&zone=0&space=0&capacity=2&date=2024-12-04
        """
        
        if not (library_names or status_type or date):
            return "Error: Atleast one parameter required"
        
        search_url=None
        query=None
        result =""
        doc_title = " ".join(library_names)
        if "Availability" in status_type:
            search_url=f"https://lib.asu.edu/hours"
            query = f"library_names={library_names}&date={date}"
            result+=await utils.perform_web_search(search_url, query,doc_title=doc_title, doc_category ="libraries_status")
        
        library_map = {
            "Tempe Campus - Hayden Library": "13858",
            "Tempe Campus - Noble Library": "1702",
            "Downtown Phoenix Campus - Fletcher Library": "1703",
            "West Campus - Library": "1707",
            "Polytechnic Campus - Library": "1704"
        }
        
        gid_map={
            "13858": "28619",
             "1702": "2897",
            "1703": "2898",
            "1707": "28611",
            "1704": "2899"
        }
             
        if "StudyRoomsAvailability" in status_type:
            transformed_date = datetime.strptime(date, '%b %d').strftime('2024-%m-%d')
            for library in library_names:
                query= library_map[library]
                search_url = f"https://asu.libcal.com/r/accessible/availability?lid={library_map[library]}&gid={gid_map[library_map[library]]}&zone=0&space=0&capacity=2&date={transformed_date}"
                result+=await utils.perform_web_search(search_url, query,doc_title=doc_title, doc_category ="libraries_status")
            
        return result
        
    async def get_live_shuttle_status(self, shuttle_route: [] = None):
        if not shuttle_route:
            return "Error: At least one route is required"
        
        shuttle_route = set(shuttle_route)
        
        doc_title = " ".join(shuttle_route)
        search_url="https://asu-shuttles.rider.peaktransit.com/"

        logger.info(shuttle_route)
        
        if len(shuttle_route) == 1:
            logger.info("\nOnly one route")
            route = next(iter(shuttle_route))
            return await utils.perform_web_search(search_url, optional_query=route,doc_title=doc_title, doc_category ="shuttles_status")

        # Multiple routes handling
        result = ""
        try:
            for route in shuttle_route:
                result += await utils.perform_web_search(search_url, optional_query=route,doc_title=doc_title, doc_category ="shuttles_status")
            logger.info("\nDone")
            return result
        except Exception as e:
            return f"Error performing shuttle search: {str(e)}"

# %% [markdown]
# #### Initializing Functions

# %%
class Live_Status_Model:
    
    def __init__(self):
        self.model = live_status_model
        self.chat=None
        self.functions = Live_Status_Model_Functions()
        self.last_request_time = time.time()
        self.request_counter = 0
        self.conversations: Dict[str, List[Dict[str, str]]] = {}
        
    async def execute_function(self, function_call):
        """Execute the called function and return its result"""
        function_name = function_call.name
        function_args = function_call.args
        
        function_mapping = {
            
            'get_live_library_status': self.functions.get_live_library_status,
            'get_live_shuttle_status': self.functions.get_live_shuttle_status,
        }
        
            
        if function_name in function_mapping:
            function_to_call = function_mapping[function_name]
            func_response = await function_to_call(**function_args)
            # response = await self.chat.send_message_async(f"{function_name} response : {func_response}")
            logger.info(f"Live Status : Function loop response : {func_response}")
            
            if func_response:
                return func_response
            else:
                logger.error(f"Error extracting text from response: {e}")
                return "Error processing response"
            
            
        else:
            raise ValueError(f"Unknown function: {function_name}")
        
    def _initialize_model(self):
        if not self.model:
            return logger.error("Model not initialized at ActionFunction")
            
        # Rate limiting check
        current_time = time.time()
        if current_time - self.last_request_time < 1.0: 
            raise Exception("Rate limit exceeded")
            
        self.last_request_time = current_time
        self.request_counter += 1
        user_id = discord_state.get("user_id")
        self.chat = self.model.start_chat(history=self._get_chat_history(user_id),enable_automatic_function_calling=True)

    def _get_chat_history(self, user_id: str) -> List[Dict[str, str]]:
        return self.conversations.get(user_id, [])

    def _save_message(self, user_id: str, role: str, content: str):
        if user_id not in self.conversations:
            self.conversations[user_id] = []
        
        self.conversations[user_id].append({
            "role": role,
            "parts": [{"text": content}]
        })
        
        # Limit the conversation length to 3 messages per user
        if len(self.conversations[user_id]) > 3:
            self.conversations[user_id].pop(0)
        
    async def determine_action(self, query: str,special_instructions:str) -> str:
        """Determines and executes the appropriate action based on the user query"""
        try:
            self._initialize_model()
            user_id = discord_state.get("user_id")
            self._save_message(user_id, "user", query)
            final_response = ""
            
            global action_command
            action_command = query

            prompt = f"""
                ### Context:
                - Current Date and Time: {datetime.now().strftime('%H:%M %d') + ('th' if 11<=int(datetime.now().strftime('%d'))<=13 else {1:'st',2:'nd',3:'rd'}.get(int(datetime.now().strftime('%d'))%10,'th')) + datetime.now().strftime(' %B, %Y') }
                - Superior Agent Instruction: {action_command}
                - Superior Agent Remarks: {special_instructions}

                {app_config.get_live_status_agent_prompt()}
                
                """

            response = await self.chat.send_message_async(prompt)
            logger.info(self._get_chat_history)
            self._save_message(user_id, "model", f"{response.parts}" )
            logger.info(f"Internal response @ Live Status Model : {response}")
            
            for part in response.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    
                    final_response = await self.execute_function(part.function_call)
                    firestore.update_message("live_status_agent_message", f"Function called {part.function_call}\n Function Response {final_response} ")
                elif hasattr(part, 'text') and part.text.strip():
                    text = part.text.strip()
                    firestore.update_message("live_status_agent_message", f"Text Response : {text}")
                    if not text.startswith("This query") and not "can be answered directly" in text:
                        final_response = text.strip()
                        logger.info(f"text response : {final_response}")
        
        # Return only the final message
            return final_response if final_response else "Live Status agent fell off! Error 404"
            
        except Exception as e:
            logger.error(f"Internal Error @ Live Status Model : {str(e)}")
            return "I apologize, but I couldn't generate a response at this time. Please try again."
        

# %% [markdown]
# #### Setting up Model 

# %%
live_status_model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config={
        "temperature": 0.0,
        "top_p": 0.1,
        "top_k": 40,
        "max_output_tokens": 2500,
        "response_mime_type": "text/plain",
    },
    safety_settings={
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    },
    system_instruction = f"""
    {app_config.get_live_status_agent_instruction}
    """,
    tools=[
        genai.protos.Tool(
            function_declarations=[

                genai.protos.FunctionDeclaration(
                    name="get_live_library_status",
                    description="Retrieves Latest Information regarding ASU Library Status",
                    parameters=content.Schema(
                        type=content.Type.OBJECT,
                        properties={

                            "status_type": content.Schema(
                                type=content.Type.ARRAY,
                                items=content.Schema(
                                    type=content.Type.STRING,
                                    enum=[
                                        "Availability", 
                                        "StudyRoomsAvailability", 
                                    ]
                                ),
                                description="Checks if library is open or close and study rooms availability"
                            ),
                            "library_names": content.Schema(
                                type=content.Type.ARRAY,
                                items=content.Schema(
                                    type=content.Type.STRING,
                                    enum=[
                                "Tempe Campus - Noble Library",
                                "Tempe Campus - Hayden Library",
                                "Downtown Phoenix Campus - Fletcher Library",
                                "West Campus - Library",
                                "Polytechnic Campus - Library",      
                                ]
                                ),
                                description="Library Name"
                            ),
                             "date": content.Schema(
                                type=content.Type.STRING,
                                description="[ Month Prefix + Date ] (ex. DEC 09, JAN 01, FEB 21, MAR 23)",
                            ),
                        },
                            required=["status_type","library_names","date"]
                    )
                ),
                genai.protos.FunctionDeclaration(
                    name="get_live_shuttle_status",
                    description="Searches for shuttle status and routes",
                    parameters=content.Schema(
                        type=content.Type.OBJECT,
                        properties={
                            "shuttle_route": content.Schema(
                                type=content.Type.ARRAY,
                                items=content.Schema(
                                    type=content.Type.STRING,
                                    enum=[
                                        "Mercado", 
                                        "Polytechnic-Tempe", 
                                        "Tempe-Downtown Phoenix-West", 
                                        "Tempe-West Express", 
                                    ]
                                ),
                                description="The Route of Buses"
                            ),
                        },
                    required= ["shuttle_route"]
                    ),
                ),
            ],
        ),
    ],
    tool_config={'function_calling_config': 'ANY'},
)



# %% [markdown]
# #### Global Instance 

# %%
asu_live_status_agent = Live_Status_Model()
logger.info("\nInitialized LiveStatusAgent")

# %% [markdown]
# ### Search Model
# 
# The `SearchModel` class is designed to handle complex search operations using Google's Generative AI model. 
# 

# %% [markdown]
# 
# #### Initialization
# - Initializes with configurable rate limiting parameters:
#   - `rate_limit_window`: Time window for rate limiting
#   - `max_requests`: Maximum number of requests allowed in the window
#   - `retry_attempts`: Number of retry attempts for function calls
# - Sets up conversation tracking and request counting
# 
# #### Function Execution
# - `execute_function` method dynamically calls the appropriate function based on the function name
# - Supports various search functions like Google search, accessing deep search agent, and retrieving updates for clubs, events, news, sports, and social media
# 
# #### Model Configuration
# - Uses the "gemini-1.5-flash" model with specific generation config settings
# - Implements safety settings to block low and above levels of hate speech and harassment
# 
# #### Search Functionality
# - `determine_action` method processes user queries and special instructions
# - Constructs prompts with current context, user query, and agent instructions
# - Handles both text responses and function calls from the model
# 
# #### Rate Limiting and Retry Mechanism
# - Implements a sophisticated rate limiting system to prevent excessive requests
# - Includes a retry mechanism for failed function calls
# 
# #### Error Handling
# - Comprehensive error logging throughout the class
# - Fallback responses for various error scenarios
# 
# 

# %% [markdown]
# #### Setting up Model functions

# %%
class Search_Model_Functions:
    def __init__(self):
        self.visited_urls = set()
        self.max_depth = 2
        self.max_links_per_page = 3
        
    async def get_latest_club_information(self, search_bar_query: str = None, organization_category: list = None, organization_campus: list = None):
        if not any([search_bar_query, organization_category, organization_campus]):
            return "At least one parameter of this function is required. Neither Search query and organization category and organization campus received. Please provide at least one parameter to perform search."
        
        search_url = "https://asu.campuslabs.com/engage/organizations"
        params = []
        organization_campus_ids = { "ASU Downtown":"257211",
                                   "ASU Online":"257214",
                                   "ASU Polytechnic":"257212",
                                   "ASU Tempe":"254417",
                                   "ASU West Valley":"257213",
                                   "Fraternity & Sorority Life":"257216",
                                   "Housing & Residential Life":"257215"}
        
        organization_category_ids = {"Academic":"13382","Barrett":"14598","Creative/Performing Arts":"13383","Cultural/Ethnic":"13384","Distinguished Student Organization":"14549","Fulton Organizations":"14815","Graduate":"13387","Health/Wellness":"13388","International":"13389","LGBTQIA+":"13391","Political":"13392","Professional":"13393","Religious/Faith/Spiritual":"13395","Service":"13396","Social Awareness":"13398","Special Interest":"13399","Sports/Recreation":"13400","Sustainability":"13402","Technology":"13403", "Veteran Groups":"14569","W.P. Carey Organizations":"14814","Women":"13405"}
        doc_title=""
        if search_bar_query:
            doc_title = search_bar_query
        elif organization_category:
            doc_title = " ".join(organization_category)
        elif organization_campus:
            doc_title = " ".join(organization_campus)
        else:
            doc_title = None

 
        if organization_campus:
            campus_id_array = [organization_campus_ids[campus] for campus in organization_campus if campus in organization_campus_ids]
            if campus_id_array:
                params.extend([f"branches={campus_id}" for campus_id in campus_id_array])
        
        if organization_category:
            category_id_array = [organization_category_ids[category] for category in organization_category if category in organization_category_ids]
            if category_id_array:
                params.extend([f"categories={category_id}" for category_id in category_id_array])
        
        if search_bar_query:
            params.append(f"query={search_bar_query.lower().replace(' ', '%20')}")
        
        if params:
            search_url += "?" + "&".join(params)
        
        return await utils.perform_web_search(search_url, doc_title=doc_title, doc_category ="clubs_info")
        
    async def get_latest_event_updates(self, search_bar_query: str = None, event_category: list = None, 
                               event_theme: list = None, event_campus: list = None, 
                               shortcut_date: str = None, event_perk: list = None):
        
        if not any([search_bar_query, event_category, event_theme, event_campus]):
            return "At least one parameter of this function is required. Neither Search query and organization category and organization campus received. Please provide at least one parameter to perform search."
        
        search_url = "https://asu.campuslabs.com/engage/events"
        params = []
        
        event_campus_ids = {
            "ASU Downtown": "257211",
            "ASU Online": "257214",
            "ASU Polytechnic": "257212",
            "ASU Tempe": "254417",
            "ASU West Valley": "257213",
            "Fraternity & Sorority Life": "257216",
            "Housing & Residential Life": "257215"
        }
        
        event_category_ids = {
            "ASU New Student Experience": "18002",
            "ASU Sync": "15695",
            "ASU Welcome Event": "12897",
            "Barrett Student Organization": "12902",
            "Black History Month": "21730",
            "C3": "19049",
            "Career and Professional Development": "12885",
            "Change The World": "12887",
            "Changemaker Central": "12886",
            "Civic Engagement": "17075",
            "Club Meetings": "12887",
            "Clubs and Organization Information": "12888",
            "Community Service": "12903",
            "Cultural Connections and Multicultural community of Excellence": "21719",
            "Culture @ ASU": "12898",
            "DeStress Fest": "19518",
            "Entrepreneurship & Innovation": "17119",
            "General": "12889",
            "Graduate": "12906",
            "Hispanic Heritage Month": "21723",
            "Homecoming": "20525",
            "In-Person Event": "17447",
            "International": "12899",
            "Memorial Union & Student Pavilion Programs": "12900",
            "Multicultural community of Excellence": "19389",
            "PAB Event": "12890",
            "Salute to Service": "12891",
            "Student Engagement Event": "12892",
            "Student Organization Event": "12893",
            "Sun Devil Athletics": "12894",
            "Sun Devil Civility": "12901",
            "Sun Devil Fitness/Wellness": "12895",
            "Sustainability": "12905",
            "University Signature Event": "12904",
            "W.P. Carey Event": "17553"
        }
        
        event_theme_ids = {
            "Arts": "arts",
            "Athletics": "athletics",
            "Community Service": "community_service",
            "Cultural": "cultural",
            "Fundraising": "fundraising",
            "GroupBusiness": "group_business",
            "Social": "social",
            "Spirituality": "spirituality",
            "ThoughtfulLearning": "thoughtful_learning"
        }
        
        event_perk_ids = {
            "Credit": "Credit",
            "Free Food": "FreeFood",
            "Free Stuff": "FreeStuff"
        }
        
        if event_campus:
            campus_id_array = [event_campus_ids[campus] for campus in event_campus if campus in event_campus_ids]
            if campus_id_array:
                params.extend([f"branches={campus_id}" for campus_id in campus_id_array])
        
        if event_category:
            category_id_array = [event_category_ids[category] for category in event_category if category in event_category_ids]
            if category_id_array:
                params.extend([f"categories={category_id}" for category_id in category_id_array])
        
        if event_theme:
            theme_id_array = [event_theme_ids[theme] for theme in event_theme if theme in event_theme_ids]
            if theme_id_array:
                params.extend([f"themes={theme_id}" for theme_id in theme_id_array])
        
        if event_perk:
            perk_id_array = [event_perk_ids[perk] for perk in event_perk if perk in event_perk_ids]
            if perk_id_array:
                params.extend([f"perks={perk_id}" for perk_id in perk_id_array])
        
        if shortcut_date:
            valid_dates = ["tomorrow", "this_weekend"]
            if shortcut_date.lower() in valid_dates:
                params.append(f"shortcutdate={shortcut_date.lower()}")
        
        if search_bar_query:
            params.append(f"query={search_bar_query.lower().replace(' ', '%20')}")
        
        if params:
            search_url += "?" + "&".join(params)
        
        doc_title=""
        if search_bar_query:
            doc_title = search_bar_query
        elif event_category:
            doc_title = " ".join(event_category)
        elif event_theme:
            doc_title = " ".join(event_theme)
        elif event_campus:
            doc_title = " ".join(event_campus)
        elif shortcut_date:
            doc_title = shortcut_date
        elif event_perk:
            doc_title = " ".join(event_perk)
        else:
            doc_title = None
        
        return await utils.perform_web_search(search_url, doc_title=doc_title, doc_category = "events_info")
        
    async def get_latest_news_updates(self, news_campus : list = None, search_bar_query: str = None,):
        if not any([search_bar_query, news_campus]):
            return "At least one parameter of this function is required. Neither Search query and news campus received. Please provide at least one parameter to perform search."
        
        search_url = "https://asu.campuslabs.com/engage/organizations"
        params = []
        news_campus_ids = { "ASU Downtown":"257211","ASU Online":"257214","ASU Polytechnic":"257212","ASU Tempe":"254417","ASU West Valley":"257213","Fraternity & Sorority Life":"257216","Housing & Residential Life":"257215"}
        doc_title=""
        if search_bar_query:
            doc_title = search_bar_query
        elif news_campus:
            doc_title = " ".join(news_campus)
        else:
            doc_title = None

        if news_campus:
            campus_id_array = [news_campus_ids[campus] for campus in news_campus if campus in news_campus_ids]
            if campus_id_array:
                params.extend([f"branches={campus_id}" for campus_id in campus_id_array])
                
        if search_bar_query:
            params.append(f"query={search_bar_query.lower().replace(' ', '%20')}")
        
        if params:
            search_url += "?" + "&".join(params)
        
        return await utils.perform_web_search(search_url,doc_title=doc_title, doc_category="news_info")
    
    async def get_latest_social_media_updates(self,  account_name: list, search_bar_query: str = None,):
        if not any([search_bar_query, account_name]):
            return "At least one parameter of this function is required. Neither Search query and news campus received. Please provide at least one parameter to perform search."
        doc_title=""
        if search_bar_query:
            doc_title = search_bar_query
        elif account_name:
            doc_title = " ".join(account_name)
        else:
            doc_title = None
        account_OBJECTs = {
            "@ArizonaState": [
                "https://x.com/ASU", 
                "https://www.instagram.com/arizonastate"
            ],
            "@SunDevilAthletics": [
                "https://x.com/TheSunDevils", 
            ],
            "@SparkySunDevil": [
                "https://x.com/SparkySunDevil", 
                "https://www.instagram.com/SparkySunDevil"
            ],
            "@ASUFootball": [
                "https://x.com/ASUFootball", 
                "https://www.instagram.com/sundevilfb/"
            ],
            "@ASUFootball": [
                "https://x.com/ASUFootball", 
                "https://www.instagram.com/sundevilfb/"
            ],
        }
        
        # Collect URLs for specified account names
        final_search_array = []
        for name in account_name:
            if name in account_OBJECTs:
                final_search_array.extend(account_OBJECTs[name])
        
        # If no URLs found for specified accounts, return empty list
        if not final_search_array:
            return []
        
        # Perform web search on each URL asynchronously
        search_results = []
        for url in final_search_array:
            search_result = await utils.perform_web_search(url,search_bar_query,doc_title=doc_title, doc_category="social_media_updates")
            search_results.extend(search_result)
        return search_results
    
    async def get_latest_sport_updates(self, search_bar_query: str = None, sport: str = None, league: str = None, match_date: str = None):
        """
        Comprehensive function to retrieve ASU sports updates using multiple data sources.
        
        Args:
            query: General search query for sports updates
            sport: Specific sport to search
            league: Sports league
            match_date: Specific match date
        
        
        Returns:
            List of sports updates or detailed information
        """
        # Validate input parameters
        if not any([search_bar_query, sport, league, match_date]):
            return "Please provide at least one parameter to perform the search."
        
        
        dynamic_query = f"ASU {sport if sport else ''} {search_bar_query if search_bar_query else ''} {league if league else ''} {match_date} site:(sundevils.com OR espn.com) -inurl:video"
        search_url=f"https://www.google.com/search?q={urllib.parse.quote(dynamic_query)}"
         
        
        return await utils.perform_web_search(search_url)

    async def get_library_resources(self, search_bar_query: str = None, resource_type: str = 'All Items'):
        """
        Retrieve ASU Library resources using ASU Library Search with robust parameter handling.

        Args:
            search_bar_query (str, optional): Search term for library resources.
            resource_type (str, optional): Type of resource to search. Defaults to 'All Items'.

        Returns:
            Union[str, dict]: Search results or error message.
        """
        # Comprehensive input validation with improved error handling
        if not search_bar_query:
            return "Error: Search query is required."
        
        # Use class-level constants for mappings to improve maintainability
        RESOURCE_TYPE_MAPPING = {
            'All Items': 'any', 'Books': 'books', 'Articles': 'articles', 
            'Journals': 'journals', 'Images': 'images', 'Scores': 'scores', 
            'Maps': 'maps', 'Sound recordings': 'audios', 'Video/Film': 'videos'
        }
        
        # Validate resource type and language with more graceful handling
        resource_type = resource_type if resource_type in RESOURCE_TYPE_MAPPING else 'All Items'
        
        # URL encode the search query to handle special characters
        encoded_query = urllib.parse.quote(search_bar_query)
        
        # Construct search URL with more robust parameter handling
        search_url = (
            f"https://search.lib.asu.edu/discovery/search"
            f"?query=any,contains,{encoded_query}"
            f",AND&pfilter=rtype,exact,{RESOURCE_TYPE_MAPPING[resource_type]}"
            "&tab=LibraryCatalog"
            "&search_scope=MyInstitution"
            "&vid=01ASU_INST:01ASU"
            "&lang=en"
            "&mode=advanced"
            "&offset=0"
        )
        doc_title=""
        if search_bar_query:
            doc_title = search_bar_query
        elif resource_type:
            doc_title = resource_type
        else:
            doc_title = None
        try:
            # Add error handling for web search
            return await utils.perform_web_search(search_url,doc_title=doc_title, doc_category ="library_resources")
        except Exception as e:
            return f"Error performing library search: {str(e)}"
        
    async def get_latest_scholarships(self, search_bar_query: str = None, academic_level:str = None,eligible_applicants: str =None, citizenship_status: str = None, gpa: str = None, focus : str = None):
    
        if not any([search_bar_query, academic_level, citizenship_status, gpa,  eligible_applicants, focus]):
            return "Please provide at least one parameter to perform the search."
        
        results =[]
        doc_title = ""
        if search_bar_query:
            doc_title = search_bar_query
        elif academic_level:
            doc_title = academic_level
        elif citizenship_status:
            doc_title = citizenship_status
        # elif college:
        #     doc_title = college
        elif focus:
            doc_title = focus
        else:
            doc_title = None
            
        
        search_url = f"https://goglobal.asu.edu/scholarship-search"
        
        query = f"academiclevel={academic_level}&citizenship_status={citizenship_status}&gpa={gpa}"
        # &college={college}
        
        result = await utils.perform_web_search(search_url,query, doc_title=doc_title, doc_category ="scholarships_info")
        
        
        results.append(result)
        
        
        search_url = f"https://onsa.asu.edu/scholarships"
        
        query = f"search_bar_query={search_bar_query}&citizenship_status={citizenship_status}&eligible_applicants={eligible_applicants}&focus={focus}"
        
        
        
        result = await utils.perform_web_search(search_url,query, doc_title=doc_title,doc_category ="scholarships_info")
        
        
        results.append(result)
        
            
        
        return results
       
    async def get_latest_job_updates( self, search_bar_query: Optional[Union[str, List[str]]] = None, job_type: Optional[Union[str, List[str]]] = None, job_location: Optional[Union[str, List[str]]] = None):
        """
        Comprehensive function to retrieve ASU Job updates using multiple data sources.
        
        Args:
            Multiple search parameters for job filtering with support for both string and list inputs
        
        Returns:
            List of search results
        """
        # Helper function to normalize input to list
        def normalize_to_list(value):
            if value is None:
                return None
            return value if isinstance(value, list) else [value]
        
        # Normalize all inputs to lists
        query_params = {
            'search_bar_query': normalize_to_list(search_bar_query),
            'job_type': normalize_to_list(job_type),
            'job_location': normalize_to_list(job_location),
        }
        
        # Remove None values
        query_params = {k: v for k, v in query_params.items() if v is not None}
        
        # Validate that at least one parameter is provided
        if not query_params:
            return "Please provide at least one parameter to perform the search."
        
        # Convert query parameters to URL query string
        # Ensure each parameter is converted to a comma-separated string if it's a list
        query_items = []
        for k, v in query_params.items():
            if isinstance(v, list):
                query_items.append(f"{k}={','.join(map(str, v))}")
            else:
                query_items.append(f"{k}={v}")
        
        query = '&'.join(query_items)
        
        search_url = "https://app.joinhandshake.com/stu/postings"
        
        results = []
        logger.info(f"Requested search query : {query}")
        doc_title = ""
        if search_bar_query:
            doc_title = " ".join(search_bar_query) if isinstance(search_bar_query, list) else search_bar_query
        elif job_type:
            doc_title = " ".join(job_type) if isinstance(job_type, list) else job_type
        elif job_location:
            doc_title = " ".join(job_location) if isinstance(job_location, list) else job_location
        else:
            doc_title = None
            
        result = await utils.perform_web_search(search_url, query, doc_title=doc_title, doc_category ="job_updates")
        results.append(result)
        
        return results       
    
    async def get_latest_class_information(self,search_bar_query: Optional[str] = None,class_term: Optional[str] = None,subject_name: Optional[Union[str, List[str]]] = None, 
    num_of_credit_units: Optional[Union[str, List[str]]] = None, 
    class_level: Optional[Union[str, List[str]]] = None,
    class_session: Optional[Union[str, List[str]]] = None,
    class_days: Optional[Union[str, List[str]]] = None,
    class_location: Optional[Union[str, List[str]]] = None,
    class_seat_availability : Optional[str] = None,
    ) -> str:
        """
        Optimized function to generate a search URL for ASU class catalog with flexible input handling.
        
        Args:
            Multiple optional parameters for filtering class search
        
        Returns:
            Constructed search URL for class catalog
        """
        
        # Helper function to convert input to query string
        
        
        
        DAYS_MAP = {
            'Monday': 'MON',
            'Tuesday': 'TUES', 
            'Wednesday': 'WED', 
            'Thursday': 'THURS', 
            'Friday': 'FRI', 
            'Saturday': 'SAT', 
            'Sunday': 'SUN'
        }
        
        
        CLASS_LEVEL_MAP = {
        'Lower division': 'lowerdivision',
        'Upper division': 'upperdivision', 
        'Undergraduate': 'undergrad',
        'Graduate': 'grad',
        '100-199': '100-199',
        '200-299': '200-299',
        '300-399': '300-399',
        '400-499': '400-499'
        }
        
        SESSION_MAP = {
            'A': 'A',
            'B': 'B', 
            'C': 'C',
            'Other': 'DYN'
        }
        
       

        TERM_MAP= {
            'Spring 2025': '2251',
            'Fall 2024': '2247', 
            'Summer 2024': '2244',
            'Spring 2024': '2241',
            'Fall 2023': '2237', 
            'Summer 2023': '2234'
        }
        
        CREDIT_UNITS_MAP = {
            '0': 'Less than 1',
            '1': '1',
            '2': '2',
            '3': '3',
            '4': '4',
            '5': '5',
            '6': '6',
            '7': '7 or more'
        }


        
        unmapped_items = []
        
        def _convert_to_query_string(input_value: Optional[Union[str, List[str]]], mapping: Dict[str, str]) -> str:
            global unmapped_items
            unmapped_items = []
            
            # Handle None input
            if input_value is None:
                return ''
            
            # Ensure input is a list
            if isinstance(input_value, str):
                input_value = [input_value]
            
            # Process each input value
            mapped_values = []
            for value in input_value:
                # Check if value exists in mapping
                if value in mapping:
                    mapped_values.append(mapping[value])
                else:
                    # Add unmapped items to global list
                    unmapped_items.append(value)
            
            # Join mapped values with URL-encoded comma
            return '%2C'.join(mapped_values) if mapped_values else ''
        
        
        
        search_bar_query = (search_bar_query or '') + ' ' + ' '.join(unmapped_items)
        search_bar_query+=subject_name
        search_bar_query = search_bar_query.strip().replace(" ", "%20")
        
        
        params = {
            'advanced': 'true',
            'campus': _convert_to_query_string(class_location, LOCATION_MAP),
            'campusOrOnlineSelection': 'A',
            'daysOfWeek': _convert_to_query_string(class_days, DAYS_MAP),
            'honors': 'F',
            'keywords': search_bar_query,
            'level': _convert_to_query_string(class_level, CLASS_LEVEL_MAP),
            'promod': 'F',
            'searchType': "open" if class_seat_availability == "Open" else "all",
            'session': _convert_to_query_string(class_session, SESSION_MAP),
            'term': _convert_to_query_string(class_term, TERM_MAP),
            'units': _convert_to_query_string(num_of_credit_units, CREDIT_UNITS_MAP)
        }
        
        logger.info(params)

        # Remove None values and construct URL
        search_url = 'https://catalog.apps.asu.edu/catalog/classes/classlist?' + '&'.join(
            f'{key}={value}' 
            for key, value in params.items() 
            if value is not None and value != ''
        )
        
        doc_title=""
        if search_bar_query:
            doc_title = search_bar_query
        elif subject_name:
            doc_title = " ".join(subject_name) if isinstance(subject_name, list) else subject_name
        elif class_term:
            doc_title = class_term
        elif class_level:
            doc_title = " ".join(class_level) if isinstance(class_level, list) else class_level
        elif class_location:
            doc_title = " ".join(class_location) if isinstance(class_location, list) else class_location
        elif class_session:
            doc_title = " ".join(class_session) if isinstance(class_session, list) else class_session
        elif num_of_credit_units:
            doc_title = " ".join(num_of_credit_units) if isinstance(num_of_credit_units, list) else num_of_credit_units
        elif class_days:
            doc_title = " ".join(class_days) if isinstance(class_days, list) else class_days

        elif class_seat_availability:
            doc_title = class_seat_availability
        else:
            doc_title = None

        return await utils.perform_web_search(search_url,doc_title=doc_title, doc_category ="classes_info")

# %% [markdown]
# #### Initializing Functions

# %%
class SearchModel:
    
    def __init__(self, 
                 rate_limit_window: float = 1.0, 
                 max_requests: int = 100,
                 retry_attempts: int = 3):
        """
        Initialize SearchModel with advanced configuration options.
        
        Args:
            rate_limit_window (float): Time window for rate limiting
            max_requests (int): Maximum number of requests allowed in the window
            retry_attempts (int): Number of retry attempts for function calls
        """
        self.model = search_model
        self.chat = None
        self.functions = Search_Model_Functions()
        self.last_request_time = time.time()
        self.request_counter = 0
        self.rate_limit_window = rate_limit_window
        self.max_requests = max_requests
        self.retry_attempts = retry_attempts
        self.conversations: Dict[str, List[Dict[str, str]]] = {}
        logger.info(f"SearchModel initialized with rate limit: {rate_limit_window}s, max requests: {max_requests}")

    async def execute_function(self, function_call):
        """
        Execute the called function with comprehensive error handling and retry mechanism.
        
        Args:
            function_call: Function call OBJECT to execute
        
        Returns:
            str: Processed function response
        """
        function_name = function_call.name
        function_args = function_call.args
        
        function_mapping = {
            'get_latest_club_information': self.functions.get_latest_club_information,
            'get_latest_event_updates': self.functions.get_latest_event_updates,
            'get_latest_news_updates': self.functions.get_latest_news_updates,
            'get_latest_social_media_updates': self.functions.get_latest_social_media_updates,
            'get_latest_sport_updates': self.functions.get_latest_sport_updates,
           'get_library_resources': self.functions.get_library_resources,
              'get_latest_scholarships': self.functions.get_latest_scholarships,
            'get_latest_job_updates': self.functions.get_latest_job_updates,
            'get_latest_class_information': self.functions.get_latest_class_information
        }
        
        if function_name not in function_mapping:
            logger.error(f"Unknown function: {function_name}")
            raise ValueError(f"Unknown function: {function_name}")
        
        function_to_call = function_mapping[function_name]
        
        for attempt in range(self.retry_attempts):
            try:
                func_response = await function_to_call(**function_args)
                
                logger.info(f"Function '{function_name}' response (Attempt {attempt + 1}): {func_response}")
                
                if func_response:
                    return func_response
                
                logger.warning(f"Empty response from {function_name}")
                
            except Exception as e:
                logger.error(f"Function call error (Attempt {attempt + 1}): {str(e)}")
                
                if attempt == self.retry_attempts - 1:
                    logger.critical(f"All retry attempts failed for {function_name}")
                    return f"Error processing {function_name}"
                
                # Exponential backoff
                await asyncio.sleep(2 ** attempt)
        
        return "No valid response from function"
    
    def _initialize_model(self):
        """
        Initialize the search model with advanced rate limiting and error checking.
        """
        if not self.model:
            logger.critical("Model not initialized")
            raise RuntimeError("Search model is not configured")
        
        current_time = time.time()
        
        if current_time - self.last_request_time < self.rate_limit_window:
            self.request_counter += 1
            if self.request_counter > self.max_requests:
                wait_time = self.rate_limit_window - (current_time - self.last_request_time)
                logger.warning(f"Rate limit exceeded. Waiting {wait_time:.2f} seconds")
                raise Exception(f"Rate limit exceeded. Please wait {wait_time:.2f} seconds")
        else:
            # Reset counter if outside the rate limit window
            self.request_counter = 1
            self.last_request_time = current_time
        
        try:
            user_id = discord_state.get('user_id')
            self.chat = self.model.start_chat(history=self._get_chat_history(user_id),enable_automatic_function_calling=True)
            logger.info("\nSearch model chat session initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize chat session: {str(e)}")
            raise RuntimeError("Could not start chat session")
        
    def _get_chat_history(self, user_id: str) -> List[Dict[str, str]]:
        return self.conversations.get(user_id, [])

    def _save_message(self, user_id: str, role: str, content: str):
        if user_id not in self.conversations:
            self.conversations[user_id] = []
        
        self.conversations[user_id].append({
            "role": role,
            "parts": [{"text": content}]
        })
        
        # Limit the conversation length to 3 messages per user
        if len(self.conversations[user_id]) > 3:
            self.conversations[user_id].pop(0)

        
    async def determine_action(self, query: str,special_instructions:str) -> str:
        """
        Advanced query processing with comprehensive error handling and logging.
        
        Args:
            query (str): User query to process
        
        Returns:
            str: Processed query response
        """
        try:
            user_id = discord_state.get("user_id")
            self._initialize_model()
            final_response = ""
            self._save_message(user_id, "user", query)
            action_command = query
            
            prompt = f"""
             ### Context:
                - Current Date and Time: {datetime.now().strftime('%H:%M %d') + ('th' if 11<=int(datetime.now().strftime('%d'))<=13 else {1:'st',2:'nd',3:'rd'}.get(int(datetime.now().strftime('%d'))%10,'th')) + datetime.now().strftime(' %B, %Y') }
                - Superior Agent Instruction: {action_command}
                - Superior Agent Remarks: {special_instructions}
                {app_config.get_search_agent_prompt()}
                
                """
                
            logger.debug(f"Generated prompt: {prompt}")
            
            try:
                response = await self.chat.send_message_async(prompt)
                logger.info(self._get_chat_history)
                self._save_message(user_id, "model", f"{response.parts}" )
                for part in response.parts:
                    if hasattr(part, 'function_call') and part.function_call: 
                        final_response = await self.execute_function(part.function_call)
                        firestore.update_message("search_agent_message", f"Function called {part.function_call}\n Function Response {final_response} ")
                    elif hasattr(part, 'text') and part.text.strip():
                        text = part.text.strip()
                        firestore.update_message("search_agent_message", f"Text Response : {text} ")
                        if not text.startswith("This query") and "can be answered directly" not in text:
                            final_response = text.strip()
            
            except Exception as response_error:
                logger.error(f"Response generation error: {str(response_error)}")
                final_response = "Unable to generate a complete response"
            
            return final_response or "Search agent encountered an unexpected issue"
        
        except Exception as critical_error:
            logger.critical(f"Critical error in determine_action: {str(critical_error)}")
            return "I'm experiencing technical difficulties. Please try again later."
        

# %% [markdown]
# #### Setting up Model 

# %%
search_model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config={
        "temperature": 0.0,
        "top_p": 0.1,
        "top_k": 40,
        "max_output_tokens": 2500,
        "response_mime_type": "text/plain",
    },
    safety_settings={
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    },
    system_instruction = f"""
    {app_config.get_search_agent_instruction()}
    """,

    tools=[
        genai.protos.Tool(
            function_declarations=[
            
                genai.protos.FunctionDeclaration(
                    name="get_latest_club_information",
                    description="Searches for clubs or organizations information with Sun Devil Search Engine",
                    parameters=content.Schema(
                        type=content.Type.OBJECT,
                        properties={
                            "search_bar_query": content.Schema(
                                type=content.Type.STRING,
                                description="Search Query",
                            ),
                            "organization_campus": content.Schema(
                                type=content.Type.ARRAY,
                                items=content.Schema(
                                    type=content.Type.STRING,
                                ),
                                description="Club/Organization campus pick from [ASU Downtown, ASU Online, ASU Polytechnic, ASU Tempe, ASU West Valley, Fraternity & Sorority Life, Housing & Residential Life]"
                            ),
                            "organization_category": content.Schema(
                                type=content.Type.ARRAY,
                                items=content.Schema(
                                    type=content.Type.STRING,
                                ),
                                description="Club/Organization Category, pick from [Academic, Barrett, Creative/Performing Arts, Cultural/Ethnic, Distinguished Student Organization, Fulton Organizations, Graduate, Health/Wellness, International, LGBTQIA+, Political, Professional, Religious/Faith/Spiritual, Service, Social Awareness, Special Interest, Sports/Recreation, Sustainability, Technology, Veteran Groups, W.P. Carey Organizations, Women]"
                            ),
                        },
                            required=["search_bar_query", "organization_category"]
                    ),
                ),
                genai.protos.FunctionDeclaration(
                    name="get_latest_event_updates",
                    description="Searches for events information with Sun Devil Search Engine",
                    parameters=content.Schema(
                        type=content.Type.OBJECT,
                        properties={
                            "search_bar_query": content.Schema(
                                type=content.Type.STRING,
                                description="Search Query"
                            ),
                            "event_campus": content.Schema(
                                type=content.Type.ARRAY,
                                items=content.Schema(
                                    type=content.Type.STRING,
                                ),
                                description="Event campus pick from [ASU Downtown, ASU Online, ASU Polytechnic, ASU Tempe, ASU West Valley, Fraternity & Sorority Life, Housing & Residential Life]"
                            ),
                            "event_category": content.Schema(
                                type=content.Type.ARRAY,
                                items=content.Schema(
                                    type=content.Type.STRING,
                                ),
                                description="Event Category, pick from [ASU New Student Experience, ASU Sync, ASU Welcome Event, Barrett Student Organization, Career and Professional Development, Club Meetings, Community Service, Cultural, DeStress Fest, Entrepreneurship & Innovation, Graduate, International, Social, Sports/Recreation, Sustainability]"
                            ),
                            "event_theme": content.Schema(
                                type=content.Type.ARRAY,
                                items=content.Schema(
                                    type=content.Type.STRING,
                                ),
                                description="Event Theme, pick from [Arts, Athletics, Community Service, Cultural, Fundraising, GroupBusiness, Social, Spirituality, ThoughtfulLearning]"
                            ),
                            "event_perk": content.Schema(
                                type=content.Type.ARRAY,
                                items=content.Schema(
                                    type=content.Type.STRING,
                                ),
                                description="Event Perk, pick from [Credit, Free Food, Free Stuff]"
                            ),
                            "shortcut_date": content.Schema(
                                type=content.Type.STRING,
                                description="Event Shortcut date, pick from [tomorrow, this_weekend]"
                            ),
                        },
                        required=["search_bar_query", "event_category"]
                    ),
                ),
                genai.protos.FunctionDeclaration(
                    name="get_latest_news_updates",
                    description="Searches for news information with Sun Devil Search Engine",
                    parameters=content.Schema(
                        type=content.Type.OBJECT,
                        properties={
                            "search_bar_query": content.Schema(
                                type=content.Type.STRING,
                                description="Search query"
                            ),
                            "news_campus": content.Schema(
                                type=content.Type.STRING,
                                description="News Campus"
                            ),
                        },
                        required=["search_bar_query"]
                    ),
                ),
                genai.protos.FunctionDeclaration(
                    name="get_latest_sport_updates",
                    description="Fetches comprehensive sports information for Arizona State University across various sports and leagues.",
                    parameters=content.Schema(
                        type=content.Type.OBJECT,
                        properties={
                            "search_bar_query": content.Schema(
                                type=content.Type.STRING,
                                description="search query to filter sports information"
                            ),
                            "sport": content.Schema(
                                type=content.Type.STRING,
                                description="Specific sport to search (e.g., 'football', 'basketball', 'baseball', 'soccer')",
                                enum=[
                                    "football", "basketball", "baseball", 
                                    "soccer", "volleyball", "softball", 
                                    "hockey", "tennis", "track and field"
                                ]
                            ),
                            "league": content.Schema(
                                type=content.Type.STRING,
                                description="League for the sport (NCAA, Pac-12, etc.)",
                                enum=["NCAA", "Pac-12", "Big 12", "Mountain West"]
                            ),
                            "match_date": content.Schema(
                                type=content.Type.STRING,
                                description="Specific match date in YYYY-MM-DD format"
                            ),
                        },
                        required=["search_bar_query","sport"]
                    )
                ),
                genai.protos.FunctionDeclaration(
                    name="get_latest_social_media_updates",
                    description="Searches for ASU social media posts from specified accounts",
                    parameters=content.Schema(
                        type=content.Type.OBJECT,
                        properties={
                            "search_bar_query": content.Schema(
                                type=content.Type.STRING,
                                description="Optional search query to filter social media posts"
                            ),
                            "account_name": content.Schema(
                                type=content.Type.ARRAY,
                                items=content.Schema(
                                    type=content.Type.STRING,
                                    enum=[
                                        "@ArizonaState", 
                                        "@SunDevilAthletics", 
                                        "@SparkySunDevil", 
                                        "@SunDevilFootball", 
                                        "@ASUFootball", 
                                        "@SunDevilFB"
                                    ]
                                ),
                                description="Pick from the List of ASU social media account names to search"
                            )
                        },
                        required=["account_name"]
                    )
                ),
                
                genai.protos.FunctionDeclaration(
                    name="get_library_resources",
                    description="Searches for Books, Articles, Journals, Etc Within ASU Library",
                    parameters=content.Schema(
                        type=content.Type.OBJECT,
                        properties={
                            "search_bar_query": content.Schema(
                                type=content.Type.STRING,
                                description="search query to filter resources"
                            ),
                           
                            "resource_type": content.Schema(
                                type=content.Type.STRING,
                                description="Pick Resource Type from the List",
                                enum=[
                                    "All Items",
                                    "Books",
                                    "Articles",
                                    "Journals",
                                    "Images",
                                    "Scores",
                                    "Maps",
                                    "Sound recordings",
                                    "Video/Film",
                                ]
                            ),
                         
                        },
                        required = ["search_bar_query", "resource_type"],
                    )
                ),
                genai.protos.FunctionDeclaration(
                    name="get_latest_scholarships",
                    description="Fetches comprehensive scholarship information for Arizona State University across various programs.",
                    parameters=content.Schema(
                        type=content.Type.OBJECT,
                        properties={
                            "search_bar_query": content.Schema(
                                type=content.Type.STRING,
                                description="General search terms for scholarships"
                            ),
                            "academic_level": content.Schema(
                                type=content.Type.STRING,
                                description="Academic level of the student",
                                enum=[
                                    "Graduate",
                                    "Undergraduate"
                                ]
                            ),
                            "citizenship_status": content.Schema(
                                type=content.Type.STRING,
                                description="Citizenship status of the applicant",
                                enum=[
                                    "US Citizen",
                                    "US Permanent Resident", 
                                    "DACA/Dreamer",
                                    "International Student (non-US citizen)"
                                ]
                            ),
                            "gpa": content.Schema(
                                type=content.Type.STRING,
                                description="Student's GPA range",
                                enum=[
                                    "2.0 – 2.24",
                                    "2.25 – 2.49",
                                    "2.50 – 2.74", 
                                    "2.75 - 2.99",
                                    "3.00 - 3.24",
                                    "3.25 - 3.49", 
                                    "3.50 - 3.74",
                                    "3.75 - 3.99",
                                    "4.00"
                                ]
                            ),
                           
                            "eligible_applicants": content.Schema(
                                type=content.Type.STRING,
                                description="Student academic standing",
                                enum=[
                                    "First-year Undergrads",
                                    "Second-year Undergrads", 
                                    "Third-year Undergrads",
                                    "Fourth-year+ Undergrads",
                                    "Graduate Students",
                                    "Undergraduate Alumni",
                                    "Graduate Alumni"
                                ]
                            ),
                            "focus": content.Schema(
                                type=content.Type.STRING,
                                description="Scholarship focus area",
                                enum=[
                                    "STEM",
                                    "Business and Entrepreneurship",
                                    "Creative and Performing Arts",
                                    "Environment and Sustainability",
                                    "Health and Medicine",
                                    "Social Science",
                                    "International Affairs",
                                    "Public Policy",
                                    "Social Justice",
                                    "Journalism and Media",
                                    "Humanities"
                                ]
                            ),
                              # "college": content.Schema(
                        #         type=content.Type.STRING,
                        #         description="ASU College or School",
                        #         enum=[
                        #             "Applied Arts and Sciences, School of",
                        #             "Business, W. P. Carey School of",
                        #             "Design & the Arts, Herberger Institute for",
                        #             "Education, Mary Lou Fulton Institute and Graduate School of",
                        #             "Engineering, Ira A. Fulton Schools of",
                        #             "Future of Innovation in Society, School for the",
                        #             "Global Management, Thunderbird School of",
                        #             "Graduate College",
                        #             "Health Solutions, College of",
                        #             "Human Services, College of",
                        #             "Integrative Sciences and Arts, College of",
                        #             "Interdisciplinary Arts & Sciences, New College of",
                        #             "Journalism & Mass Communication, Walter Cronkite School of",
                        #             "Law, Sandra Day O'Connor College of",
                        #             "Liberal Arts and Sciences, The College of",
                        #             "Nursing and Health Innovation, Edson College of",
                        #             "Public Service and Community Solutions, Watts College of",
                        #             "Sustainability, School of",
                        #             "Teachers College, Mary Lou Fulton",
                        #             "University College"
                        #         ]
                        #     ),
                        }, 
                    )
                ),
                genai.protos.FunctionDeclaration(
                    name="get_latest_job_updates",
                    description="Searches for jobs from ASU Handshake",
                    parameters=content.Schema(
                        type=content.Type.OBJECT,
                        properties={
                            "search_bar_query": content.Schema(
                                type=content.Type.STRING,
                                description="Optional search query to filter jobs"
                            ),
                            "job_type": content.Schema(
                                type=content.Type.ARRAY,
                                items=content.Schema(
                                    type=content.Type.STRING,
                                    enum=[
                                        "Full-Time", 
                                        "Part-Time", 
                                        "Internship", 
                                        "On-Campus"
                                    ]
                                ),
                                description="Pick from the List of Job Types to search"
                            ),
                            "job_location": content.Schema(
                                type=content.Type.ARRAY,
                                items=content.Schema(
                                    type=content.Type.STRING,
                                    enum=[
                                        "Tempe, Arizona, United States",

                                        "Mesa, Arizona, United States",

                                        "Phoenix, Arizona, United States",
                                    ]
                                ),
                                description="Pick from the List of ASU Locations to search"
                            ),
                        },
                        
                    )
                ),
                genai.protos.FunctionDeclaration(
                    name="get_latest_class_information",
                    description="Searches for ASU Classes information indepth with ASU Catalog Search",
                    parameters=content.Schema(
                        type=content.Type.OBJECT,
                        properties={
                            "search_bar_query": content.Schema(
                                type=content.Type.STRING,
                                description=" search query to filter classes"
                            ),
                           
                            "class_seat_availability": content.Schema(
                                type=content.Type.STRING,
                                description="Class Availability : Open | All",
                                enum=[
                                    "Open",
                                    "All"
                                ]
                            ),
                            "class_term": content.Schema(
                                type=content.Type.STRING,
                                description="Pick from this list for Class Term",
                                enum=[
                                "Fall 2026",
                                "Summer 2026",
                                "Spring 2026",
                                "Fall 2025",
                                "Summer 2025",
                                "Spring 2025",
                                ]

                            ),
                            "subject_name": content.Schema(
                                type=content.Type.STRING,
                                description="""Class/Course Name """,
                                
                            ),
                            
                            "num_of_credit_units": content.Schema(
                                type=content.Type.ARRAY,
                                items=content.Schema(
                                    type=content.Type.STRING,
                                    enum=[
                                        "1", 
                                        "2", 
                                        "3", 
                                        "4", 
                                        "5", 
                                        "6",
                                        "7",
                                    ]
                                ),
                                description="Pick from the List of Class Credits"
                            ),
                            "class_session": content.Schema(
                                type=content.Type.ARRAY,
                                items=content.Schema(
                                    type=content.Type.STRING,
                                    enum=[
                                        "A", 
                                        "B", 
                                        "C", 
                                        "Other", 
                                    ]
                                ),
                                description="Pick from the List of Class Sessions"
                            ),
                            "class_days": content.Schema(
                                type=content.Type.ARRAY,
                                items=content.Schema(
                                    type=content.Type.STRING,
                                    enum=[
                                        "Monday", 
                                        "Tuesday", 
                                        "Wednesday", 
                                        "Thursday", 
                                        "Friday", 
                                        "Saturday", 
                                        "Sunday", 
                                    ]
                                ),
                                description="Pick from the List of Class Sessions"
                            ),
                            "class_location": content.Schema(
                                type=content.Type.ARRAY,
                                items=content.Schema(
                                    type=content.Type.STRING,
                                    enum=[
                                        "TEMPE",
                                        "WEST",
                                        "POLY",
                                        "OFFCAMP",
                                        "PHOENIX",
                                        "LOSANGELES",
                                        "CALHC",
                                        "ASUSYNC",
                                        "ASUONLINE",
                                        "ICOURSE"
                                        ]
                                ),
                                description="Pick from the List of Class Locations"
                            ),
                          
                        },
                            
                    )
                ),
            ],
        ),
    ],
    tool_config={'function_calling_config': 'AUTO'},
)



# %% [markdown]
# #### Global Instance 

# %%
asu_search_agent = SearchModel()
logger.info("\nInitialized SearchModel Instance")

# %% [markdown]
# ### Discord Model
# 
# The `DiscordModel` class is designed to handle Discord-specific interactions and commands using Google's Generative AI model. 

# %% [markdown]
# 
# 
# #### Initialization
# - Initializes with a `discordmodel`, functions (`DiscordModelFunctions`), and conversation tracking.
# - Implements rate limiting and request counting.
# 
# #### Model Configuration
# - Uses the "gemini-1.5-flash" model with specific generation config settings.
# - Implements safety settings to block low and above levels of hate speech and harassment.
# 
# #### Function Declarations
# - Defines various function declarations for Discord-specific actions:
#   - Notifying moderators and helpers
#   - Creating forum posts and events
#   - Sending bot feedback
#   - Accessing server information
# 
# #### Discord Interactions
# - `handle_discord_server_info`: Provides information about the Sparky Discord Server and Bot.
# - `access_live_status_agent`: Interfaces with a live status agent for real-time information.
# - `send_bot_feedback`: Handles user feedback about the bot.
# 
# #### Conversation Management
# - Implements methods to save and retrieve chat history for users.
# - Manages conversations in a dictionary format.
# 
# #### Action Determination
# - `determine_action` method processes user queries and special instructions.
# - Constructs prompts with current context, user query, and agent instructions.
# - Handles both text responses and function calls from the model.
# 
# #### Rate Limiting
# - Implements a sophisticated rate limiting system to prevent excessive requests.
# 
# #### Error Handling
# - Comprehensive error logging throughout the class.
# - Fallback responses for various error scenarios.
# 
# 

# %% [markdown]
# #### Setting up Model functions

# %%
class Discord_Model_Functions:
    def __init__(self):
        self.discord_client = discord_state.get('discord_client')
        logger.info(f"Initialized Discord Client : {self.discord_client}")
        self.guild = discord_state.get('target_guild')
        self.user_id=discord_state.get('user_id')
        self.user=discord_state.get('user')
        logger.info(f"Initialized Discord Guild : {self.guild}")
     
    async def notify_discord_helpers(self, short_message_to_helper: str) -> str:
        self.guild = discord_state.get('target_guild')
        self.user_id=discord_state.get('user_id')
        self.user=discord_state.get('user')
        logger.info(f"Initialized Discord Guild : {self.guild}")

        if not request_in_dm:
            return "User can only access this command in private messages. It seems like the user is trying to access this command in a discord server. Exiting command."

        await utils.update_text("Checking available discord helpers...")

        logger.info("Contact Model: Handling contact request for helper notification")

        try:

            if not self.guild:
                return "Unable to find the server. Please try again later."

            # Check if user is already connected to a helper
            existing_channel = discord.utils.get(self.guild.channels, name=f"help-{self.user_id}")
            if existing_channel:
                utils.update_ground_sources([existing_channel.jump_url])
                return f"User already has an open help channel."

            # Find helpers
            helper_role = discord.utils.get(self.guild.roles, name="Helper")
            if not helper_role:
                return "Unable to find helpers. Please contact an administrator."

            helpers = [member for member in self.guild.members if helper_role in member.roles and member.status != discord.Status.offline]
            if not helpers:
                return "No helpers are currently available. Please try again later."

            # Randomly select a helper
            selected_helper = random.choice(helpers)

            # Create a private channel
            overwrites = {
                self.guild.default_role: discord.PermissionOverwrite(read_messages=False),
                user: discord.PermissionOverwrite(read_messages=True, send_messages=True),
                selected_helper: discord.PermissionOverwrite(read_messages=True, send_messages=True)
            }
            
            category = discord.utils.get(self.guild.categories, name="Customer Service")
            if not category:
                return "Unable to find the Customer Service category. Please contact an administrator."

            channel = await self.guild.create_text_channel(f"help-{self.user_id}", category=category, overwrites=overwrites)

            # Send messages
            await channel.send(f"{user.mention} and {selected_helper.mention}, this is your help channel.")
            await channel.send(f"User's message: {short_message_to_helper}")

            # Notify the helper via DM
            await selected_helper.send(f"You've been assigned to a new help request. Please check {channel.mention}")
            utils.update_ground_sources([channel.jump_url])
            return f"Server Helper Assigned: {selected_helper.name}\n"

        except Exception as e:
            logger.error(f"Error notifying helpers: {str(e)}")
            return f"An error occurred while notifying helpers: {str(e)}"

    async def notify_moderators(self, short_message_to_moderator: str) -> str:
        self.guild = discord_state.get('target_guild')
        self.user_id=discord_state.get('user_id')
        self.user=discord_state.get('user')
        
        logger.info(f"Initialized Discord Guild : {self.guild}")


        if not discord_state.get('request_in_dm'):
            return "User can only access this command in private messages. It seems like the user is trying to access this command in a discord server. Exiting command."

        await utils.update_text("Checking available discord moderators...")

        logger.info("Contact Model: Handling contact request for moderator notification")

        try:
            if not self.guild:
                return "Unable to find the server. Please try again later."

            # Check if user is already connected to a helper
            existing_channel = discord.utils.get(self.guild.channels, name=f"support-{self.user_id}")
            if existing_channel:
                utils.update_ground_sources([existing_channel.jump_url])
                return f"User already has an open support channel."
            # Find helpers/moderators
            helper_role = discord.utils.get(self.guild.roles, name="mod")
            if not helper_role:
                return "Unable to find helpers. Please contact an administrator."

            helpers = [member for member in self.guild.members if helper_role in member.roles]
            if not helpers:
                return "No helpers are currently available. Please try again later."

            # Randomly select a helper
            selected_helper = random.choice(helpers)

            # Create a private channel
            overwrites = {
                self.guild.default_role: discord.PermissionOverwrite(read_messages=False),
                self.user: discord.PermissionOverwrite(read_messages=True, send_messages=True),
                selected_helper: discord.PermissionOverwrite(read_messages=True, send_messages=True)
            }
            
            category = discord.utils.get(self.guild.categories, name="Customer Service")
            if not category:
                return "Unable to find the Customer Service category. Please contact an administrator."

            channel = await self.guild.create_text_channel(f"support-{self.user_id}", category=category, overwrites=overwrites)

            # Send messages
            await channel.send(f"{self.user.mention} and {selected_helper.mention}, this is your support channel.")
            await channel.send(f"User's message: {short_message_to_moderator}")

            # Notify the helper via DM
            await selected_helper.send(f"You've been assigned to a new support request. Please check {channel.mention}")
            utils.update_ground_sources([channel.jump_url])
            return f"Moderator Assigned: {selected_helper.name}"

        except Exception as e:
            logger.error(f"Error notifying moderators: {str(e)}")
            return f"An error occurred while notifying moderators: {str(e)}"

    # async def start_recording_discord_call(self,channel_id:Any) -> str: 

        
    #     logger.info(f"Initialized Discord Guild : {self.guild}")
    #     await utils.update_text("Checking user permissions...")
       
    #     if not discord_state.get('user_has_mod_role'):
    #         return "User does not have enough permissions to start recording a call. This command is only accessible by moderators. Exiting command..."

    #     if not discord_state.get('request_in_dm'):
    #         return "User can only access this command in private messages. It seems like the user is trying to access this command in a discord server. Exiting command."

    #     if not discord_state.get('user_voice_channel_id'):
    #         return "User is not in a voice channel. User needs to be in a voice channel to start recording. Exiting command..."

    #     logger.info("Discord Model: Handling recording request")

    #     return f"Recording started!"

    async def create_discord_forum_post(self, title: str, category: str, body_content_1: str, body_content_2: str, body_content_3: str, link:str=None) -> str:
        self.guild = discord_state.get('target_guild')
        
        logger.info(f"Initialized Discord Guild : {self.guild}")
        await utils.update_text("Checking user permissions...")


        if not discord_state.get('request_in_dm'):
            return "User can only access this command in private messages. It seems like the user is trying to access this command in a discord server. Exiting command."

        logger.info("Discord Model: Handling discord forum request with context")

        try:
            if not self.guild:
                return "Unable to find the server. Please try again later."
            try:
                
                # Find the forum channel 
                forum_channel = discord.utils.get(self.guild.forums, name='qna')  # Replace with your forum channel name
            except Exception as e:
                logger.error(f"Error finding forum channel: {str(e)}")
                return f"An error occurred while finding the forum channel: {str(e)}"
            if not forum_channel:
                return "Forum channel not found. Please ensure the forum exists."

            # Create the forum post
            content = f"{body_content_1}\n\n{body_content_2}\n\n{body_content_3}".strip()
            if link:
                content+=f"\n[Link]({link})"
            try:
                logger.info(f"Forum channel ID: {forum_channel.id if forum_channel else 'None'}")
                
                thread = await forum_channel.create_thread(
                    name=title,
                    content=content,
                )

            except Exception as e:
                
                logger.error(f"Error creating forum thread: {str(e)}")
                return f"An error occurred while creating the forum thread: {str(e)}"
            logger.info(f"Created forum thread {thread.message.id} {type(thread)}")
            
            utils.update_ground_sources([f"https://discord.com/channels/1256076931166769152/{thread.id}"])
            return f"Forum post created successfully.\nTitle: {title}\nDescription: {content[:100]}...\n"
        

        except discord.errors.Forbidden:
            return "The bot doesn't have permission to create forum posts. Please contact an administrator."
        except discord.errors.HTTPException as e:
            logger.error(f"HTTP error creating forum post: {str(e)}")
            return f"An error occurred while creating the forum post: {str(e)}"
        except Exception as e:
            logger.error(f"Error creating forum post: {str(e)}")
            return f"An unexpected error occurred while creating the forum post: {str(e)}"
    
    async def create_discord_announcement(self, ping: str, title: str, category: str, body_content_1: str, body_content_2: str, body_content_3: str, link:str = None) -> str:
        self.discord_client = discord_state.get('discord_client')
        self.guild = discord_state.get('target_guild')
        
        await utils.update_text("Checking user permissions...")


        logger.info(f"Discord Model: Handling discord announcement request with context")

        if not discord_state.get('request_in_dm'):
            return "User can only access this command in private messages. It seems like the user is trying to access this command in a discord server. Exiting command."

        if not discord_state.get('user_has_mod_role'):
            return "User does not have enough permissions to create an announcement. This command is only accessible by moderators. Exiting command..."

        try:
            # Find the announcements channel
            announcements_channel = discord.utils.get(self.discord_client.get_all_channels(), name='announcements')
            if not announcements_channel:
                return "Announcements channel not found. Please ensure the channel exists."

            # Create the embed
            embed = discord.Embed(title=title, color=discord.Color.blue())
            embed.add_field(name="Category", value=category, inline=False)
            embed.add_field(name="Details", value=body_content_1, inline=False)
            if body_content_2:
                embed.add_field(name="Additional Information", value=body_content_2, inline=False)
            if body_content_3:
                embed.add_field(name="More Details", value=body_content_3, inline=False)
            if link:
                embed.add_field(name="Links", value=link, inline=False)

            # Send the announcement
            message = await announcements_channel.send(content="@som", embed=embed)
            utils.update_ground_sources([message.jump_url])
            return f"Announcement created successfully."

        except Exception as e:
            logger.error(f"Error creating announcement: {str(e)}")
            return f"An error occurred while creating the announcement: {str(e)}"
  
    async def create_discord_event(self, title: str, time_start: str, time_end: str, description: str, img_provided: Any = None) -> str:
        self.guild = discord_state.get('target_guild')
        
        logger.info(f"Initialized Discord Guild : {self.guild}")
        await utils.update_text("Checking user permissions...")


        if not discord_state.get('request_in_dm'):
            return "User can only access this command in private messages. It seems like the user is trying to access this command in a discord server. Exiting command."

        if not discord_state.get('user_has_mod_role'):
            return "User does not have enough permissions to create an event. This command is only accessible by moderators. Exiting command..."

        logger.info("Discord Model: Handling discord event creation request")

        try:
            if self.guild:
                return "Unable to find the server. Please try again later."

            # Parse start and end times
            start_time = datetime.fromisoformat(time_start)
            end_time = datetime.fromisoformat(time_end)

            # Create the event
            event = await self.guild.create_scheduled_event(
                name=title,
                description=description,
                start_time=start_time,
                end_time=end_time,
                location="Discord",  # or specify a different location if needed
                privacy_level=discord.PrivacyLevel.guild_only
            )

            # If an image was provided, set it as the event cover
            if img_provided:
                await event.edit(image=img_provided)

            # Create an embed for the event announcement
            embed = discord.Embed(title=title, description=description, color=discord.Color.blue())
            embed.add_field(name="Start Time", value=start_time.strftime("%Y-%m-%d %H:%M:%S"), inline=True)
            embed.add_field(name="End Time", value=end_time.strftime("%Y-%m-%d %H:%M:%S"), inline=True)
            embed.add_field(name="Location", value="Discord", inline=False)
            embed.set_footer(text=f"Event ID: {event.id}")

            # Send the announcement to the announcements channel
            announcements_channel = discord.utils.get(self.guild.text_channels, name="announcements")
            if announcements_channel:
                await announcements_channel.send(embed=embed)
            
            utils.update_ground_sources([event.url])

            return f"Event created successfully.\nTitle: {title}\nDescription: {description[:100]}...\nStart Time: {start_time}\nEnd Time: {end_time}\n"

        except discord.errors.Forbidden:
            return "The bot doesn't have permission to create events. Please contact an administrator."
        except ValueError as e:
            return f"Invalid date format: {str(e)}"
        except Exception as e:
            logger.error(f"Error creating event: {str(e)}")
            return f"An unexpected error occurred while creating the event: {str(e)}"
    
    async def search_discord(self,query:str):
        results = await utils.perform_web_search(optional_query=query,doc_title =query)
        return results
    
    async def create_discord_poll(self, question: str, options: List[str], channel_name: str) -> str:
        self.guild = discord_state.get('target_guild')
        

        await utils.update_text("Checking user permissions...")

        if not discord_state.get('request_in_dm'):
            return "User can only access this command in private messages. Exiting command."

        if not discord_state.get('user_has_mod_role'):
            return "User does not have enough permissions to create a poll. This command is only accessible by moderators. Exiting command..."

        logger.info("Discord Model: Handling discord poll creation request")

        try:
            if not self.guild:
                return "Unable to find the server. Please try again later."

            # Find the specified channel
            channel = discord.utils.get(self.guild.text_channels, name=channel_name)
            if not channel:
                return f"Channel '{channel_name}' not found. Please check the channel name and try again."

            # Create the poll message
            poll_message = f"📊 **{question}**\n\n"
            emoji_options = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟"]
            try:
                for i, option in enumerate(options):  # Limit to 10 options
                    poll_message += f"{emoji_options[i]} {option}\n"
                    
            except Exception as e:
                logger.error(f"Error creating poll options: {str(e)}")
                return f"An unexpected error occurred while creating poll options: {str(e)}"
            
            # Send the poll message
            try:
                poll = await channel.send(poll_message)
            except Exception as e:
                logger.error(f"Error sending poll message: {str(e)}")
                return  f"An unexpected error occurred while sending poll: {str(e)}"
            
            utils.update_ground_sources([poll.jump_url])  

            # Add reactions
            try:
                
                for i in range(len(options)):
                    await poll.add_reaction(emoji_options[i])
            except Exception as e:
                logger.error(f"Error adding reactions to poll: {str(e)}")
                return f"An unexpected error occurred while adding reactions to poll: {str(e)}"
            
            return f"Poll created successfully in channel '{channel_name}'.\nQuestion: {question}\nOptions: {', '.join(options)}"

        except discord.errors.Forbidden:
            return "The bot doesn't have permission to create polls or send messages in the specified channel. Please contact an administrator."
        except Exception as e:
            logger.error(f"Error creating poll: {str(e)}")
            return f"An unexpected error occurred while creating the poll: {str(e)}"

# %% [markdown]
# #### Initializing functions

# %%
class DiscordModel:
    
    def __init__(self):
        self.model = discord_model
        self.chat = None
        self.functions = Discord_Model_Functions()
        self.last_request_time = time.time()
        self.request_counter = 0
        self.conversations: Dict[str, List[Dict[str, str]]] = {}
        
    def _initialize_model(self):
        if not self.model:
            return logger.error("Model not initialized at ActionFunction")
            
        # Rate limiting check
        current_time = time.time()
        if current_time - self.last_request_time < 1.0:  # 1 second cooldown
            raise Exception("Rate limit exceeded")
            
        self.last_request_time = current_time
        self.request_counter += 1
        user_id = discord_state.get("user_id")
        self.chat = self.model.start_chat(history=self._get_chat_history(user_id),enable_automatic_function_calling=True)
        
    def _get_chat_history(self, user_id: str) -> List[Dict[str, str]]:
        return self.conversations.get(user_id, [])

    def _save_message(self, user_id: str, role: str, content: str):
        if user_id not in self.conversations:
            self.conversations[user_id] = []
        
        self.conversations[user_id].append({
            "role": role,
            "parts": [{"text": content}]
        })
        
        # Limit the conversation length to 3 messages per user
        if len(self.conversations[user_id]) > 3:
            self.conversations[user_id].pop(0)
        
    async def execute_function(self, function_call):
        """Execute the called function and return its result"""
        function_name = function_call.name
        function_args = function_call.args
        
        function_mapping = {
            'notify_moderators': self.functions.notify_moderators,
            'notify_discord_helpers': self.functions.notify_discord_helpers,
            'create_discord_forum_post': self.functions.create_discord_forum_post,
            'create_discord_announcement': self.functions.create_discord_announcement,
            'create_discord_poll': self.functions.create_discord_poll,
            'search_discord': self.functions.search_discord,
        }
        
        if function_name in function_mapping:
            function_to_call = function_mapping[function_name]
            func_response = await function_to_call(**function_args)
            # response = await self.chat.send_message_async(f"{function_name} response : {func_response}")
            
            if func_response:
                # self._save_message(user_id, "model", f"""(Only Visible to You) System Tools - Discord Agent Response: {func_response}""")
                return func_response
            else:
                logger.error(f"Error extracting text from response: {e}")
                return "Error processing response"
                
                
        else:
            raise ValueError(f"Unknown function: {function_name}")
    
    async def determine_action(self, query: str,special_instructions:str) -> str:
        """Determines and executes the appropriate action based on the user query"""
        try:
            self._initialize_model()
            user_id = discord_state.get("user_id")
            self._save_message(user_id, "user", query)
            final_response = ""
            # Simplified prompt that doesn't encourage analysis verbosity
            prompt = f"""
            ### Context:
            - Current Date and Time: {datetime.now().strftime('%H:%M %d') + ('th' if 11<=int(datetime.now().strftime('%d'))<=13 else {1:'st',2:'nd',3:'rd'}.get(int(datetime.now().strftime('%d'))%10,'th')) + datetime.now().strftime(' %B, %Y') }
            - Superior Agent Instruction: {query}
            - Superior Agent Remarks (if any): {special_instructions}
            {app_config.get_discord_agent_prompt()}
            """
            
            response = await self.chat.send_message_async(prompt)
            logger.info(self._get_chat_history)
            self._save_message(user_id, "model", f"{response.parts}" )
            for part in response.parts:
                if hasattr(part, 'function_call') and part.function_call: 
                    # Execute function and store only its result
                    final_response = await self.execute_function(part.function_call)
                    firestore.update_message("discord_agent_message", f"Function called {part.function_call}\n Function Response {final_response} ")
                elif hasattr(part, 'text') and part.text.strip():
                    # Only store actual response content, skip analysis messages
                    text = part.text.strip()
                    firestore.update_message("discord_agent_message", f"Text Response {text} ")
                    if not text.startswith("This query") and not "can be answered directly" in text:
                        final_response = text.strip()
            
        
        # Return only the final message
            return final_response if final_response else "Hi! How can I help you with ASU or the Discord server today?"
            
        except Exception as e:
            logger.error(f"Discord Model : Error in determine_action: {str(e)}")
            return "I apologize, but I couldn't generate a response at this time. Please try again."
        

# %% [markdown]
# #### Setting up the model 

# %%


discord_model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config={
        "temperature": 0.0, 
        "top_p": 0.1,
        "top_k": 40,
        "max_output_tokens": 2500,
        "response_mime_type": "text/plain",
    },safety_settings={
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    },

system_instruction = f""" {app_config.get_discord_agent_instruction()}
""",
    tools=[
        genai.protos.Tool(
            function_declarations=[
                      
                genai.protos.FunctionDeclaration(
                    name="notify_moderators",
                    description="Contacts Discord moderators (Allowed only in Private Channels)",
                    parameters=content.Schema(
                        type=content.Type.OBJECT,
                        properties={
                            "short_message_to_moderator": content.Schema(
                                type=content.Type.STRING,
                                description="Message for moderators "
                            ),
                        },
                    ),
                ),
                genai.protos.FunctionDeclaration(
                    name="notify_discord_helpers",
                    description="Contacts Discord helpers (Allowed only in Private Channels)",
                    parameters=content.Schema(
                        type=content.Type.OBJECT,
                        properties={
                            "short_message_to_helpers": content.Schema(
                                type=content.Type.STRING,
                                description="Message for helpers "
                            ),
                        },
                    ),
                ),
                genai.protos.FunctionDeclaration(
                    name="search_discord",
                    description="Search for messages on discord server",
                    parameters=content.Schema(
                        type=content.Type.OBJECT,
                        properties={
                            "query": content.Schema(
                                type=content.Type.STRING,
                                description="Keywords to search"
                            ),
                        },
                    ),
                ),
                
             
                # genai.protos.FunctionDeclaration(
                #     name="start_recording_discord_call",
                #     description="Starts recording a voice call (Allowed to special roles only)",
                #     parameters=content.Schema(
                #         type=content.Type.OBJECT,
                #         properties={
                #             "channel_id": content.Schema(
                #                 type=content.Type.STRING,
                #                 description="Voice channel ID to record"
                #             ),
                #         },
                          
                #     ),
                # ),
                genai.protos.FunctionDeclaration(
                    name="create_discord_poll",
                    description="Creates a poll in a specified Discord channel (Allowed only in Private Channels)",
                    parameters=content.Schema(
                        type=content.Type.OBJECT,
                        properties={
                            "question": content.Schema(
                                type=content.Type.STRING,
                                description="The main question for the poll"
                            ),
                            "options": content.Schema(
                                type=content.Type.ARRAY,
                                items=content.Schema(type=content.Type.STRING),
                                description="List of options for the poll (maximum 10)"
                            ),
                            "channel_name": content.Schema(
                                type=content.Type.STRING,
                                description="The name of the channel where the poll should be posted"
                            )
                        },
                        required=["question", "options", "channel_name"]
                    ),
                ),


                genai.protos.FunctionDeclaration(
                    name="get_user_profile_details",
                    description="Retrieves user profile information",
                    parameters=content.Schema(
                        type=content.Type.OBJECT,
                        properties={
                            "context": content.Schema(
                                type=content.Type.STRING,
                                description="User context"
                            ),
                        },
                    ),
                ),
                genai.protos.FunctionDeclaration(
                    name="create_discord_announcement",
                    description="Creates a server announcement (Allowed to special roles only in Private Channels)",
                    parameters=content.Schema(
                        type=content.Type.OBJECT,
                        properties={
                            "ping": content.Schema(
                                type=content.Type.STRING,
                                description="The role or user to ping with the announcement (e.g., @everyone, @role, or user ID)"
                            ),
                            "title": content.Schema(
                                type=content.Type.STRING,
                                description="The title of the announcement"
                            ),
                            "category": content.Schema(
                                type=content.Type.STRING,
                                description="The category of the announcement"
                            ),
                            "body_content_1": content.Schema(
                                type=content.Type.STRING,
                                description="The main content of the announcement"
                            ),
                            "body_content_2": content.Schema(
                                type=content.Type.STRING,
                                description="Additional content for the announcement (optional)"
                            ),
                            "body_content_3": content.Schema(
                                type=content.Type.STRING,
                                description="More details for the announcement (optional)"
                            ),
                            "link": content.Schema(
                                type=content.Type.STRING,
                                description="Links"
                            ),
                            
                        },
                        required=["title", "category", "body_content_1","body_content_2","body_content_3"]
                    ),
                ),

                genai.protos.FunctionDeclaration(
                    name="create_discord_forum_post",
                    description="Creates a new forum post (Allowed only in Private Channels)",
                    parameters=content.Schema(
                        type=content.Type.OBJECT,
                        properties={
                            "title": content.Schema(
                                type=content.Type.STRING,
                                description="The title of the forum post"
                            ),
                            "category": content.Schema(
                                type=content.Type.STRING,
                                description="The category tag for the forum post"
                            ),
                            "body_content_1": content.Schema(
                                type=content.Type.STRING,
                                description="The main content of the forum post"
                            ),
                            "body_content_2": content.Schema(
                                type=content.Type.STRING,
                                description="Additional content for the forum post (optional)"
                            ),
                            "body_content_3": content.Schema(
                                type=content.Type.STRING,
                                description="More details for the forum post (optional)"
                            ),
                            "link": content.Schema(
                                type=content.Type.STRING,
                                description="Links"
                            ),
                        },
                        required=["title", "category", "body_content_1","body_content_2","body_content_3"]
                    ),
                ),
                genai.protos.FunctionDeclaration(
                    name="create_discord_event",
                    description="Creates a new Discord event (Allowed only in Private Channels for users with required permissions)",
                    parameters=content.Schema(
                        type=content.Type.OBJECT,
                        properties={
                            "title": content.Schema(
                                type=content.Type.STRING,
                                description="The title of the Discord event"
                            ),
                            "time_start": content.Schema(
                                type=content.Type.STRING,
                                description="The start time of the event in ISO format (e.g., '2023-12-31T23:59:59')"
                            ),
                            "time_end": content.Schema(
                                type=content.Type.STRING,
                                description="The end time of the event in ISO format (e.g., '2024-01-01T01:00:00')"
                            ),
                            "description": content.Schema(
                                type=content.Type.STRING,
                                description="The description of the event"
                            ),
                            "img_provided": content.Schema(
                                type=content.Type.STRING,
                                description="URL or file path of an image to be used as the event cover (optional)"
                            ),
                        },
                        required=["title", "time_start", "time_end", "description"]
                    ),
                ),
                
               
            ],
        ),
    ],
    tool_config={'function_calling_config': 'ANY'},
)


# %% [markdown]
# #### global Instance

# %%
asu_discord_agent = DiscordModel()
logger.info("\nInitailized DIscord Model Instance")

# %% [markdown]
# ### Action Model
# 
# This `ActionModel` serves as a central component for determining and executing appropriate actions based on user queries, integrating various functions and agents to provide comprehensive responses in the ASU Discord Research Assistant Bot.
# 

# %% [markdown]
# #### Initialization
# - Initializes with a model (`action_model`), functions (`ActionModelFunctions`), and conversation tracking.
# - Implements rate limiting and request counting.
# 
# #### Function Execution
# - `execute_function` method dynamically calls the appropriate function based on the function name.
# - Supports various action functions like performing web searches, accessing other agents, and retrieving specific information.
# 
# #### Model Configuration
# - Uses the "gemini-1.5-flash" model with specific generation config settings.
# - Implements safety settings to block low and above levels of hate speech and harassment.
# 
# #### Action Determination
# - `determine_action` method processes user queries and special instructions.
# - Constructs prompts with current context, user query, and agent instructions.
# - Handles both text responses and function calls from the model.
# 
# #### Response Processing
# - Iterates through model response parts, handling both text and function calls.
# - Executes functions when called and formats the final response.
# 
# #### Error Handling
# - Comprehensive error logging throughout the class.
# - Fallback responses for various error scenarios.
# 
# 

# %% [markdown]
# #### Setting up Model Functions

# %%
class Action_Model_Functions:
    
    def __init__(self):
        self.conversations = {}
        self.client = genai2.Client(api_key=app_config.get_api_key())
        self.model_id = "gemini-2.0-flash-exp"
        self.discord_client = discord_state.get('discord_client')
        self.guild = discord_state.get('target_guild')
        self.user_id=discord_state.get('user_id')
        self.user= discord_state.get('user')
        self.google_search_tool = Tool(google_search=GoogleSearch())
    
    def get_final_url(self,url):
        try:
            response = requests.get(url, allow_redirects=True)
            return response.url
        except Exception as e:
            logger.error(e)
            return e  

    async def access_search_agent(self, instruction_to_agent: str, special_instructions: str):
        logger.info(f"Action Model : accessing search agent with instruction {instruction_to_agent} with special instructions {special_instructions}")
        await (google_sheet.increment_function_call(discord_state.get('user_id'), 'H'))
        await (google_sheet.increment_function_call(discord_state.get('user_id'), 'N'))
        try:
            response = await asu_search_agent.determine_action(instruction_to_agent,special_instructions)
            return response
        except Exception as e:
            logger.error(f"Error in access search agent: {str(e)}")
            return f"Search Agent Not Responsive"
         
    async def access_discord_agent(self, instruction_to_agent: str,special_instructions: str):
        logger.info(f"Action Model : accessing discord agent with instruction {instruction_to_agent} with special instructions {special_instructions}")
        await (google_sheet.increment_function_call(discord_state.get('user_id'), 'J'))
        await (google_sheet.increment_function_call(discord_state.get('user_id'), 'N'))
        try:
            response = await asu_discord_agent.determine_action(instruction_to_agent,special_instructions)
            
            return response
        except Exception as e:
            logger.error(f"Error in access discord agent: {str(e)}")
            return f"Discord Agent Not Responsive"
        
    async def get_user_profile_details(self) -> str:
        """Retrieve user profile details from the Discord server"""
        self.guild = discord_state.get('target_guild')
        self.user_id = discord_state.get('user_id')
        logger.info(f"Discord Model: Handling user profile details request for user ID: {user_id}")

        if not request_in_dm:
            return "User can only access this command in private messages. It seems like the user is trying to access this command in a discord server. Exiting command."

        try:
            # If no user_id is provided, use the requester's ID
            if not user_id:
                user_id = self.user_id

            member = await self.guild.fetch_member(user_id)
            if not member:
                return f"Unable to find user with ID {user_id} in the server."

            # Fetch user-specific data (customize based on your server's setup)
            join_date = member.joined_at.strftime("%Y-%m-%d")
            roles = [role.name for role in member.roles if role.name != "@everyone"]
            
            # You might need to implement these functions based on your server's systems
            # activity_points = await self.get_user_activity_points(user_id)
            # leaderboard_position = await self.get_user_leaderboard_position(user_id)
            # - Activity Points: {activity_points}
            # - Leaderboard Position: {leaderboard_position}

            profile_info = f"""
            User Profile for {member.name}#{member.discriminator}:
            - Join Date: {join_date}
            - Roles: {', '.join(roles)}
            - Server Nickname: {member.nick if member.nick else 'None'}
            """

            return profile_info.strip()

        except discord.errors.NotFound:
            return f"User with ID {user_id} not found in the server."
        except Exception as e:
            logger.error(f"Error retrieving user profile: {str(e)}")
            return f"An error occurred while retrieving the user profile: {str(e)}"
    
    async def get_discord_server_info(self) -> str:
             
        self.discord_client = discord_state.get('discord_client')
        logger.info(f"Initialized Discord Client : {self.discord_client}")
        self.guild = discord_state.get("target_guild")
        
        logger.info(f"Initialized Discord Guild : {self.guild}")
        """Create discord forum post callable by model"""

        
        logger.info(f"Discord Model : Handling discord server info request with context")
                
        
        return f"""1.Sparky Discord Server - Sparky Discord Server is a place where ASU Alumni's or current students join to hangout together, have fun and learn things about ASU together and quite frankly!
        2. Sparky Discord Bot -  AI Agent built to help people with their questions regarding ASU related information and sparky's discord server. THis AI Agent can also perform discord actions for users upon request."""
    
    async def access_live_status_agent(self, instruction_to_agent: str, special_instructions: str):
        logger.info(f"Action Model : accessing live status agent with instruction {instruction_to_agent} with special instructions {special_instructions}")
        await (google_sheet.increment_function_call(discord_state.get('user_id'), 'K'))
        await (google_sheet.increment_function_call(discord_state.get('user_id'), 'N'))
        
        try:
            response = await asu_live_status_agent.determine_action(instruction_to_agent,special_instructions)
            return response
        except Exception as e:
            logger.error(f"Error in deep search agent: {str(e)}")
            return "I apologize, but I couldn't retrieve the information at this time."
              
    async def send_bot_feedback(self, feedback: str) -> str:
        self.user = discord_state.get('user') 
        self.discord_client = discord_state.get('discord_client')
        
        await utils.update_text("Opening feedbacks...")
        
        logger.info("Contact Model: Handling contact request for server feedback")

        try:
            # Find the feedbacks channel
            feedbacks_channel = discord.utils.get(self.discord_client.get_all_channels(), name='feedback')
            if not feedbacks_channel:
                return "feedbacks channel not found. Please ensure the channel exists."

            # Create an embed for the feedback
            embed = discord.Embed(title="New Server feedback", color=discord.Color.green())
            embed.add_field(name="feedback", value=feedback, inline=False)
            embed.set_footer(text=f"Suggested by {self.user.name}")

            # Send the feedback to the channel
            message = await feedbacks_channel.send(embed=embed)

            # Add reactions for voting
            await message.add_reaction('👍')
            await message.add_reaction('👎')
            
            utils.update_ground_sources([message.jump_url])
            
            return f"Your feedback has been successfully submitted."

        except Exception as e:
            logger.error(f"Error sending feedback: {str(e)}")
            return f"An error occurred while sending your feedback: {str(e)}"
    
    def _get_chat_history(self, user_id):
        return self.conversations.get(user_id, [])

    def _save_message(self, user_id: str, role: str, content: str):
        if user_id not in self.conversations:
            self.conversations[user_id] = []
        
        self.conversations[user_id].append({
            "role": role,
            "parts": [{"text": content}]
        })
        
        # Limit the conversation length to 3 messages per user
        if len(self.conversations[user_id]) > 3:
            self.conversations[user_id].pop(0)

    async def access_google_agent(self, original_query: str, detailed_query: str, generalized_query: str, relative_query: str, categories: list):
        firestore.update_message("category", categories)
        await (google_sheet.increment_function_call(discord_state.get('user_id'), 'L'))
        await (google_sheet.increment_function_call(discord_state.get('user_id'), 'M'))
        
        user_id = discord_state.get('user_id')
        responses=[]
        logger.info(f"Action Model: accessing Google Search with instruction {original_query}")
        try:
            # Perform database search
            queries = [
                {"search_bar_query": original_query},
                {"search_bar_query": detailed_query},
                {"search_bar_query": generalized_query},
                {"search_bar_query": relative_query}
            ]
            for query in queries:
                response = await utils.perform_database_search(query["search_bar_query"], categories) or []
                responses.append(response)

            responses = [resp for resp in responses if resp]
        except:
            logger.error("No results found in database")
            pass
        # Get chat history
        
        chat_history = self._get_chat_history(user_id)

        # Prepare the prompt
        prompt = f"""
        
        {app_config.get_google_agent_prompt()}
        
        - If applicable, you may use the related database information : {responses}
        
        Chat History:
        {chat_history}

        User's Query: {original_query}

        Deliver a direct, actionable response that precisely matches the query's specificity."""
        
        try:     
            # Generate response
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt,
                config=GenerateContentConfig(
                    tools=[self.google_search_tool],
                    response_modalities=["TEXT"],
                    system_instruction=f"{app_config.get_google_agent_instruction()}",
                    max_output_tokens=600
                )
            )
            
            grounding_sources = [self.get_final_url(chunk.web.uri) for candidate in response.candidates if candidate.grounding_metadata and candidate.grounding_metadata.grounding_chunks for chunk in candidate.grounding_metadata.grounding_chunks if chunk.web]
            
            utils.update_ground_sources(grounding_sources)
            
            response_text = "".join([part.text for part in response.candidates[0].content.parts if part.text])


            # Save the interaction to chat history
            self._save_message(user_id, "user", original_query)
            self._save_message(user_id, "model", response_text)

            logger.info(response_text)

            if not response_text:
                logger.error("No response from Google Search")
                return None
            return response_text
        except Exception as e:
            logger.info(f"Google Search Exception {e}")
            return responses 

# %% [markdown]
# #### Initializing Functions

# %%
class ActionModel:
    
    def __init__(self):
        self.model = action_model
        self.chat = None
        self.functions = Action_Model_Functions()
        self.last_request_time = time.time()
        self.request_counter = 0

    def _initialize_model(self):
        if not self.model:
            raise Exception("Model is not available.")
        current_time = time.time()
        if current_time - self.last_request_time < 1.0:
            raise Exception("Rate limit exceeded. Please try again later.")
        self.last_request_time = current_time
        self.request_counter += 1
        self.chat = self.model.start_chat(enable_automatic_function_calling=True)

    async def execute_function(self, function_call: Any) -> str:
        function_mapping = {
            'access_search_agent': self.functions.access_search_agent,
            'access_google_agent': self.functions.access_google_agent,
            'access_discord_agent': self.functions.access_discord_agent,
            'send_bot_feedback': self.functions.send_bot_feedback,
            'access_live_status_agent': self.functions.access_live_status_agent,
            'get_user_profile_details': self.functions.get_user_profile_details,
            'get_discord_server_info': self.functions.get_discord_server_info,
        }

        function_name = function_call.name
        function_args = function_call.args

        if function_name not in function_mapping:
            raise ValueError(f"Unknown function: {function_name}")
        
        function_to_call = function_mapping[function_name]
        return await function_to_call(**function_args)

    async def process_gemini_response(self, response: Any) -> tuple[str, bool, Any]:
        text_response = ""
        has_function_call = False
        function_call = None
        logger.info(response)

        for part in response.parts:
            if hasattr(part, 'text') and part.text.strip():
                text_response += f"\n{part.text.strip()}"
                firestore.update_message("action_agent_message", f"Text Response : {text_response} ")
            if hasattr(part, 'function_call') and part.function_call:
                has_function_call = True
                function_call = part.function_call
                temp_func =  {
                "function_call": {
                    "name": part.function_call.name,
                    "args": dict(part.function_call.args)
                    }
                }
                firestore.update_message("action_agent_message", json.dumps(temp_func, indent=2))

        return text_response, has_function_call, function_call

    async def determine_action(self, query: str) -> List[str]:
        try:
            final_response=""
            self._initialize_model()
            responses = []
            prompt = f"""
            ### Context:
            - Current Date and Time: {datetime.now().strftime('%H:%M %d') + ('th' if 11<=int(datetime.now().strftime('%d'))<=13 else {1:'st',2:'nd',3:'rd'}.get(int(datetime.now().strftime('%d'))%10,'th')) + datetime.now().strftime(' %B, %Y') }
            - User Query: {query}
            {app_config.get_action_agent_prompt()}
            """
            
            response = await self.chat.send_message_async(prompt)
            logger.info(f"RAW TEST RESPONSE : {response}")
            
            while True:
                text_response, has_function_call, function_call = await self.process_gemini_response(response)
                responses.append(text_response)
                final_response += text_response
                if not has_function_call:
                    break
                function_result = await self.execute_function(function_call)
                firestore.update_message("action_agent_message", f"""(User cannot see this response) System Generated - \n{function_call.name}\nResponse: {function_result}\nAnalyze the response and answer the user's question.""")
                logger.info("\nAction Model @ Function result is: %s", function_result)
                response = await self.chat.send_message_async(f"""(User cannot see this response) System Generated - \n{function_call.name}\nResponse: {function_result}\nAnalyze the response and answer the user's question.""")
                
            final_response = " ".join(response.strip() for response in responses if response.strip())
            
            return final_response.strip()
        
        except Exception as e:
            logger.error(f"Error in determine_action: {e}")
            return ["I'm sorry, I couldn't generate a response. Please try again."]

# %% [markdown]
# #### Setting up the model

# %%
action_model = genai.GenerativeModel(
    
    model_name="gemini-1.5-flash",
    
    generation_config={
        "temperature": 0.0, 
        "top_p": 0.1,
        "top_k": 40,
        "max_output_tokens": 3100,
        "response_mime_type": "text/plain",
    },
    safety_settings={
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    },
    
    system_instruction = f""" {app_config.get_action_agent_instruction()}""",
    
    tools=[
        genai.protos.Tool(
            function_declarations=[
                            
                genai.protos.FunctionDeclaration(
                    name="access_search_agent",
                    description="Has ability to search for ASU-specific Targeted , real-time information extraction related to Jobs, Scholarships, Library Catalog, News, Events, Social Media, Sport Updates, Clubs",
                    parameters=content.Schema(
                        type=content.Type.OBJECT,
                        properties={
                            "instruction_to_agent": content.Schema(
                                type=content.Type.STRING,
                                description="Tasks for the agent"
                            ),
                            "special_instructions": content.Schema(
                                type=content.Type.STRING,
                                description="Remarks about previous search or Special Instructions to Agent"
                            ),
                        },
                        required= ["instruction_to_agent","special_instructions"],
                    ),
                ),
                
                genai.protos.FunctionDeclaration(
                    name="access_discord_agent",
                    description="Has ability to post announcement/event/poll and connect user to moderator/helper request",
                    parameters=content.Schema(
                        type=content.Type.OBJECT,
                        properties={
                            "instruction_to_agent": content.Schema(
                                type=content.Type.STRING,
                                description="Tasks for the agent"
                            ),
                            "special_instructions": content.Schema(
                                type=content.Type.STRING,
                                description="Remarks about previous search or Special Instructions to Agent"
                            ),
                        },
                        required= ["instruction_to_agent","special_instructions"]
                    ),   
                ),
                
                genai.protos.FunctionDeclaration(
                    name="send_bot_feedback",
                    description="Submits user's feedbacks about sparky",
                    parameters=content.Schema(
                        type=content.Type.OBJECT,
                        properties={
                            "feedback": content.Schema(
                                type=content.Type.STRING,
                                description="Feedback by the user"
                            ),
                        },
                    required=["feedback"]
                    ),
                ),
                   
                genai.protos.FunctionDeclaration(
                    name="get_discord_server_info",
                    description="Get Sparky Discord Server related Information",
                    parameters=content.Schema(
                        type=content.Type.OBJECT,
                        properties={
                            "context": content.Schema(
                                type=content.Type.STRING,
                                description="Context of Information"
                            ),
                        },
                          
                    ),
                ),
                 
                genai.protos.FunctionDeclaration(
                    name="access_live_status_agent",
                    description="Has ability to fetch realtime live shuttle/bus, library and StudyRooms status.",
                    parameters=content.Schema(
                        type=content.Type.OBJECT,
                        properties={
                            "instruction_to_agent": content.Schema(
                                type=content.Type.STRING,
                                description="Tasks for Live Status Agent"
                            ),
                            "special_instructions": content.Schema(
                                type=content.Type.STRING,
                                description="Special Instructions to the agent"
                            ),
                        },
                        required= ["instruction_to_agent","special_instructions"],
                    ),
                ),
                 
                genai.protos.FunctionDeclaration(
                    name="get_user_profile_details",
                    description="Get Sparky Discord Server related Information",
                    parameters=content.Schema(
                        type=content.Type.OBJECT,
                        properties={
                            "context": content.Schema(
                                type=content.Type.STRING,
                                description="Context of Information"
                            ),
                        },
                          
                    ),
                ),

                genai.protos.FunctionDeclaration(
                    name="access_google_agent",
                    description="Performs Google Search through to provide rapid result summary",
                    parameters=content.Schema(
                        type=content.Type.OBJECT,
                        properties={
                            "original_query": content.Schema(
                                type=content.Type.STRING,
                                description="Original Query to Search"
                            ),
                            "detailed_query": content.Schema(
                                type=content.Type.STRING,
                                description="Detailed query related to the question"
                            ),
                            "generalized_query": content.Schema(
                                type=content.Type.STRING,
                                description="General query related to the question"
                            ),
                            "relative_query": content.Schema(
                                type=content.Type.STRING,
                                description="Other query related to the question"
                            ),
                            "categories": content.Schema(
                                type=content.Type.ARRAY,
                                items=content.Schema(
                                    type=content.Type.STRING,
                                    enum=[
                                        "libraries_status", 
                                        "shuttles_status", 
                                        "clubs_info", 
                                        "scholarships_info", 
                                        "job_updates", 
                                        "library_resources",  
                                        "classes_info", 
                                        "events_info", 
                                        "news_info", 
                                        "social_media_updates", 
                                    ]
                                ),
                                description="Documents Category Filter"
                            ),

                        },
                        required=["original_query","detailed_query","generalized_query","relative_query","categories"]
                    ),
                ),    
            ],
        ),
    ],
    tool_config={'function_calling_config': 'AUTO'},
)


# %% [markdown]
# #### Global Instance 

# %%
asu_action_agent = ActionModel()
logger.info("\nInitialized ActionAgent Global Instance")

# %% [markdown]
# ## RAG Pipeline
# 
# This `RAGPipeline` class serves as a streamlined interface for processing user queries, leveraging the action agent for determining appropriate responses and implementing robust error handling for reliable operation in the ASU Discord Research Assistant Bot.
# 

# %% [markdown]
# 
# 
# ### Initialization
# - The class is initialized without any specific parameters.
# 
# ### Query Processing
# - The `process_question` method is the main entry point for handling user queries:
#   - Takes a `question` parameter as input.
#   - Utilizes the `asu_action_agent` to determine the appropriate action for the query.
#   - Implements error handling and logging for the processing pipeline.
# 
# ### Integration with Action Agent
# - Delegates the actual query processing to the `asu_action_agent.determine_action` method.
# - This integration allows for flexible action determination based on the input question.
# 
# ### Error Handling
# - Implements comprehensive error logging for query processing failures.
# - Raises exceptions with detailed error messages for debugging purposes.
# 
# ### Logging
# - Utilizes a logger to track the pipeline's operations and potential issues.
# 
# ### Usage
# - An instance of the `RAGPipeline` class is created and stored in the `asu_rag` variable.
# - The pipeline's successful initialization is logged for verification.
# 
# 

# %%

class RAGPipeline:                
    async def process_question(self,question: str) -> str:
        try:
            response = await asu_action_agent.determine_action(question)
            logger.info("\nRAG Pipeline called")
            return response
        except Exception as e:
            logger.error(f"RAG PIPELINE : Error processing question: {str(e)}")
            raise

# %% [markdown]
# #### Global Instance 

# %%
asu_rag =  RAGPipeline()

logger.info("\n----------------------------------------------------------------")
logger.info("\nASU RAG INITIALIZED SUCCESSFULLY")
logger.info("\n---------------------------------------------------------------")

# %% [markdown]
# The `ASUDiscordBot` class provides Discord integration:
# - Handles command registration and event management
# - Implements channel validation and question processing
# - Manages response chunking for long answers
# - Provides error handling and user feedback
# - Includes configuration options for customization
# 

# %% [markdown]
# ## Verification System
# 
# The verification system consists of three main components: VerifyButton, VerificationModal, and OTPVerificationModal. Here's an overview of each:
# 
# 
# 
# 
# 

# %%

class VerifyButton(discord.ui.Button):
    def __init__(self):
        super().__init__(label="Verify", style=discord.ButtonStyle.primary, custom_id="verify_button")

    async def callback(self, interaction: discord.Interaction):
        await interaction.response.send_modal(VerificationModal())

# %% [markdown]
# ### VerificationModal
# 
# This class extends `discord.ui.Modal` and handles the initial step of the verification process.
# 
# 
# 

# %% [markdown]
# 
# - Prompts users to enter their ASU email
# - Validates the email format
# - Generates and sends a one-time password (OTP) to the provided email
# - Creates a button for users to proceed to OTP verification

# %%
class VerificationModal(discord.ui.Modal):
    def __init__(self):
        super().__init__(title="ASU Email Verification")
        self.email = discord.ui.TextInput(
            label="ENTER YOUR ASU EMAIL",
            placeholder="yourname@asu.edu",
            custom_id="email_input"
        )
        self.creds = Credentials.from_service_account_file(
            'client_secret.json',
            scopes=['https://www.googleapis.com/auth/spreadsheets']
        )
        self.service = build('sheets', 'v4', credentials=self.creds)
        self.stored_otp = None
        self.spreadsheet_id = app_config.get_spreadsheet_id()
        self.add_item(self.email)

    async def on_submit(self, interaction: discord.Interaction):
        if  not self.stored_otp:  # First submission - email only
            if self.validate_asu_email(self.email.value):
                self.stored_otp = self.generate_otp()
                self.send_otp_email(self.email.value, self.stored_otp)
                view = discord.ui.View()
                button = discord.ui.Button(label="Enter OTP", style=discord.ButtonStyle.primary)
                async def button_callback(button_interaction):
                    await button_interaction.response.send_modal(OTPVerificationModal(self.stored_otp, self.email.value, self.spreadsheet_id, self.creds, self.service))
                button.callback = button_callback
                view.add_item(button)
                await interaction.response.send_message("OTP has been sent to your email. Click the button below to enter it.", view=view, ephemeral=True)
            else:
                await interaction.response.send_message("Invalid ASU email. Please try again.", ephemeral=True)

    def validate_asu_email(self, email):
        return re.match(r'^[a-zA-Z0-9._%+-]+@asu\.edu$', email) is not None

    def generate_otp(self):
        return ''.join(str(random.randint(0, 9)) for _ in range(6))

    def send_otp_email(self, email, otp):
        sender_email = app_config.get_gmail()
        sender_password = app_config.get_gmail_pass()
        message = MIMEText(f"Your OTP for ASU Discord verification is {otp}")
        message['Subject'] = "ASU Discord Verification OTP"
        message['From'] = sender_email
        message['To'] = email
        
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, email, message.as_string())
          

# %% [markdown]
# ### OTPVerificationModal
# 
# This class extends `discord.ui.Modal` and handles the final step of the verification process.
# 
# 

# %% [markdown]
# 
# - Prompts users to enter the OTP sent to their email
# - Verifies the entered OTP
# - Assigns the "verified" role to the user upon successful verification
# - Updates the Google Sheet with user information

# %%
class OTPVerificationModal(discord.ui.Modal):
    def __init__(self, correct_otp, email,spreadsheet_id,creds,service):
        super().__init__(title="Enter OTP")
        self.correct_otp = correct_otp
        self.email = email
        self.spreadsheet_id = spreadsheet_id
        self.otp = discord.ui.TextInput(
            label="ENTER OTP",
            placeholder="Enter the 6-digit OTP sent to your email",
            custom_id="otp_input"
        )
        self.service = service
        self.creds = creds
        self.add_item(self.otp)

    async def on_submit(self, interaction: discord.Interaction):
        if self.otp.value == self.correct_otp:
            await self.verify_member(interaction, self.email)
        else:
            await interaction.response.send_message("Incorrect OTP. Please try again.", ephemeral=True)

    async def verify_member(self, interaction: discord.Interaction, email):
        verified_role = discord.utils.get(interaction.guild.roles, name="verified")
        if verified_role:
            await interaction.user.add_roles(verified_role)
            await (google_sheet.add_new_user(interaction.user, email))
            await interaction.response.send_message("You have been verified!", ephemeral=True)
        else:
            await interaction.response.send_message("Verification role not found. Please contact an administrator.", ephemeral=True)

    


# %% [markdown]
# ## ASUDiscordBot Class
# 
# This class serves as the main interface between Discord and the bot's backend systems, managing user interactions, command processing, and response delivery.
# 

# %% [markdown]
# #### BotConfig Class
# 
# This dataclass defines the configuration for the Discord bot:
# 

# %% [markdown]
# 
# - `command_name`: Name of the bot command (default: "ask")
# - `command_description`: Description of the bot command
# - `max_question_length`: Maximum allowed length for user questions (300 characters)
# - `max_response_length`: Maximum length for bot responses (2000 characters)
# - `chunk_size`: Size of message chunks for long responses (1900 characters)
# - `token`: Discord bot token retrieved from app configuration
# - `thinking_timeout`: Timeout for bot's "thinking" state (60 seconds)
# 
# 

# %%

@dataclass
class BotConfig:
    """Configuration for Discord bot"""
    command_name: str = "ask"
    command_description: str = "Ask a question about ASU"
    max_question_length: int = 300
    max_response_length: int = 2000
    chunk_size: int = 1900
    token: str = app_config.get_discord_bot_token()  
    thinking_timeout: int = 60



# %% [markdown]
# 
# #### Initialization
# - Initializes with a RAG pipeline and optional configuration
# - Sets up Discord client, command tree, and service for web interactions
# 
# #### Key Methods
# - `_register_commands`: Sets up the "ask" command
# - `_register_events`: Handles bot ready event
# - `_handle_ask_command`: Processes user questions
# - `_validate_channel`: Ensures command is used in correct channel
# - `_validate_question_length`: Checks if question is within length limits
# - `_process_and_respond`: Handles question processing and response generation
# - `_send_chunked_response`: Sends long responses in chunks
# - `_send_error_response`: Handles error messaging
# - `_handle_ready`: Syncs command tree and sets up verification button
# - `start`: Starts the Discord bot
# - `close`: Closes the Discord bot connection
# 
# 

# %%
class ASUDiscordBot:
    
    """Discord bot for handling ASU-related questions"""

    def __init__(self, rag_pipeline, config: Optional[BotConfig] = None):
        """
        Initialize the Discord bot.
        
        Args:
            rag_pipeline: RAG pipeline instance
            config: Optional bot configuration
        """
        logger.info("\nInitializing ASUDiscordBot")
        self.config = config or BotConfig()
        self.rag_pipeline = rag_pipeline
        
        # Initialize Discord client
        
        self.client = discord_state.get('discord_client')
        self.tree = app_commands.CommandTree(self.client)
        self.guild = self.client.get_guild(1256076931166769152)
        self.service = Service(ChromeDriverManager().install())
        
        # Register commands and events
        self._register_commands()
        self._register_events()
   
    def _register_commands(self) -> None:
        """Register Discord commands"""
        
        @self.tree.command(
            name=self.config.command_name,
            description=self.config.command_description
        )
        async def ask(interaction: discord.Interaction, question: str):
            await self._handle_ask_command(interaction, question)
        
    def _register_events(self) -> None:
        """Register Discord events"""
        
        @self.client.event
        async def on_ready():
            await self._handle_ready()
            
    async def _handle_ask_command(
        self,
        interaction: discord.Interaction,
        question: str) -> None:
        """
        Handle the ask command.
        
        Args:
            interaction: Discord interaction
            question: User's question
        """
        logger.info(f"User {interaction.user.name} asked: {question}")
        user = interaction.user
        user_id= interaction.user.id
        request_in_dm = isinstance(interaction.channel, discord.DMChannel)
        self.guild = self.client.get_guild(1256076931166769152)
        target_guild = self.client.get_guild(1256076931166769152)
        user_has_mod_role= None
        member = None
        user_voice_channel_id=None
        # Reset all states

        if target_guild:
            try:
                member = await target_guild.fetch_member(interaction.user.id)
                if member:
                    required_role_name = "mod" 
                    user_has_mod_role = any(
                        role.name == required_role_name for role in member.roles
                    )
                    
                    # Check voice state
                    if member.voice:
                        user_voice_channel_id = member.voice.channel.id
                else:
                    return "You are not part of Sparky Discord Server. Access to command is restricted."

                    
            except discord.NotFound:
                return "You are not part of Sparky Discord Server. Access to command is restricted."
        await asu_scraper.__login__(app_config.get_handshake_user(),app_config.get_handshake_pass() )
        discord_state.update(user=user, target_guild=target_guild, request_in_dm=request_in_dm,user_id=user_id, guild_user = member, user_has_mod_role=user_has_mod_role,user_voice_channel_id=user_voice_channel_id)
        firestore.update_collection("direct_messages" if request_in_dm else "guild_messages" )
         
        try:
            if not await self._validate_channel(interaction):
                return
            if not await self._validate_question_length(interaction, question):
                return
            
            await self._process_and_respond(interaction, question)

        except Exception as e:
            error_msg = f"Error processing ask command: {str(e)}"
            logger.error(error_msg, exc_info=True)
            await self._send_error_response(interaction)

    async def _validate_channel(self, interaction: discord.Interaction) -> bool:
        """Validate if command is used in correct channel"""
        if not discord_state.get('request_in_dm') and interaction.channel.id != 1323387010886406224:
            await interaction.response.send_message(
                "Please use this command in the designated channel: #general",
                ephemeral=True
            )
            return False
        return True

    async def _validate_question_length(
        self,
        interaction: discord.Interaction,
        question: str) -> bool:
        """Validate question length"""
        if len(question) > self.config.max_question_length:
            await interaction.response.send_message(
                f"Question too long ({len(question)} characters). "
                f"Please keep under {self.config.max_question_length} characters.",
                ephemeral=True
            )
            return False
        return True

    async def _process_and_respond(
        self,
        interaction: discord.Interaction,
        question: str ) -> None:
        """Process question and send response"""
        try:
            
            await interaction.response.defer(thinking=True)
            global task_message
            task_message = await interaction.edit_original_response(content="⋯ Understanding your question")
            await utils.start_animation(task_message)
            response = await self.rag_pipeline.process_question(question)
            await self._send_chunked_response(interaction, response)
            logger.info(f"Successfully processed question for {interaction.user.name}")
            await asu_store.store_to_vector_db()
            await (google_sheet.increment_function_call(discord_state.get('user_id'), 'G'))
            await (google_sheet.increment_function_call(discord_state.get('user_id'), 'N'))
            await (google_sheet.update_user_column(interaction.user.id, 'E', question))
            await (google_sheet.update_user_column(interaction.user.id, 'F', response))
            
            await google_sheet.perform_updates()
            
            firestore.update_message("user_message", question)
            document_id = await firestore.push_message()
            logger.info(f"Message pushed with document ID: {document_id}")

        except asyncio.TimeoutError:
            logger.error("Response generation timed out")
            await self._send_error_response(
                interaction,
                "Sorry, the response took too long to generate. Please try again."
            )
        except Exception as e:
            logger.error(f"Error processing question at discord class: {str(e)}", exc_info=True)
            await self._send_error_response(interaction)

    async def setup_verify_button(self):
        channel = self.client.get_channel(1323386003896926248)  # Verify channel ID
        if channel:
            view = discord.ui.View(timeout=None)
            view.add_item(VerifyButton())
            await channel.send("Click here to verify", view=view)

    async def _send_chunked_response( self,interaction: discord.Interaction,response: str) -> None:
        """Send response in chunks if needed"""
        try:
            ground_sources = utils.get_ground_sources()
            # Create buttons for each URL
            buttons = []
            for url in ground_sources:
                domain = urlparse(url).netloc
                button = discord.ui.Button(label=domain, url=url, style=discord.ButtonStyle.link)
                buttons.append(button)
            # Custom link for feedbacks
            button = discord.ui.Button(label="Feedback", url="https://discord.com/channels/1256076931166769152/1323386415337177150", style=discord.ButtonStyle.link)
            buttons.append(button)

            view = discord.ui.View()
            for button in buttons:
                view.add_item(button)

            if len(response) > self.config.max_response_length:
                chunks = [
                    response[i:i + self.config.chunk_size]
                    for i in range(0, len(response), self.config.chunk_size)
                ]
                global task_message
                await utils.stop_animation(task_message, chunks[0])
                for chunk in chunks[1:-1]:
                    await interaction.followup.send(content=chunk)
                await interaction.followup.send(content=chunks[-1], view=view)
            else:
                await utils.stop_animation(task_message, response,View=view)
            
            utils.clear_ground_sources()

        except Exception as e:
            logger.error(f"Error sending response: {str(e)}", exc_info=True)
            await self._send_error_response(interaction)

    async def _send_error_response(
        self,
        interaction: discord.Interaction,
        message: str = "Sorry, I encountered an error processing your question. Please try again.") -> None:
        """Send error response to user"""
        try:
            if not interaction.response.is_done():
                await interaction.response.send_message(
                    content=message,
                    ephemeral=True
                )
            else:
                await interaction.followup.send(
                    content=message,
                    ephemeral=True
                )
        except Exception as e:
            logger.error(f"Error sending error response: {str(e)}", exc_info=True)

    async def _handle_ready(self):
        try:
            await self.tree.sync()
            logger.info(f'Bot is ready! Logged in as {self.client.user}')
            await self.setup_verify_button()  # Set up the verify button when the bot starts
        except Exception as e:
            logger.error(f"Error in ready event: {str(e)}", exc_info=True)

    async def start(self) -> None:
        """Start the Discord bot"""
        try:
            await self.client.start(self.config.token)
        except Exception as e:
            logger.error(f"Failed to start bot: {str(e)}", exc_info=True)
            raise

    async def close(self) -> None:
        """Close the Discord bot"""
        try:
            await self.client.close()
        except Exception as e:
            logger.error(f"Error closing bot: {str(e)}", exc_info=True)

def run_discord_bot(rag_pipeline, config: Optional[BotConfig] = None):
    """Run the Discord bot"""
    bot = ASUDiscordBot(rag_pipeline, config)
    
    async def run():
        try:
            await bot.start()
        except KeyboardInterrupt:
            logger.info("\nBot shutdown requested")
            await bot.close()
        except Exception as e:
            logger.error(f"Bot error: {str(e)}", exc_info=True)
            await bot.close()

    # Run the bot
    asyncio.run(run())

if __name__ == "__main__":
    config = BotConfig(
        token=app_config.get_discord_bot_token(),
    )
    run_discord_bot(asu_rag, config)


