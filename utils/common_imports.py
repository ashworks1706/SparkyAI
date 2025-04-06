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
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings 
from langchain_qdrant import  QdrantVectorStore 
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import ScaNN
from langchain_community.document_loaders import UnstructuredURLLoader
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams, OptimizersConfigDiff
from huggingface_hub import login
from sentence_transformers import CrossEncoder
import numpy as np
from sklearn.cluster import KMeans
from qdrant_client.http.models import Filter, FieldCondition, MatchAny

## Google 
from google.protobuf.json_format import MessageToDict
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

from google import genai as genai_vertex
from google.genai import types
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch

from config.app_config import AppConfig
from config.bot_config import BotConfig

from rag.data_processor import DataPreprocessor
from rag.agents import Agents
from database.firestore import Firestore
from utils.discord_state import DiscordState
from utils.utils import Utils
from discord_bot import ASUDiscordBot
from rag.web_scrape import ASUWebScraper
from rag.vector_store import VectorStore
from rag.group_chat import GroupChat
