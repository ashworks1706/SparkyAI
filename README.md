# SparkyAI - University Copilot

This repository has been recently made public, the old repository had exposed credential issues.

A Discord bot designed to assist Arizona State University (ASU) students with access to resources, including news, events, scholarships, courses, and more.

![{2B61349D-750C-4418-A76E-15CB3AAB0B8B}](https://github.com/user-attachments/assets/642fd6d6-5232-4347-b1dc-3e78d3d0c758)

![{5B0799E0-D15C-4017-867A-F1DEB1FDA2DC}](https://github.com/user-attachments/assets/e19a175a-2c70-4af8-88d8-6303b9729cda)

![{2DDB8F4F-5F0E-4828-8FDD-847E67C40A65}](https://github.com/user-attachments/assets/7fbce508-e180-4f8f-9d7f-11feac5757e8)

SparkyAI leverages a complex architecture to deliver accurate and context-aware responses:

- **Retrieval-Augmented Generation (RAG)**: Combines vector search with large language models for precise information retrieval.
- **Multi-Agent System**: Utilizes specialized AI agents or targeted task execution.
- **Vector Database**: Implements Qdrant for efficient semantic search and document retrieval.

## Features

- **Maximum Inner Product Search (MIPS)**: Optimized vector similarity search for large-scale datasets.
- **RAPTOR Retrieval**: Hierarchical document representation for nuanced information retrieval.
- **Multi-step Reasoning**: Synthesizes information from multiple sources to answer complex queries.
- **Dynamic Content Extraction**: Combines Selenium-based scraping with AI-powered content refinement.
- **Automatic Version Control**: Intelligent document updating based on timestamp comparisons.
- **Cross-Encoder Reranking**: Implements a cross-encoder model to rerank initial retrieval results, improving the relevance of top results.
- **HNSWlib Integration**: Incorporates a graph-based approach for efficient in-memory searches, complementing existing vector search methods.
- **ScaNN Integration**: Utilizes Google's ScaNN library for scalable and efficient handling of large-scale, high-dimensional vector datasets.
- **Custom Embedding Model**: Utilizes the BAAI/bge-large-en-v1.5 model for high-quality text embeddings.
- **Cross-Encoder Reranking**: Implements the cross-encoder/ms-marco-MiniLM-L-6-v2 model to rerank initial retrieval results, improving the relevance of top results

## Performance Optimizations

- **Batch Processing**: Efficient document storage and retrieval in configurable batches.
- **Caching Mechanisms**: Implements strategic caching for frequently accessed data.
- **Asynchronous Operations**: Utilizes asyncio for non-blocking I/O operations.
- **Retry Mechanisms**: Robust error handling with configurable retry attempts for critical operations.
- **Multi-Method Search**: Combines RAPTOR, similarity search, MIPS, and ScaNN for comprehensive and efficient information retrieval.
- **Result Deduplication**: Implements intelligent merging and deduplication of search results from multiple methods.

## Architecture

![image](https://github.com/user-attachments/assets/8fb16d4d-387c-402b-9b42-a8a25138dcc4)

![image](https://github.com/user-attachments/assets/c2228237-c7e2-4f50-84e4-30ca07c7d2f0)

## Key Features

The code implements the following key features:

### Web Scraping & Summarization

The `ASUWebScraper` class handles web scraping of ASU pages:

- Uses Selenium for dynamic content and BeautifulSoup for static HTML parsing
- Implements methods for scraping specific ASU resources like course catalogs, library resources, and job postings
- Extracts content using both Selenium-based scraping and Jina AI-powered content extraction

The `DataPreprocessor` class handles summarization:

- Cleans and structures scraped text using NLP techniques like tokenization and lemmatization
- Uses AI agents (likely the Gemini model) to refine and summarize content

### Discord Integration

The `DiscordState` class manages the Discord bot's state and interactions:

- Initializes Discord intents for message content and member access
- Tracks user information, roles, and channel details
- Provides methods to update and retrieve bot state

### AI-Driven Information Retrieval

The system uses a Retrieval-Augmented Generation (RAG) architecture:

- The `VectorStore` class manages document storage and retrieval using Qdrant vector database
- Implements semantic search capabilities using HuggingFace embeddings
- The `Utils` class contains methods for similarity search and database querying
- The `RAPTOR ` class reranks and creates a tree of documents

### Utils Class Enhancements

- **Multi-Method Search**: Orchestrates searches across RAPTOR, similarity, and MIPS methods.
- **Result Merging**: Intelligently combines and deduplicates results from various search methods.
- **Ground Source Management**: Tracks and manages unique source URLs for comprehensive information retrieval.
- **Caching**: Implements query and document ID caching for improved performance.

### Webhooks

The system uses various sets of custom web scraping functions with search query manipulation to fetch most results-

- The `ASUWebScraper` class can be extended to fetch data from different ASU platforms

## Database Integration

The ASU Discord Bot utilizes two database systems for different purposes:

### Vector Store Operations

- **Qdrant Integration**: Utilizes Qdrant for efficient vector storage and retrieval.
- **MIPS Search**: Implements Maximum Inner Product Search for optimized similarity queries.
- **HNSWlib Indexing**: Builds and uses HNSW indexes for fast approximate nearest neighbor search.
- **Automatic Index Building**: Dynamically constructs search indexes for improved query performance.

### Google Sheets Database

The `GoogleSheet` class manages interactions with a Google Sheets database, primarily used for moderator oversight and user tracking.

Key features:

- User Management: Stores and retrieves user information, including Discord IDs, names, and email addresses.
- Function Call Tracking: Increments counters for various bot function calls, allowing moderators to monitor usage patterns.
- Data Retrieval: Provides methods to fetch all users or specific user data.
- Data Updates: Allows updating user-specific information in the spreadsheet.

Implementation details:

- Uses the Google Sheets API for read and write operations.
- Implements methods like `get_all_users()`, `add_new_user()`, and `update_user_column()`.
- Provides error handling and logging for database operations.

### Firestore Database

The `Firestore` class manages interactions with Google's Firestore, used for storing bot-related data and chat histories.

Key features:

- Chat History: Stores complete conversation histories between users and the bot.
- Message Categorization: Organizes messages by different agent types (action, discord, google, live status, search).
- Real-time Updates: Leverages Firestore's real-time capabilities for instant data synchronization.

Implementation details:

- Initializes a Firestore client using Firebase Admin SDK.
- Implements methods to update collections, add new messages, and retrieve chat histories.
- Stores messages with timestamps and user IDs for comprehensive tracking.

### The modular architecture allows for easy extension:

- Multiple agent classes (e.g., `SuperiorAgent`,  `RAGSearchAgent`, `ShuttleStatusAgent`) can be customized for different tasks
- The `AppConfig` class centralizes configuration management, making it easy to add new features
- The use of asynchronous programming (async/await) throughout the codebase allows for efficient handling of concurrent operations

## Technologies Used

- **AI/ML**: Gemini, LangChain, TensorFlow
- **NLP**: Hugging Face Transformers, NLTK
- **Embeddings**: BAAI/bge-large-en-v1.5 model
- **Cross Encoder**:  cross-encoder/ms-marco-MiniLM-L-6-v2
- **Vector Search**: Qdrant
- **Web Scraping**: Selenium, BeautifulSoup4
- **APIs**: Discord API, Firebase API
- **Databases**: Firestore, Notion
- **Asynchronous Programming**: asyncio
- **Containerization**: Docker

## Agent Descriptions

| Name                                      | What It Does                                                                                                            |
| ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| **Superior Agent**                  | Handles main messages, decides on direct responses or function calls, and utilizes multiple agents/functions as needed. |
| **RAG Search Agent**                | Performs search over RAG knowledge base, has also access to general Google search utilityfor queries                   |
| **News Media Agent**                | Access to asu social media and asu news                                                                                 |
| **Shuttle Status Agent**            | access to asu campus rider portal                                                                                       |
| **Discord Agent**                   | access to discord channels : announcements, feedbacks, customer service, polls, posts,                                  |
| **Library Agent**                   | access to library.asu and live rooms status asu                                                                         |
| **Sports Agent**                    | access to sundevilathletics.asu                                                                                         |
| **Student Jobs Agent**              | access to workday.asu                                                                                                   |
| **Student Clubs Events<br />Agent** | access to sundevilsync.asu                                                                                              |
|                                           | access to scholarships.asu                                                                                              |

### Finetune

We are finetuning gemini 1.5 flash model (Superior Agent) with our custom dataset containing 560 examples of different interactions between agents with humans aswell as agents with other agents to increase the accuracy of reasoning aswell as general responses to students:

| **Category**               | **Description**                                                                                 | **Proportion (%)** | **Number of Examples** |
| -------------------------------- | ----------------------------------------------------------------------------------------------------- | ------------------------ | ---------------------------- |
| **Factual Questions**      | Questions requiring concise, factual answers, such as "What are the library hours?"                   | 23.6%                    | 130                          |
| **Action-Based Questions** | Queries requiring JSON function calls, such as "Find the latest scholarships."                        | 32.7%                    | 180                          |
| **Hybrid Questions**       | Queries needing reasoning + a function call, such as "Can you summarize this and get related events?" | 32.7%                    | 180                          |
| **Jailbreak Commands**     | Edge-case inputs requiring safe acknowledgment or refusal                                             | 10.9%                    | 60                           |

### Extensible Design

## Getting Started

Follow these steps to set up and run the ASU Discord Bot on your local machine.

### 1. Clone the Repository

First, clone the repository to your local machine using Git.

```bash
git clone https://github.com/ashworks1706/SparkyAI.git
cd sparkyai
```

---

### 2. Set Up a Virtual Environment

Create and activate a Python virtual environment to isolate dependencies.

#### On Linux/Mac:

```bash
python3 -m venv venv
source venv/bin/activate
```

#### On Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

---

### 3. Install Dependencies

Install all required Python.12.3 packages from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

---

### 4. Configure API Keys and Credentials

The bot requires several API keys and configuration files to function properly.

1. **Create a `appConfig.json` file** in the `config/` folder with the following structure [Example](/__sample__appConfig.json):
2. **Add Firebase API credentials** [Example](/__sample_firebase_secret.json):

   - Download the `firebase_secret.json` file from your Firebase Cloud Console.
   - Place it in the `config/` folder.

---

### 5. Install Docker

The project uses Docker for containerization, which simplifies deployment and ensures consistent environments across different systems.

Follow [Docker installation instructions](https://docs.docker.com/get-docker/) for your operating system:

- **Windows**: Install Docker Desktop with WSL2 backend
- **macOS**: Install Docker Desktop
- **Linux**: Install Docker and Docker Compose separately

For Linux users, install Docker Compose if not included with your Docker installation:

```bash
sudo apt-get install docker-compose  # Debian/Ubuntu
sudo dnf install docker-compose      # Fedora
```

---

### 6. Run the Bot with Docker

The project has been containerized with Docker to handle all dependencies (including Chrome/Chromedriver for web scraping) and services automatically. The setup is optimized for cross-platform compatibility (Linux, Windows WSL, macOS).

#### Build and start all services:

```bash
# For Docker Compose v2 (newer installations)
docker compose up -d

# For Docker Compose v1 (older installations)
docker-compose up -d
```

This command will:

- Build the Docker image with all required dependencies
- Start the main application (Discord bot)
- Start the background fetch service
- Set up and run the Qdrant vector database
- Configure networking between services

#### View application logs:

```bash
# View logs for the main Discord bot
docker compose logs -f app

# View logs for the background fetch service
docker compose logs -f background-fetch
```

#### Stop all services:

```bash
docker compose down
```

If everything is set up correctly, you should see logs indicating that the bot has started successfully when you check the app logs.

#### Platform-Specific Notes:

- **Windows/WSL**: The Docker setup includes virtual display configuration for headless Chrome
- **macOS M1/M2/M3 Users**: The image will automatically use the ARM64 architecture
- **Linux**: Includes all necessary system libraries for Chrome

---

### Troubleshooting Common Issues

1. **Docker Container Issues:**

   - Ensure Docker and Docker Compose are installed and running.
   - Check container status with `docker-compose ps`
   - View specific container logs with:
     ```bash
     docker-compose logs -f service_name
     ```
   - Restart services with `docker-compose restart service_name`
2. **Missing API Keys:**

   - Double-check that your `config/appConfig.json` file is correctly formatted.
   - Ensure `config/firebase_secret.json` is properly set up.
   - Make sure these files are in the config directory before building Docker images.
3. **Network Issues:**

   - Ensure ports 6333 and 6334 are available for Qdrant.
   - Check if services can communicate using the Docker network.
   - Verify no firewall is blocking container communications.
4. **Volume Permissions:**

   - Check permissions on mounted volumes if you encounter access issues:
     ```bash
     ls -la config/ logs/ qdrant_storage/
     ```
   - Fix permissions if needed: `sudo chown -R $(id -u):$(id -g) config/ logs/ qdrant_storage/`

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any ideas or improvements.

## Citations

This project draws inspiration from and builds upon the following research papers:

1. [Amagata, D., &amp; Hara, T. (2023). Reverse Maximum Inner Product Search: Formulation, Algorithms, and Analysis. ACM Transactions on the Web, 17(4), 1-23](https://dl.acm.org/doi/pdf/10.1145/3587215)
2. [Sun, P. (2020). Announcing ScaNN: Efficient Vector Similarity Search. Google Research Blog](https://research.google/blog/announcing-scann-efficient-vector-similarity-search/)
3. [Guo, R., Sun, P., Lindgren, E., Geng, Q., Simcha, D., Chern, F., &amp; Kumar, S. (2020). Accelerating Large-Scale Inference with Anisotropic Vector Quantization. International Conference on Machine Learning (ICML)](https://arxiv.org/pdf/1908.10396)
4. [Dong, W., Moses, C., &amp; Li, K. (2024). SOAR: Improved Indexing for Approximate Nearest Neighbor Search. arXiv preprint arXiv:2404.00774](https://www.arxiv.org/pdf/2411.06158)
5. [Kandpal, N., Jiang, H., Kong, X., Teng, J., &amp; Chen, J. (2024). RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval. arXiv preprint arXiv:2401.18059v1](https://arxiv.org/pdf/2401.18059v1)
6. [Guo, R., Kumar, S., Choromanski, K., &amp; Simcha, D. (2019). Quantization based Fast Inner Product Search. arXiv preprint arXiv:1509.01469](https://arxiv.org/pdf/1509.01469)

These papers have significantly contributed to the field of vector similarity search, maximum inner product search (MIPS), and efficient indexing techniques, which are fundamental to this project's approach.

## License

This project is licensed under the [MIT License](LICENSE).

## Immediate Updates Needed

- Develop ML Ops Pipeline
- Update and Migrate to SDC platform

## Contact

- Author: Ash
- Portfolio: [ashworks.dev](https://ashworks.dev)
- LinkedIn: [linkedin.com/ashworks](https://linkedin.com/ashworks)
