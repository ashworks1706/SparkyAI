# ASU Discord Bot - (Public)

This repository has been recently made public, the old repository had exposed credential issues.

A Discord bot designed to assist Arizona State University (ASU) students with access to resources, including news, events, scholarships, courses, and more.

![{2B61349D-750C-4418-A76E-15CB3AAB0B8B}](https://github.com/user-attachments/assets/642fd6d6-5232-4347-b1dc-3e78d3d0c758)

![{5B0799E0-D15C-4017-867A-F1DEB1FDA2DC}](https://github.com/user-attachments/assets/e19a175a-2c70-4af8-88d8-6303b9729cda)

![{2DDB8F4F-5F0E-4828-8FDD-847E67C40A65}](https://github.com/user-attachments/assets/7fbce508-e180-4f8f-9d7f-11feac5757e8)

## Advanced Architecture

![image](https://github.com/user-attachments/assets/18e2c5fc-b777-4b22-b63e-5bd97ffde5cd)

![image](https://github.com/user-attachments/assets/8fb16d4d-387c-402b-9b42-a8a25138dcc4)

![image](https://github.com/user-attachments/assets/c2228237-c7e2-4f50-84e4-30ca07c7d2f0)

![image](https://github.com/user-attachments/assets/1ab12d43-3e37-4020-bec1-0dc8dc4ea1db)


SparkyAI leverages a complex architecture to deliver accurate and context-aware responses:

- **Retrieval-Augmented Generation (RAG)**: Combines vector search with large language models for precise information retrieval.
- **Multi-Agent System**: Utilizes specialized AI agents (Action, Google, Search, Live Status, Discord) for targeted task execution.
- **Vector Database**: Implements Qdrant for efficient semantic search and document retrieval.

## Advanced Features

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

### Utils Class Enhancements

- **Multi-Method Search**: Orchestrates searches across RAPTOR, similarity, MIPS, and ScaNN methods.
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

### Finetune

We are finetuning gemini 1.5 flash model (Action Agent) with our custom dataset containing 560 examples of different interactions between agents with humans aswell as agents with other agents to increase the accuracy of reasoning aswell as general responses to students:

| **Category**               | **Description**                                                                                 | **Proportion (%)** | **Number of Examples** |
| -------------------------------- | ----------------------------------------------------------------------------------------------------- | ------------------------ | ---------------------------- |
| **Factual Questions**      | Questions requiring concise, factual answers, such as "What are the library hours?"                   | 23.6%                    | 130                          |
| **Action-Based Questions** | Queries requiring JSON function calls, such as "Find the latest scholarships."                        | 32.7%                    | 180                          |
| **Hybrid Questions**       | Queries needing reasoning + a function call, such as "Can you summarize this and get related events?" | 32.7%                    | 180                          |
| **Jailbreak Commands**     | Edge-case inputs requiring safe acknowledgment or refusal                                             | 10.9%                    | 60                           |

### Extensible Design

The modular architecture allows for easy extension:

- Multiple agent classes (e.g., `ActionAgent`, `SearchAgent`, `GoogleAgent`) can be customized for different tasks
- The `AppConfig` class centralizes configuration management, making it easy to add new features
- The use of asynchronous programming (async/await) throughout the codebase allows for efficient handling of concurrent operations

## Technologies Used

- **AI/ML**: Gemini Vertex AI, LangChain, TensorFlow
- **NLP**: Hugging Face Transformers, NLTK
- **Embeddings**: BAAI/bge-large-en-v1.5 model
- **Cross Encoder**:  cross-encoder/ms-marco-MiniLM-L-6-v2
- **Vector Search**: Qdrant, FAISS
- **Web Scraping**: Selenium, BeautifulSoup4
- **APIs**: Discord API, Google Sheets API, Google Cloud Storage
- **Databases**: Firestore, Google Sheets (for moderation)
- **Asynchronous Programming**: asyncio
- **Containerization**: Docker

## Agent Descriptions

| Name                        | Importance                                           | What It Does                                                                                                              | Functions                                                                                                                                                                                                                                                                            |
| --------------------------- | ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Action Agent**      | Central coordinator for user interactions            | Handles main messages, decides on direct responses or function calls, and utilizes multiple agents/functions as needed.   | -`access_search_agent<br>`- `access_discord_agent<br>`- `access_google_agent<br>`- `access_live_status_agent<br>`- `get_discord_server_info<br>`- `get_user_profile_details`                                                                                             |
| **Google Agent**      | Specialized search functionality                     | Performs Google searches for queries; defers to Action Agent for complex queries requiring the Search Agent.              | -`google_search_tool`                                                                                                                                                                                                                                                              |
| **Search Agent**      | Executes functions for queries from the Action Agent | Executes specific functions to retrieve information based on queries passed by the Action Agent; responds in JSON format. | -`get_latest_club_information<br>`- `get_latest_event_updates<br>`- `get_latest_news_updates<br>`- `get_latest_social_media_updates<br>`- `get_latest_sport_updates<br>`- `get_library_resources<br>`- `get_latest_scholarships<br>`- `get_latest_class_information` |
| **Live Status Agent** | Manages live status-related queries                  | Executes live status functions for queries from the Action Agent; responds in JSON format.                                | -`get_live_library_status<br>`- `get_live_shuttle_status`                                                                                                                                                                                                                        |
| **Discord Agent**     | Handles Discord-specific functionalities             | Executes Discord-related functions for queries from the Action Agent; responds in JSON format.                            | -`notify_discord_helpers<br>`- `notify_moderators` `<br>`- `create_discord_forum_post<br>`- `create_discord_announcement<br>`- `create_discord_event<br>`- `create_discord_poll`                                                                                       |

## Getting Started

1. Clone the repository:

   ```bash
   git clone https://github.com/somwrks/SparkyAI.git
   cd sparkyai
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Add API keys:

   - Create `config.json` with keys for Google Search, Discord, and ASU webhooks.
   - Add `clientsecret.json` for GoogleSheet API.
4. Run the bot:

   ```bash
   python main.py
   ```

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

## Contact

- Author: Ash
- Portfolio: [ashworks.dev](https://ashworks.dev)
- LinkedIn: [linkedin.com/ashworks](https://linkedin.com/ashworks)
