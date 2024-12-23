# ASU Discord Bot - (Public)

This repository has been recently made public, the old repository had exposed credential issues.

A Discord bot designed to assist Arizona State University (ASU) students with access to resources, including news, events, scholarships, courses, and more.

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

### Webhooks

The system uses various sets of custom web scraping functions with search query manipulation to fetch most results-

- The `ASUWebScraper` class can be extended to fetch data from different ASU platforms


## Database Integration

The ASU Discord Bot utilizes two database systems for different purposes:

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

We are finetuning gemini 1.5 flash model with our custom dataset containing 128,000 examples of different interactions between agents with humans aswell as agents with other agents to increase the accuracy of reasoning aswell as general responses to students:

| **Category**                  | **Description**                             | **Proportion (%)** | **Number of Examples** |
|--------------------------------|---------------------------------------------|---------------------|-------------------------|
| **Factual Questions**          | Questions requiring concise, factual answers, such as "What are the library hours?" | 30%                 | 38,400                 |
| **Action-Based Questions**     | Queries requiring JSON function calls, such as "Find the latest scholarships."      | 25%                 | 32,000                 |
| **Hybrid Questions**           | Queries needing reasoning + a function call, such as "Can you summarize this and get related events?" | 20%                 | 25,600                 |
| **Agent Communication Responses** | Multi-agent interaction responses combining reasoning and function calls           | 15%                 | 19,200                 |
| **Jailbreak Commands**         | Edge-case inputs requiring safe acknowledgment or refusal                            | 10%                 | 12,800                 |


### Extensible Design

The modular architecture allows for easy extension:

- Multiple agent classes (e.g., `ActionAgent`, `SearchAgent`, `GoogleAgent`) can be customized for different tasks
- The `AppConfig` class centralizes configuration management, making it easy to add new features
- The use of asynchronous programming (async/await) throughout the codebase allows for efficient handling of concurrent operations


## Technologies Used

- Python
- Gemini Vertex AI, LangChain, TensorFlow, Selenium
- Google Sheets API, Discord API
- Qdrant Vector DB, Docker, FireStore, Google Cloud Storage, GDE

## Agent Descriptions


| Name | Importance | What It Does | Functions |
|------|------------|--------------|-----------|
| **Action Agent** | Central coordinator for user interactions | Handles main messages, decides on direct responses or function calls, and utilizes multiple agents/functions as needed. | - `access_search_agent`<br>- `access_discord_agent`<br>- `get_google_results`<br>- `access_live_status_agent`<br>- `get_discord_server_info`<br>- `get_user_profile_details` |
| **Google Agent** | Specialized search functionality | Performs Google searches for queries; defers to Action Agent for complex queries requiring the Search Agent. | - `google_search_tool` |
| **Search Agent** | Executes functions for queries from the Action Agent | Executes specific functions to retrieve information based on queries passed by the Action Agent; responds in JSON format. | - `get_latest_club_information`<br>- `get_latest_event_updates`<br>- `get_latest_news_updates`<br>- `get_latest_social_media_updates`<br>- `get_latest_sport_updates`<br>- `get_library_resources`<br>- `get_latest_scholarships`<br>- `get_latest_class_information` |
| **Live Status Agent** | Manages live status-related queries | Executes live status functions for queries from the Action Agent; responds in JSON format. | - `get_live_library_status`<br>- `get_live_shuttle_status` |
| **Discord Agent** | Handles Discord-specific functionalities | Executes Discord-related functions for queries from the Action Agent; responds in JSON format. | - `notify_discord_helpers`<br>- `notify_moderators`<br>- `start_recording_discord_call`<br>- `create_discord_forum_post`<br>- `create_discord_announcement`<br>- `create_discord_event`<br>- `create_discord_poll` |

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

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

- Author: Ash (Som)
- Portfolio: [somwrks.com](https://somwrks.com)
- LinkedIn: [linkedin.com/somwrks](https://linkedin.com/somwrks)

