# SparkyAI — ASU Discord Bot + agentic RAG system (not maintained)

SparkyAI is an old, no-longer-maintained project: a Discord bot built to help Arizona State University (ASU) students find information and complete small, concrete tasks across ASU’s web ecosystem. It combines retrieval-augmented generation (RAG) with a lightweight multi-agent/tool-routing architecture, plus a background ingestion pipeline that keeps a knowledge base fresh as pages change.

This repository is public because people still discover it and ask how it works. The short answer is: it is both (1) an ASU student-facing Discord bot, and (2) a practical exploration of agent collaboration and advanced RAG retrieval strategies under real latency and operational constraints.

This repository has been recently made public; the older repository had credential exposure issues.

![{2B61349D-750C-4418-A76E-15CB3AAB0B8B}](https://github.com/user-attachments/assets/642fd6d6-5232-4347-b1dc-3e78d3d0c758)

![{5B0799E0-D15C-4017-867A-F1DEB1FDA2DC}](https://github.com/user-attachments/assets/e19a175a-2c70-4af8-88d8-6303b9729cda)

![{2DDB8F4F-5F0E-4828-8FDD-847E67C40A65}](https://github.com/user-attachments/assets/7fbce508-e180-4f8f-9d7f-11feac5757e8)

## What it was trying to solve

Student queries are rarely just “search and answer”. They are messy, time-sensitive, and often require multi-step work: browsing multiple official pages, extracting the relevant parts, combining them, and responding with grounded context. The engineering constraint is that you need to do this fast enough for a chat interface, and safely enough that the bot does not hallucinate policies, deadlines, or locations.

SparkyAI approached this with a retrieval-first design. For many requests, the model’s job is not to invent an answer, but to select, compress, and cite the right pieces of ASU-specific context, and only then generate a response.

## How it works (conceptual forward pass)

A user message comes in through Discord. A supervisor-style agent decides whether to answer directly or to delegate to a specialized agent/tool. Delegation can mean pulling from the ASU knowledge base (RAG), running targeted web extraction, or calling a domain-specific workflow (for example, shuttles, library information, student jobs, clubs/events, and similar categories).

On the retrieval side, the pipeline embeds the query, pulls candidate documents from a vector database (Qdrant), optionally applies multiple retrieval strategies, and then reranks the best candidates using a cross-encoder. A simple mental model is that the system tries to maximize relevance under a fixed latency budget: first retrieve a broad set cheaply, then spend compute on reranking only the top candidates.

If you prefer a compact scoring view:

- Dense retrieval (candidate generation):

  $$s_{\text{dense}}(q, d) = \langle E(q), E(d) \rangle$$

- Cross-encoder reranking (precision at top-k):

  $$s_{\text{rerank}}(q, d) = \text{CE}(q, d)$$

The response is then synthesized from the retrieved context, and state is persisted for moderation, analytics, and debugging (for example, chat history and function-call usage).

## Architecture (implementation pointers)

![image](https://github.com/user-attachments/assets/8fb16d4d-387c-402b-9b42-a8a25138dcc4)

![image](https://github.com/user-attachments/assets/c2228237-c7e2-4f50-84e4-30ca07c7d2f0)

If you are reading the code, the fastest way to orient yourself is by entry points and “loops”:

- `discord_bot.py` handles Discord events and message routing.
- `main.py` is the main application entry.
- `background_fetch.py` runs background ingestion/update work.
- `rag/`, `agents/`, `agent_tools/`, `utils/`, `database/`, `config/` contain the retrieval stack, the agent/tool wrappers, and persistence.

The background ingestion loop exists because ASU pages drift. Without refresh, RAG quality decays quickly. The code includes scraping/extraction (Selenium + BeautifulSoup, plus content extraction helpers), preprocessing/summarization, and vector-store writes.

## Agents (pragmatic, tool-oriented)

The multi-agent design here is closer to modular tool routing than to fully autonomous agents. The “superior” agent orchestrates a request, and specialized agents focus on specific domains (news/media, shuttle status, Discord channel context, library status, sports, student jobs, clubs/events, and similar). Some requests are plain text Q&A, others are explicit function calls with structured outputs.

## Running it today (expect breakage)

There is Docker support (`Dockerfile`, `docker-compose.yml`) intended to run the bot, the background fetch service, and Qdrant. You will also need credentials under `config/` (see the sample JSON files referenced below). If you try to run this project today, expect to patch integrations: parts of the stack were built before the current “skills/MCP” ecosystem, and the Gemini SDK used at the time has since been deprecated.

Basic setup:

```bash
git clone https://github.com/ashworks1706/SparkyAI.git
cd SparkyAI
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Docker Compose:

```bash
docker compose up -d
```

## Project status and why it stopped

SparkyAI is not maintained. I stopped active work mainly because agentic systems are operationally expensive: multi-step tool calls, retrieval + reranking, retries, and background ingestion add real cost and latency. This repo also predates a lot of the modern baseline for production agent systems, so keeping it up-to-date would largely mean rebuilding core parts of it.

Even so, the project was worth doing. It taught me a lot about where RAG actually fails in practice (stale pages, messy HTML, conflicting sources, long-tail queries) and where “agents” become expensive in real deployments (routing overhead, tool failures, rate limits, and the complexity of keeping state).

Today I treat this repository as a marker of what my curiosity led me to build while trying to understand agents and advanced RAG latency management.

## Where the work continued

If you are interested in the next iterations of these ideas, see:

- `https://github.com/ashworks1706/kaelum` — exploring efficient, verifiable agent reasoning and routing.
- `https://github.com/ashworks1706/piramid` — efforts focused on RAG latency issues for larger-scale applications.

## Samples / configuration

- `config/appConfig.json` example: `__sample__appConfig.json`
- Firebase secret example: `__sample_firebase_secret.json`

## References

This project draws inspiration from work on vector similarity search, MIPS, approximate nearest neighbor indexing, and hierarchical retrieval:

1. Amagata, D., & Hara, T. (2023). Reverse Maximum Inner Product Search: Formulation, Algorithms, and Analysis.
2. Sun, P. (2020). Announcing ScaNN: Efficient Vector Similarity Search.
3. Guo, R., Sun, P., Lindgren, E., Geng, Q., Simcha, D., Chern, F., & Kumar, S. (2020). Accelerating Large-Scale Inference with Anisotropic Vector Quantization.
4. Dong, W., Moses, C., & Li, K. (2024). SOAR: Improved Indexing for Approximate Nearest Neighbor Search.
5. Kandpal, N., Jiang, H., Kong, X., Teng, J., & Chen, J. (2024). RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval.
6. Guo, R., Kumar, S., Choromanski, K., & Simcha, D. (2019). Quantization based Fast Inner Product Search.

## License

MIT (see `LICENSE`).
