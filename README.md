# SparkyAI: A Multi-Agent University Copilot

SparkyAI is a Discord-native university copilot designed for Arizona State University (ASU) students who need fast, context-aware access to courses, scholarships, events, jobs, campus services, and official updates. The project combines retrieval-augmented generation (RAG), specialized tool-using agents, and persistent conversational memory so that responses are not only fluent but also grounded in retrievable evidence and institutional context. The repository was made public after credential hardening and infrastructure cleanup from an older private development history.

![{2B61349D-750C-4418-A76E-15CB3AAB0B8B}](https://github.com/user-attachments/assets/642fd6d6-5232-4347-b1dc-3e78d3d0c758)

![{5B0799E0-D15C-4017-867A-F1DEB1FDA2DC}](https://github.com/user-attachments/assets/e19a175a-2c70-4af8-88d8-6303b9729cda)

![{2DDB8F4F-5F0E-4828-8FDD-847E67C40A65}](https://github.com/user-attachments/assets/7fbce508-e180-4f8f-9d7f-11feac5757e8)

## Abstract

Modern campus assistants must operate under three simultaneous constraints: broad domain coverage, high retrieval precision, and robust interaction continuity across users and channels. SparkyAI addresses this constraint triad through a hierarchical multi-agent architecture in which a superior routing model dispatches queries to specialized functional agents and retrieval pipelines. The system integrates Qdrant-based semantic indexing, RAPTOR-inspired hierarchical retrieval, cross-encoder reranking, and asynchronous service orchestration, yielding a practical framework for institution-scale question answering and task support. The objective is not merely to answer isolated questions, but to synthesize reliable, source-aware responses from dynamic campus data surfaces.

## System Overview

At runtime, SparkyAI behaves as an orchestrated sequence of inference and retrieval stages rather than a single monolithic chatbot call. User input enters through Discord commands, where contextual metadata such as user identity, server channel, and moderation constraints are captured. A superior agent then estimates the required action class and either answers directly or delegates control to one or more domain agents. These domain agents are configured to access focused resources, including ASU news streams, sports schedules, scholarship portals, student organizations, library status endpoints, and employment systems. Retrieved documents are normalized, filtered, and reranked before final response synthesis.

The architecture is designed as a retrieval-first decision process. Instead of treating generation as the primary source of truth, SparkyAI prioritizes evidence acquisition and then composes natural language around ranked context. This design reduces unsupported claims and improves user trust when compared with purely parametric generation.

## Mathematical Framing

The retrieval layer can be formalized as a maximum inner product objective over query and document embeddings. Given a user query embedding $q \in \mathbb{R}^d$ and candidate document embeddings $x_i \in \mathbb{R}^d$, initial relevance is approximated by

$$
i^* = \arg\max_i \langle q, x_i \rangle,
$$

where $i^*$ denotes the index of the candidate document with the highest relevance score under inner-product similarity. This corresponds to the MIPS criterion used in dense retrieval systems.

Because a single similarity pass is insufficient for multi-source campus data, SparkyAI composes a hybrid score from multiple retrieval channels. If $s_{\text{sim}}$, $s_{\text{raptor}}$, and $s_{\text{ann}}$ denote normalized scores from semantic similarity, hierarchical RAPTOR retrieval, and approximate nearest-neighbor pathways, then candidate ranking can be described as

$$
S(d \mid q) = \alpha\, s_{\text{sim}}(d,q) + \beta\, s_{\text{raptor}}(d,q) + \gamma\, s_{\text{ann}}(d,q), \quad \alpha+\beta+\gamma=1.
$$

In this formulation, $s_{\text{sim}}$ captures embedding-space semantic proximity, $s_{\text{raptor}}$ captures hierarchy-informed retrieval quality, and $s_{\text{ann}}$ captures approximate nearest-neighbor relevance; each score is normalized to a common $[0,1]$ range per query before weighted composition. A cross-encoder reranker then refines top-$k$ candidates with a relevance posterior,

$$
P(r=1\mid q,d) = \sigma\!\big(f_{\theta}(q,d)\big),
$$

where $r=1$ indicates that a query-document pair is relevant, $f_{\theta}$ is the cross-encoder scoring function, and $\sigma$ is the logistic transform. Final answer grounding is effectively a constrained conditional generation objective,

$$
\hat{y} = \arg\max_y\; P\big(y\mid q, D_k\big),
$$

with $D_k$ denoting the reranked supporting context set.

## Architectural Components

The platform is organized around a superior routing agent, specialist domain agents, retrieval/indexing services, and persistence middleware. The superior layer determines whether a request is informational, action-oriented, or hybrid. Specialist agents execute scoped tool calls for domains such as courses, scholarships, events, sports, media, and campus logistics. The retrieval subsystem stores and serves embeddings through Qdrant and supplements dense retrieval with hierarchical abstractions inspired by RAPTOR. A preprocessing pipeline performs extraction, cleaning, and summarization from both static and dynamic sources, including Selenium-assisted collection where JavaScript rendering is necessary.

Persistent data management is bifurcated by function. Firestore is used for chat histories and interaction-state continuity, while tabular moderation and oversight operations are handled through Google Sheets integration in the current architecture. This separation allows conversational memory and operational analytics to evolve independently without coupling every concern into a single backend.

## Agentic Reasoning and Control Flow

SparkyAI’s reasoning process can be viewed as hierarchical policy selection. Let $a \in \mathcal{A}$ denote an action policy where $\mathcal{A}$ includes direct response, retrieval search, or specialist delegation. The superior agent approximates

$$
a^* = \arg\max_{a \in \mathcal{A}} P(a \mid q, c),
$$

where $q$ is the user query and $c$ is interaction context (history, channel state, and metadata). Delegated agents then instantiate tool-level calls and return structured evidence for response composition. This reduces prompt overload, since each agent solves a narrower subproblem with domain-specific constraints.

## Data Processing and Retrieval Pipeline

Ingestion begins with source collection from institutional pages and related endpoints. Raw content is cleaned and segmented into retrieval units suitable for embedding and indexing. Embeddings are generated with `BAAI/bge-large-en-v1.5`, and top candidates are retrieved through similarity and ANN mechanisms. RAPTOR-style hierarchy construction supports coarse-to-fine retrieval when queries are broad or compositional. The system then applies reranking, deduplication, and source consolidation before generation. This ordering is important: by delaying generation until after ranking and merge operations, SparkyAI improves factual alignment and reduces contradictory context windows.

## Model Adaptation and Fine-Tuning Context

The repository includes a fine-tuning track for the superior decision layer using approximately 560 examples of interaction trajectories. The dataset distribution spans factual prompts, action-oriented requests that require function calls, hybrid prompts requiring both reasoning and tooling, and safety-critical adversarial edge-case inputs requiring robust refusal and control behavior. In practical terms, this tuning objective is less about linguistic style and more about improving policy selection quality under realistic student workflows.

## Technology Stack

SparkyAI is implemented primarily in Python and deploys on a containerized stack. Core model orchestration uses Gemini APIs with LangChain-adjacent retrieval patterns. NLP preprocessing and embedding workflows rely on Hugging Face ecosystem components and NLTK utilities. Vector retrieval is backed by Qdrant, while dynamic web extraction uses Selenium and BeautifulSoup. Runtime deployment is orchestrated with Docker and Docker Compose, and service-level persistence is provided by Firestore and auxiliary Google Sheets integration.

## Setup and Local Execution

To reproduce the environment, clone the repository and enter the project directory.

```bash
git clone https://github.com/ashworks1706/SparkyAI.git
cd SparkyAI
```

Create and activate a virtual environment before dependency installation.

```bash
python3 -m venv venv
source venv/bin/activate
```

On Windows, use the equivalent activation command:

```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies from `requirements.txt`.

```bash
pip install -r requirements.txt
```

Create `config/appConfig.json` following the sample schema in `__sample__appConfig.json`, then place Firebase credentials in `config/firebase_secret.json` using `__sample_firebase_secret.json` as structure guidance.

Containerized execution is the recommended path because Chrome/Chromedriver dependencies for scraping are already provisioned in the Docker workflow.

```bash
docker compose up -d
```

If your environment uses legacy Compose binaries, use:

```bash
docker-compose up -d
```

The main services include the Discord bot runtime, background fetch worker, and Qdrant. Logs can be streamed through:

```bash
docker compose logs -f app
docker compose logs -f background-fetch
```

Shutdown is performed with:

```bash
docker compose down
```

## Operational Notes and Troubleshooting

Most startup failures in local deployments arise from configuration incompleteness, missing secrets, or container networking constraints. If services fail to initialize, first verify that Docker is healthy and all declared containers are up. Next validate that `config/appConfig.json` and `config/firebase_secret.json` are present and correctly formatted. If retrieval endpoints are unavailable, confirm that Qdrant ports `6333` and `6334` are free and reachable within the container network. On Linux hosts, volume permission mismatches can block startup; when this occurs, normalize ownership for `config/`, `logs/`, and `qdrant_storage/` to the active user.

## Research Lineage and Citations

SparkyAI’s retrieval design draws from literature on inner-product search, approximate nearest-neighbor indexing, and hierarchical retrieval. Relevant references include reverse MIPS formulations, ScaNN and anisotropic quantization for accelerated high-dimensional search, and RAPTOR for tree-organized abstraction in retrieval workflows. The current project integrates these ideas into a practical campus-assistant implementation where latency, evidence quality, and domain coverage must be jointly optimized.

1. Amagata, D., & Hara, T. (2023). Reverse Maximum Inner Product Search: Formulation, Algorithms, and Analysis. *ACM Transactions on the Web, 17*(4), 1-23. https://dl.acm.org/doi/pdf/10.1145/3587215
2. Sun, P. (2020). Announcing ScaNN: Efficient Vector Similarity Search. Google Research Blog. https://research.google/blog/announcing-scann-efficient-vector-similarity-search/
3. Guo, R., Sun, P., Lindgren, E., Geng, Q., Simcha, D., Chern, F., & Kumar, S. (2020). Accelerating Large-Scale Inference with Anisotropic Vector Quantization. *ICML*. https://arxiv.org/pdf/1908.10396
4. Dong, W., Moses, C., & Li, K. (2024). SOAR: Improved Indexing for Approximate Nearest Neighbor Search. arXiv preprint. https://www.arxiv.org/pdf/2411.06158
5. Kandpal, N., Jiang, H., Kong, X., Teng, J., & Chen, J. (2024). RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval. arXiv preprint. https://arxiv.org/pdf/2401.18059v1
6. Guo, R., Kumar, S., Choromanski, K., & Simcha, D. (2019). Quantization based Fast Inner Product Search. arXiv preprint. https://arxiv.org/pdf/1509.01469

## Contribution and Contact

Contributions are welcomed through issues and pull requests, particularly for improvements in retrieval quality, reliability engineering, and agent safety controls. For project context and collaboration inquiries, reach out to Ash at [ashworks.dev](https://ashworks.dev) or via [linkedin.com/in/ashworks](https://www.linkedin.com/in/ashworks).

## License

This project is distributed under the [MIT License](LICENSE).
