
# **ASU Discord Research Assistant Bot**

A powerful Discord bot designed to assist Arizona State University (ASU) students with streamlined access to resources, including the latest news, events, scholarships, courses, and more. Leveraging cutting-edge technologies, it uses AI-driven techniques to provide accurate, real-time responses to user queries while integrating directly with ASU's online ecosystem.

---

## **Key Features**


- **Web Scraping & Summarization:** Scrapes relevant ASU web pages, processes data, and provides concise, AI-generated summaries.
- **Discord Integration:** User-friendly bot interface to answer student queries on various ASU-related topics.
- **Optimized Search & Retrieval:** Implements advanced RAG (Retrieval-Augmented Generation) techniques for efficient document retrieval and processing.
- **Custom Workflows:** Provides precise responses in a strict format to ensure clarity and prevent abuse.
- **AI-Driven Information Retrieval:** Utilizes Retrieval-Augmented Generation (RAG) architecture to combine vector-based database search with LLM responses for accurate and context-aware answers.  
- **Contextual Summarization:** Employs Gemini and LangChain models for summarizing extensive ASU resources into concise and query-specific formats.  
- **Web Scraping & Document Ranking:** Integrates advanced scraping techniques with RAPTOR (Relevance-Aware Prioritized Topical Object Ranking) to enhance retrieval accuracy.  
- **Real-Time AI Interaction:** AI dynamically processes questions and retrieves results based on semantic understanding, leveraging cutting-edge natural language processing.  
- **Extensive Webhooks:** Dynamic access to the latest ASU clubs, events, news, scholarships, dining options, and shuttle timings.
- **Analytics Dashboard:** A comprehensive data visualization tool for tracking bot interactions and user activity.
- **Extensible Design:** Modular implementation to support future features like FaceID verification, action-based responses, and multimedia analysis.

---

## **Technologies Used**

- **Programming Languages:** Python, JavaScript
- **AI & ML Frameworks:** Gemini Vertex AI, LangChain, TensorFlow.  
- **Vector Search:** Combines RAPTOR and RAG architecture with vector databases for semantic search and ranking.  
- **Data Handling:** Implements advanced compression techniques for large document processing while preserving query-relevant content.  
- **Scalable Design:** Ensures seamless integration of AI-driven workflows using modular architectures for future scalability.  
- **Libraries & Frameworks:** TensorFlow, Firecrawl, Jina AI, Selenium
- **APIs & Platforms:** Google Search API, Discord API, ASU Webhooks
- **Database Systems:** Vector DB, Replit-hosted persistent storage
- **Hosting:** Replit for seamless deployment and accessibility

---

## **Project Workflow**

1. **Keyword Extraction:** AI models extract semantic meanings from user queries.  
2. **Document Retrieval:** Combines keyword-based search and vector embeddings using a vector database for enhanced contextual matching.  
3. **RAG Integration:**  
   - **Retrieval:** Finds and ranks documents using RAPTOR for multi-query synthesis.  
   - **Augmented Generation:** Merges retrieved content with LLM-generated responses for relevance and detail.  
4. **Content Summarization:** Summarizes large documents into precise, user-friendly answers using AI models like LangChain and TensorFlow-based LLMs.  
5. **Discord Response Delivery:** Presents structured results through the bot interface with options for follow-up or refinement.  
6. **Optimization & Monitoring:** Logs function calls, errors, and warnings while ensuring real-time updates.

---

## **Getting Started**

### **Installation**

1. **Clone Repository:**  
   ```bash
   git clone <repository-link>
   cd <repository-folder>
   ```

2. **Set Up Environment:**  
   Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. **Add API Keys:**  
   Update `config.json` with API keys for Google Search, Discord, and ASU webhooks.

4. **Run the Bot:**  
   Start the bot locally:
   ```bash
   python main.py
   ```

5. **Host on Replit:**  
   Follow Replit hosting guidelines for deployment.

---

## Roadmap

| Category | Task | Description | Status |
|----------|------|-------------|--------|
| Technologies | Setup technologies | Integrate Gemini, LangChain, Google Search API, and Discord API | ✅ Completed |
| Workflow | Create project workflow diagram | Map out question processing, scraping, and summarization steps | ✅ Completed |
| Web Scraping | Develop scraping algorithms | Identify keywords, select URLs, and preprocess responses | ✅ Completed |
| | Use advanced scraping tools | Implement Firecrawl, Jina AI, and Selenium for efficient scraping | ✅ Completed |
| Model Integration | Experiment with different models | Test and refine LLMs for efficiency and accuracy | ✅ Completed |
| | Optimize prompt templates | Develop strict, abuse-resistant prompt templates | ✅ Completed |
| Discord Bot | Setup and host bot | Convert the project into a Discord bot and host it on Replit | ✅ Completed |
| | Implement Discord parameters | Differentiate bot responses for jobs, scholarships, sports, dining, and more | ✅ Completed |
| | Add verification and memory features | ASURITE-based verification with FaceID integration | ✅ Completed |
| Optimizations | Implement generative AI for search queries | Use generative AI like Gemini Vertex for efficient searches | ✅ Completed |
| | Refine answer formats | Return concise responses in JSON format and improve document metadata handling | ✅ Completed |
| | Add keyword-based and filter search | Enable efficient keyword and metadata-based document search | ✅ Completed |
| Data Handling | Update and optimize database | Use proper metadata structures and fetch only useful links | ✅ Completed |
| | Compress documents for context | Use LLM to compress large documents into question-relevant summaries | ✅ Completed |
| | Implement RAPTOR ranking | Combine rankings from multiple queries for robust document retrieval | 🔄 In Progress |
| | Add data analytics board | Track and visualize metrics like response times, query steps, and user activity | 🔄 In Progress |
| Additional Features | Add ASURITE verification | Implement ASURITE login and dashboard access with FaceID integration | ✅ Completed |
| | Enhance multimedia capabilities | Enable image-to-text, text-to-image, and audio-to-text capabilities | 🔄 In Progress |
| | Add advanced response templates | Create reusable response templates for consistent answers | ✅ Completed |
| | Create voice-call counseling | Enable voice support for personalized responses | ❌ Pending |
| | Develop math helper | Integrate a module for solving math queries | ❌ Pending |
| Testing | Final testing and debugging | Test all features, refine functions, and fix bugs | 🔄 In Progress |

### Legend:
- ✅ Completed: Task is done
- 🔄 In Progress: Task is being worked on
- ❌ Pending: Task is yet to be started

### Phase 1: Core Features (Completed)
- Setup core classes, functions, and workflows
- Implement scraping, keyword identification, and AI summarization
- Deploy Discord bot with Replit hosting

### Phase 2: Enhancements (Completed)
- Integrate webhook functionalities (news, scholarships, sports, dining options)
- Implement strict prompt templates for LLMs
- Optimize document meta-data processing with RAPTOR and MIPS

### Phase 3: Advanced Features (In Progress)
- **AI Agents:** Add functions for announcements, events, polls, and Q&A sessions
- **Context Management:** Refine memory handling and clear user memory at intervals
- **Multimedia Capabilities:** Implement image-to-text and text-to-audio conversion
- **FaceID Verification:** Enable ASURITE-based access with optional facial recognition

### Phase 4: Analytics & Final Testing (In Progress)
- Create a data analytics dashboard for moderators
- Test extensively with different LLMs and refine workflows for production readiness

## Post-Funding Goals
- Implement comprehensive data analytics board on website
- Create specific agents for jobs, scholarships, library, forums, announcements, clubs, and moderator helper connections
- Finetune Gemini action model and search model
- Implement live sports matches webhook
- Integrate MemGPT for enhanced memory management
- Develop parking spot information feature
- Expand multimedia capabilities (image and audio generation/analysis)
- Conduct final testing and optimization

---

## **Contributing**

Contributions are welcome! Please open an issue or submit a pull request for any ideas or improvements. Make sure to follow the project’s coding guidelines and test your changes thoroughly.

---

## **License**

This project is licensed under the [MIT License](LICENSE).

---

## **Contact**

- **Author:** Ash (Som)  
- **Portfolio:** [somwrks.com](https://somwrks.com)  
- **LinkedIn:** [linkedin.com/somwrks](https://linkedin.com/somwrks)

---

### **To-Do Breakdown**
Refer to the **`Scratch Todos`** section in the repository for detailed and actionable development tasks.

---

Let me know if you'd like further tweaks or more additions to the README!
