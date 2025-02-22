

from utils.common_imports import *
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
