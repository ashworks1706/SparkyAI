from utils.common_imports import *
import json
import numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import CrossEncoder

class RaptorRetriever:
    def __init__(self, vector_store_class, logger, vector_store,num_levels=3, branching_factor=5):
        try:
            self.logger = logger
            self.vector_store_class = vector_store_class
            self.num_levels = num_levels
            self.vector_store=vector_store
            self.branching_factor = branching_factor
            self.tree = self.build_raptor_tree()
            logger.info(f"@raptor.py RAPTOR Retriever initialized.")
        except Exception as e:
            logger.error(f"@raptor.py Error initializing RAPTOR Retriever: {str(e)}")
            raise e

    def build_raptor_tree(self):
        tree = {}
        self.logger.info(f"@raptor.py Building RAPTOR tree...")
        all_docs = self.vector_store_class.get_all_documents()
        self.logger.info(f"@raptor.py Retrieved {len(all_docs)} documents for tree construction")
        all_embeddings = self.vector_store_class.get_embeddings(all_docs)

        if not all_embeddings:
            self.logger.warning(f"@raptor.py No embeddings found. The vector store may be empty.")
            return tree

        all_embeddings = np.array(all_embeddings)
        try:
            for level in range(self.num_levels):
                n_clusters = min(self.branching_factor ** (level + 1), len(all_embeddings))
                kmeans = KMeans(n_clusters=n_clusters)
                cluster_labels = kmeans.fit_predict(all_embeddings)

                clusters = {}
                for i, label in enumerate(cluster_labels):
                    if label not in clusters:
                        clusters[label] = []
                    clusters[label].append(all_docs[i])
                    
                    # Assign cluster IDs to documents if we're at level 0
                    if level == 0:
                        # Update the document with its cluster ID in the vector store
                        try:
                            doc_id = getattr(all_docs[i], 'id', None) or getattr(all_docs[i], 'metadata', {}).get('id')
                            if doc_id:
                                # Convert NumPy int32/64 to regular Python int if necessary
                                cluster_label = int(label)
                                self.vector_store.update_document_metadata(
                                    doc_id, 
                                    {"cluster_id": cluster_label}
                                )
                                self.logger.debug(f"@raptor.py Assigned cluster ID {cluster_label} to document {doc_id}")
                        except Exception as e:
                            self.logger.error(f"@raptor.py Failed to update document cluster ID: {str(e)}")

                summaries = {label: self.generate_summary(docs) for label, docs in clusters.items()}
                self.logger.info(f"@raptor.py Generated summaries for level {level + 1}")
                tree[f"level_{level}"] = {
                    "clusters": clusters,
                    "summaries": summaries
                }
                self.logger.info(f"@raptor.py Built tree for level {level + 1}")

                all_docs = list(summaries.values())
                self.logger.info(f"@raptor.py Retrieved {len(all_docs)} documents for next level")
                all_embeddings = self.vector_store_class.get_embeddings(all_docs)
                self.logger.info(f"@raptor.py Retrieved embeddings for next level")

                if not all_embeddings:
                    self.logger.warning(f"@raptor.py No embeddings found for level {level + 1}. Stopping tree construction.")
                    break

                all_embeddings = np.array(all_embeddings)
        except Exception as e:
            self.logger.error(f"@raptor.py Error building RAPTOR tree: {str(e)}")
            raise e

        self.logger.info(f"@raptor.py Building RAPTOR tree completed.")
        self.logger.info(f"@raptor.py RAPTOR Tree Structure:\n{self.format_tree_structure(tree)}")
        return tree

    def format_tree_structure(self, tree):
        def format_level(level, level_data):
            formatted = f"Level {level}:\n"
            for cluster_id, cluster_data in level_data["clusters"].items():
                formatted += f"  Cluster {cluster_id}: {len(cluster_data)} documents\n"
            return formatted

        formatted_tree = ""
        for level, level_data in tree.items():
            formatted_tree += format_level(level, level_data)
        return formatted_tree

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
            self.logger.info(f"@raptor.py Generated summary for document.")
        return " ".join(summaries)[:200]

    def retrieve(self, query, top_k=5):
        try:
            self.logger.info(f"@raptor.py Retrieving documents for query: {query[:50]}...")
            
            try:
                query_embedding = self.vector_store_class.embedding_model.embed_query(query)
                self.logger.info(f"@raptor.py Generated query embedding successfully")
            except Exception as e:
                self.logger.error(f"@raptor.py Error generating query embedding: {str(e)}")
                return []
                
            current_level = self.num_levels - 1
            self.logger.info(f"@raptor.py Starting retrieval from level {current_level}")
            
            try:
                current_node = self.tree[f"level_{current_level}"]
                self.logger.info(f"@raptor.py Accessed level {current_level} with {len(current_node.get('summaries', {}))} summaries")
            except KeyError:
                self.logger.error(f"@raptor.py Level {current_level} not found in tree")
                return []
                
            while current_level >= 0:
                self.logger.debug(f"@raptor.py Processing at level {current_level}")
                try:
                    summaries = current_node["summaries"]
                    self.logger.debug(f"@raptor.py Found {len(summaries)} summaries at level {current_level}")
                    
                    try:
                        summary_embeddings = self.vector_store_class.get_embeddings(list(summaries.values()))
                        self.logger.debug(f"@raptor.py Generated {len(summary_embeddings)} summary embeddings")
                    except Exception as e:
                        self.logger.error(f"@raptor.py Error generating summary embeddings at level {current_level}: {str(e)}")
                        return []
                    
                    try:
                        best_cluster = max(summaries.keys(), key=lambda x: np.dot(query_embedding, summary_embeddings[list(summaries.keys()).index(x)]))
                        self.logger.info(f"@raptor.py Selected cluster {best_cluster} at level {current_level}")
                    except (ValueError, IndexError) as e:
                        self.logger.error(f"@raptor.py Error finding best cluster: {str(e)}")
                        return []

                    if current_level == 0:
                        self.logger.info(f"@raptor.py Reached leaf level, retrieving documents from cluster {best_cluster}")
                        try:
                            # Convert NumPy int32 to standard Python int
                            try:
                                cluster_id = int(best_cluster)
                                self.logger.debug(f"@raptor.py Converted cluster_id {best_cluster} to {cluster_id}")
                            except (TypeError, ValueError) as e:
                                self.logger.error(f"@raptor.py Failed to convert cluster_id: {str(e)}")
                                return []
                            
                            try:
                                # Use the correct filter parameter format that Qdrant expects
                                self.logger.debug(f"@raptor.py Executing similarity search for cluster {cluster_id}")
                                filter_dict = {"must": [{"key": "cluster_id", "match": {"value": cluster_id}}]}
                                initial_results = self.vector_store.similarity_search(
                                    query, 
                                    k=top_k,
                                    filter=filter_dict
                                )
                                self.logger.info(f"@raptor.py Retrieved {len(initial_results)} initial results from cluster {cluster_id}")
                            except Exception as e:
                                self.logger.error(f"@raptor.py Error during similarity search: {str(e)}")
                                self.logger.warning(f"@raptor.py Falling back to default search without cluster filter")
                                try:
                                    initial_results = self.vector_store.similarity_search(query, k=top_k)
                                    self.logger.info(f"@raptor.py Fallback retrieved {len(initial_results)} results")
                                except Exception as e2:
                                    self.logger.error(f"@raptor.py Fallback search also failed: {str(e2)}")
                                    return []
                            
                            if not initial_results:
                                self.logger.warning(f"@raptor.py No results found in cluster {cluster_id}")
                                return []
                                
                            try:
                                reranked_results = self.rerank_results(query, initial_results, top_k)
                                self.logger.info(f"@raptor.py Returning {len(reranked_results)} reranked results")
                                return reranked_results
                            except Exception as e:
                                self.logger.error(f"@raptor.py Error during reranking: {str(e)}")
                                self.logger.warning(f"@raptor.py Falling back to initial results")
                                return initial_results[:top_k]
                        except Exception as e:
                            self.logger.error(f"@raptor.py Error during final retrieval: {str(e)}")
                            return []

                    current_level -= 1
                    self.logger.info(f"@raptor.py Moving down to level {current_level}")
                    
                    try:
                        current_node = self.tree[f"level_{current_level}"]
                        self.logger.debug(f"@raptor.py Accessed next level with {len(current_node.get('summaries', {}))} summaries")
                    except KeyError as e:
                        self.logger.error(f"@raptor.py Error accessing level {current_level}: {str(e)}")
                        return []
                        
                except KeyError as e:
                    self.logger.error(f"@raptor.py Missing key in tree structure at level {current_level}: {str(e)}")
                    return []
                except Exception as e:
                    self.logger.error(f"@raptor.py Unexpected error processing level {current_level}: {str(e)}")
                    return []

            self.logger.warning(f"@raptor.py No results found in RAPTOR tree after traversal")
            return []
            
        except Exception as e:
            self.logger.error(f"@raptor.py Critical error in retrieve method: {str(e)}")
            return []

    def rerank_results(self, query, initial_results, top_k=5):
        try:
            self.logger.info(f"@raptor.py Starting reranking with {len(initial_results)} documents")
            
            try:
                cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
                self.logger.info(f"@raptor.py Cross-encoder model loaded successfully")
            except Exception as e:
                self.logger.error(f"@raptor.py Error loading cross-encoder model: {str(e)}")
                return initial_results[:top_k]  # Fallback to initial results
            
            try:
                pairs = []
                for doc in initial_results:
                    try:
                        content = self.get_document_content(doc)
                        pairs.append([query, content])
                        self.logger.debug(f"@raptor.py Created pair with content length: {len(content)}")
                    except Exception as e:
                        self.logger.error(f"@raptor.py Error extracting content from document: {str(e)}")
                        pairs.append([query, ""])  # Add empty content as fallback
                
                self.logger.info(f"@raptor.py Created {len(pairs)} query-document pairs for reranking")
            except Exception as e:
                self.logger.error(f"@raptor.py Error creating document pairs: {str(e)}")
                return initial_results[:top_k]  # Fallback to initial results
                
            try:
                scores = cross_encoder.predict(pairs)
                self.logger.info(f"@raptor.py Generated {len(scores)} scores with cross-encoder")
                self.logger.debug(f"@raptor.py Score range: min={min(scores) if len(scores) > 0 else 'N/A'}, max={max(scores) if len(scores) > 0 else 'N/A'}")
            except Exception as e:
                self.logger.error(f"@raptor.py Error predicting scores with cross-encoder: {str(e)}")
                return initial_results[:top_k]  # Fallback to initial results
                
            try:
                scored_results = list(zip(scores, initial_results))
                sorted_results = sorted(scored_results, key=lambda x: x[0], reverse=True)
                reranked_results = [doc for _, doc in sorted_results]
                self.logger.info(f"@raptor.py Successfully reranked results")
                
                if len(reranked_results) > top_k:
                    self.logger.info(f"@raptor.py Returning top {top_k} of {len(reranked_results)} reranked results")
                else:
                    self.logger.info(f"@raptor.py Returning all {len(reranked_results)} reranked results")
                    
                return reranked_results[:top_k]
            except Exception as e:
                self.logger.error(f"@raptor.py Error sorting and slicing results: {str(e)}")
                return initial_results[:top_k]  # Fallback to initial results
                
        except Exception as e:
            self.logger.error(f"@raptor.py Unexpected error in reranking: {str(e)}")
            return initial_results[:top_k]  # Fallback to initial results

    def get_document_content(self, doc):
        if isinstance(doc, str):
            try:
                doc_dict = json.loads(doc)
                self.logger.info(f"@raptor.py Retrieved document content.")
                return doc_dict.get('page_content', '')
            except json.JSONDecodeError:
                return doc
        else:
            return getattr(doc, 'page_content', str(doc))
        
    def update_raptor_tree(self, new_documents :list):
        """
        Update the RAPTOR tree with new documents without rebuilding the entire tree.
        
        Args:
            new_documents: List of new documents to add to the tree
        """
        try:
            self.logger.info(f"@raptor.py Updating RAPTOR tree with {len(new_documents)} new documents")
            
            if not new_documents:
                self.logger.info(f"@raptor.py No new documents to add.")
                return
            
            # Get embeddings for new documents
            try:
                new_embeddings = self.vector_store_class.get_embeddings(new_documents)
                self.logger.info(f"@raptor.py Generated embeddings for {len(new_embeddings)} new documents")
            except Exception as e:
                self.logger.error(f"@raptor.py Error generating embeddings: {str(e)}")
                return
            
            if not new_embeddings:
                self.logger.warning(f"@raptor.py Could not generate embeddings for new documents.")
                return
            
            new_embeddings = np.array(new_embeddings)
            
            # Start with the lowest level (level_0)
            try:
                level_0 = self.tree.get("level_0", {})
                clusters = level_0.get("clusters", {})
                self.logger.debug(f"@raptor.py Retrieved level 0 with {len(clusters)} clusters")
            except Exception as e:
                self.logger.error(f"@raptor.py Error accessing tree structure: {str(e)}")
                return
            
            # For each new document, find the closest cluster and add it
            for i, embedding in enumerate(new_embeddings):
                try:
                    # If this is the first update, we need to initialize the tree
                    if not self.tree:
                        self.logger.info(f"@raptor.py Tree is empty. Building a new tree instead of updating.")
                        self.tree = self.build_raptor_tree()
                        return
                    
                    if not clusters:
                        self.logger.info(f"@raptor.py No clusters found in level 0. Building a new tree.")
                        self.tree = self.build_raptor_tree()
                        return
                    
                    # Find the closest cluster for the new document
                    closest_cluster = None
                    max_similarity = -float('inf')
                    
                    for cluster_id, cluster_docs in clusters.items():
                        try:
                            cluster_embedding = self.vector_store_class.get_embeddings([level_0["summaries"][cluster_id]])[0]
                            similarity = np.dot(embedding, cluster_embedding)
                            
                            self.logger.debug(f"@raptor.py Similarity with cluster {cluster_id}: {similarity}")
                            
                            if similarity > max_similarity:
                                max_similarity = similarity
                                closest_cluster = cluster_id
                        except Exception as e:
                            self.logger.error(f"@raptor.py Error calculating similarity for cluster {cluster_id}: {str(e)}")
                            continue
                    
                    # Add the document to the closest cluster
                    if closest_cluster is not None:
                        clusters[closest_cluster].append(new_documents[i])
                        
                        # Update the document's cluster_id in the vector store
                        try:
                            doc_id = getattr(new_documents[i], 'id', None) or getattr(new_documents[i], 'metadata', {}).get('id')
                            if doc_id:
                                # Convert NumPy int32/64 to regular Python int if necessary
                                cluster_label = int(closest_cluster)
                                self.vector_store.update_document_metadata(
                                    doc_id, 
                                    {"cluster_id": cluster_label}
                                )
                                self.logger.debug(f"@raptor.py Updated cluster ID {cluster_label} for document {doc_id} in vector store")
                        except Exception as e:
                            self.logger.error(f"@raptor.py Failed to update document cluster ID in vector store: {str(e)}")
                        
                        # Update the summary for this cluster
                        try:
                            level_0["summaries"][closest_cluster] = self.generate_summary(clusters[closest_cluster])
                            self.logger.info(f"@raptor.py Added document to cluster {closest_cluster} in level 0 and updated summary")
                        except Exception as e:
                            self.logger.error(f"@raptor.py Error updating summary for cluster {closest_cluster}: {str(e)}")
                    else:
                        self.logger.warning(f"@raptor.py Could not find a suitable cluster for document {i}")
                except Exception as e:
                    self.logger.error(f"@raptor.py Error processing document {i}: {str(e)}")
                    continue
            
            # Propagate changes up the tree
            try:
                for level in range(1, self.num_levels):
                    level_key = f"level_{level}"
                    if level_key not in self.tree:
                        self.logger.debug(f"@raptor.py No more levels to update after level {level-1}")
                        break
                        
                    prev_level_key = f"level_{level-1}"
                    self.logger.info(f"@raptor.py Propagating changes to level {level}")
                    
                    # Update summaries in the current level
                    for cluster_id in self.tree[level_key]["clusters"]:
                        try:
                            child_docs = []
                            for child_cluster in self.tree[level_key]["clusters"][cluster_id]:
                                child_docs.append(self.tree[prev_level_key]["summaries"][child_cluster])
                            
                            self.tree[level_key]["summaries"][cluster_id] = self.generate_summary(child_docs)
                            self.logger.info(f"@raptor.py Updated summary for cluster {cluster_id} in level {level}")
                        except Exception as e:
                            self.logger.error(f"@raptor.py Error updating cluster {cluster_id} at level {level}: {str(e)}")
            except Exception as e:
                self.logger.error(f"@raptor.py Error propagating changes up the tree: {str(e)}")
            
            self.logger.info(f"@raptor.py RAPTOR tree update completed.")
            self.logger.info(f"@raptor.py Updated Tree Structure:\n{self.format_tree_structure(self.tree)}")
        
        except Exception as e:
            self.logger.error(f"@raptor.py Error updating RAPTOR tree: {str(e)}")
            raise e
