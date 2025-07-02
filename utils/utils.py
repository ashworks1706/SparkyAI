from utils.common_imports import *
from rag.raptor import RaptorRetriever
class Utils:
    def __init__(self,vector_store_class,asu_data_processor,asu_scraper,logger,group_chat):
        """Initialize the Utils from utils.common_imports import *
classwith task tracking and logging."""
        try:
            self.logger=logger
            self.tasks = []
            self.asu_data_processor = asu_data_processor
            self.asu_scraper = asu_scraper
            self.current_content = "Understanding your query..."
            self.message = None
            self.cached_doc_ids = []
            self.ground_sources =[]
            self.cached_queries=[]
            self.vector_store_class = vector_store_class
            self.vector_store = self.vector_store_class.get_vector_store()
            self.raptor_retriever = RaptorRetriever(vector_store_class=self.vector_store_class,logger=self.logger, vector_store=self.vector_store)
            self.group_chat=group_chat
            self.logger.info(f"@utils.py Group Chat setup successfully {group_chat}")

            self.logger.info(f"@utils.py \nUtils instance initialized successfully")
        except Exception as e:
            self.logger.error(f"@utils.py Failed to initialize Utils: {e}")
            pass

    async def start_animation(self, message):
        """Start the loading animation using Discord's built-in thinking indicator"""
        try:
            self.message = message
            self.logger.info(f"@utils.py Animation started for message: {message.id}")
        except:
            self.logger.error(f"@utils.py Failed to start animation:")
            pass
        

    async def update_text(self, new_content):
        """Update text while maintaining task history"""
        try:
            # Append previous content to tasks
            if self.current_content:
                self.tasks.append(self.current_content)
                self.logger.debug(f"Added task to history: {self.current_content}")

            # Update current content
            self.current_content = new_content
            self.logger.debug(f"Updated current content to: {new_content}")

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
            self.logger.info(f"@utils.py Message updated with {len(display_lines)} tasks")

        except:
            self.logger.error(f"@utils.py Failed to update text")
            pass
            # Optionally, you could re-raise the exception or handle it differently

    async def stop_animation(self, message=None, final_content=None,View=None):
        """Stop animation and display final content"""
        try:
            # Edit message with final content if provided
            if message and final_content:
                if View:
                    await message.edit(content=final_content,view=View)
                await message.edit(content=final_content)
                self.logger.info(f"@utils.py Final content set: {final_content}")

            # Reset internal state
            self.tasks = []
            self.current_content = ""
            self.message = None
            self.logger.info(f"@utils.py \nAnimation stopped and state reset")

        except Exception as e:
            self.logger.error(f"@utils.py Error stopping animation: {e}")

    def format_search_results(self, engine_context):
        """Format search results into a readable string."""
        if not engine_context:
            return "No search results found."
        try:
            self.logger.info(f"@utils.py \nFormatting search results")
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
            self.logger.error(f"@utils.py Error formatting search results: {str(e)}")
            return "Error formatting search results."
    
    async def perform_raptor_tree_update(self):
        try:
            await self.raptor_retriever.update_raptor_tree()
        except Exception as e:  
            self.logger.error(f"@utils.py Error updating RAPTOR tree: {str(e)}")
            raise ValueError("Failed to update RAPTOR tree")
        self.logger.info(f"@utils.py \nRAPTOR tree updated successfully")
        
    async def perform_web_search(self,search_url:str =None,  optional_query : str = None, doc_title : str =None, doc_category : str = None):
        try:
            # Initial search
            self.logger.info(f"@utils.py \nPerforming Web Search")

            try:
                documents = await self.asu_scraper.engine_search(search_url, optional_query)
            except Exception as e:
                self.logger.error(f"@utils.py 145 Error during web search: {str(e)}")
                raise ValueError("Failed to perform web search")
            
            if type(documents) == str:
                self.logger.info(f"@utils.py 149 No documents found matching the query")
                return documents
            
            if not documents:
                raise ValueError("@utils.py 153 No documents found matching the query")
            
            self.logger.info(documents) 
            self.logger.info(f"@utils.py \nPreprocessing documents...")
            
            try: 
                
                processed_docs = await self.asu_data_processor.process_documents(
                    documents=documents, 
                    search_context= self.group_chat.get_text(),
                    title = doc_title, category = doc_category
                )
            except Exception as e:
                self.logger.error(f"@utils.py 161 Error during document processing: {str(e)}")
                raise ValueError("Failed to process documents")
            
            self.logger.info(f"@utils.py \nDocuments processed successfully")
            
            self.logger.info(f"@utils.py Processed {len(processed_docs)} documents")
            
            try:
                # Update RAPTOR tree with new documents
                self.logger.info(f"@utils.py \nUpdating RAPTOR tree with new documents")
                self.raptor_retriever.queue_raptor_tree(processed_docs)
            except Exception as e:  
                self.logger.error(f"@utils.py 171 Error updating RAPTOR tree: {str(e)}")
                raise ValueError("Failed to update RAPTOR tree")
            
            self.logger.info(f"@utils.py \nStoring documents in vector store")
            try:
                store = self.vector_store_class.queue_documents(processed_docs)
            except Exception as e:
                self.logger.error(f"@utils.py 180 Error storing documents in vector store: {str(e)}")
                raise ValueError("Failed to store documents in vector store")
            
            
            self.logger.info(f"@utils.py \nDocuments stored successfully")
            
            results = []
            extracted_urls = []
            
            try:
                for doc in processed_docs:
                    doc_info = {
                        'content': doc.page_content,
                        'metadata': doc.metadata,
                        'timestamp': doc.metadata.get('timestamp'),
                        'url': doc.metadata.get('url')
                    }
                    results.append(doc_info)
                    
                    # Safely extract URLs
                    sources = doc.metadata.get('url', [])
                    if sources:
                        # Handle different URL formats (string, list, nested list)
                        if isinstance(sources, str):
                            extracted_urls.append(sources)
                        elif isinstance(sources, list):
                            # Handle nested lists by flattening them
                            for item in sources:
                                if isinstance(item, list):
                                    extracted_urls.extend([url for url in item if isinstance(url, str)])
                                elif isinstance(item, str):
                                    extracted_urls.append(item)
            except Exception as e:
                self.logger.error(f"@utils.py Error 213 processing individual document: {str(e)}")
                self.logger.error(f"@utils.py Documents : {processed_docs}")
                # Continue with next document   
                
            self.update_ground_sources(extracted_urls)

            results = self.format_search_results(results)

            return results

        except Exception as e:
            self.logger.error(f"@utils.py Error 224 in web search: {str(e)}")
            return "No results found on web"
       
    
    async def perform_similarity_search(self, query: str, categories: list):
        try:
            self.logger.info(f"@utils.py Performing similarity search with query: {query}")
            if not self.vector_store:
                self.logger.info(f"@utils.py \nVector Store not initialized")
                raise ValueError("Vector store not properly initialized")

            # Correct filter construction using Qdrant's Filter class
            filter_conditions = None
            try:
                if categories and len(categories) > 0:
                    filter_conditions = Filter(
                        must=[
                            FieldCondition(
                                key="metadata.category", 
                                match=MatchAny(any=categories)
                            )
                        ]
                    )
                    self.logger.info(f"@utils.py Filter conditions created for categories: {categories}")
            except Exception as e:
                self.logger.error(f"@utils.py Error creating filter conditions: {str(e)}")
                # Continue without filters if there's an error

            # Perform similarity search with optional filtering
            try:
                # Get more results than needed to allow for embeddings-based filtering
                raw_results = self.vector_store.similarity_search_with_score(
                    query, 
                    k=20,  # Fetch more results to allow filtering
                    filter=filter_conditions
                )
                
                # Extract documents and their scores
                scored_results = [(doc, score) for doc, score in raw_results]
                
                # Embeddings-based filtering: Keep only documents with a score above a threshold
                threshold = 0.75  # Adjust this threshold based on your use case
                filtered_results = [doc for doc, score in scored_results if score >= threshold]
                
                self.logger.info(f"@utils.py Filtered results count after embeddings-based filtering: {len(filtered_results)}")
                
                # Sort filtered results by timestamp (newer first)
                filtered_results.sort(key=lambda x: x.metadata.get('timestamp', ''), reverse=True)
                
                # Limit final results to top 10 documents
                results = filtered_results[:10]

                self.logger.info(f"@utils.py Similarity search completed, found {len(results)} results prioritized by relevance and recency")
            except Exception as e:
                self.logger.error(f"@utils.py Error during vector store similarity search: {str(e)}")
                return None

            # Check if results are empty
            if not results:
                self.logger.info(f"@utils.py \nNo documents found in vector store")
                return "No documents found in database"
            
            try:
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

                self.logger.info(f"@utils.py Retrieved and processed {len(documents)} documents from vector store")
                return documents
            except Exception as e:
                self.logger.error(f"@utils.py Error processing similarity search results: {str(e)}")
                return None

        except Exception as e:
            self.logger.error(f"@utils.py Error during similarity search: {str(e)}")
            return None


    def merge_search_results(self, raptor_results, similarity_results):
        """Merge and deduplicate search results from both retrieval methods"""
        try:
            self.logger.info(f"@utils.py Merging {len(raptor_results) if raptor_results else 0} RAPTOR results and {len(similarity_results) if similarity_results else 0} similarity results")
            
            # Handle cases where either result list might be None
            raptor_results = raptor_results or []
            similarity_results = similarity_results or []
            
            # Convert LangChain Document objects to dictionaries to ensure consistent handling
            combined_results = []
            try:
                for item in raptor_results:
                    if hasattr(item, 'page_content') and hasattr(item, 'metadata'):
                        # Convert Document object to dict
                        combined_results.append({
                            'content': item.page_content,
                            'metadata': item.metadata,
                            'score': getattr(item, 'score', 0)
                        })
                    else:
                        combined_results.append(item)
                
                combined_results.extend(similarity_results)
                self.logger.info(f"@utils.py Combined results count: {len(combined_results)}")
            except Exception as e:
                self.logger.error(f"@utils.py Error combining results: {str(e)}")
                combined_results = []
                # Try to salvage results if possible
                if raptor_results:
                    combined_results = [
                        {'content': doc.page_content, 'metadata': doc.metadata} 
                        if hasattr(doc, 'page_content') else doc 
                        for doc in raptor_results
                    ]
                elif similarity_results:
                    combined_results = similarity_results
            
            # Deduplicate by content similarity using a simple approach
            try:
                deduplicated_results = []
                seen_urls = set()
                content_hashes = set()
                
                self.logger.info(f"@utils.py Deduplicating results")
                
                for result in combined_results:
                    try:
                        # Deduplicate by URL first
                        url = None
                        metadata = result.get('metadata', {}) if isinstance(result, dict) else getattr(result, 'metadata', {})
                        
                        if isinstance(metadata, dict):
                            url = metadata.get('url')
                        
                        # Create a hashable key from URL
                        url_key = None
                        if url:
                            if isinstance(url, list):
                                flat_urls = []
                                for item in url:
                                    if isinstance(item, list):
                                        flat_urls.extend(str(u) for u in item)
                                    else:
                                        flat_urls.append(str(item))
                                url_key = tuple(flat_urls)
                            elif isinstance(url, str):
                                url_key = url
                        
                        # Use document ID as fallback if URL is not available
                        if not url_key and isinstance(metadata, dict):
                            doc_id = metadata.get('id')
                            if doc_id:
                                url_key = f"id:{doc_id}"
                        
                        # Content-based deduplication
                        content = result.get('content', '') if isinstance(result, dict) else getattr(result, 'page_content', '')
                        # Create a simple hash of the content (first 100 chars)
                        content_hash = hash(content[:100].lower().strip())
                        
                        # Add document if neither URL nor content was seen before
                        if (url_key and url_key not in seen_urls) or (content_hash not in content_hashes):
                            if url_key:
                                seen_urls.add(url_key)
                            content_hashes.add(content_hash)
                            deduplicated_results.append(result)
                        else:
                            self.logger.debug(f"@utils.py Skipping redundant result")
                    except Exception as e:
                        self.logger.error(f"@utils.py Error processing individual result: {str(e)}")
                        # Continue with next result
                
                self.logger.info(f"@utils.py Deduplicated results count: {len(deduplicated_results)}")
            except Exception as e:
                self.logger.error(f"@utils.py Error during deduplication: {str(e)}")
                deduplicated_results = combined_results[:10]  # Fallback to first 10 combined results
            
            try:
                # Sort by relevance score, defaulting to 0 if score is missing
                sorted_results = sorted(
                    deduplicated_results, 
                    key=lambda x: (
                        x.get('score', 0) if isinstance(x, dict) else 
                        getattr(x, 'score', 0)
                    ),
                    reverse=True
                )
                self.logger.info(f"@utils.py Successfully sorted results by relevance score")
            except Exception as e:
                self.logger.error(f"@utils.py Error sorting results: {str(e)}")
                sorted_results = deduplicated_results  # Fallback to unsorted results
            
            # Reorder results to improve retrieval effectiveness (similar to LongContextReorder)
            # Mix high and low ranked documents to avoid performance degradation
            if len(sorted_results) > 5:
                reordered_results = []
                half = len(sorted_results) // 2
                for i in range(half):
                    reordered_results.append(sorted_results[i])  # High ranked
                    if i + half < len(sorted_results):
                        reordered_results.append(sorted_results[i + half])  # Low ranked
                
                # Add any remaining documents
                for i in range(2 * half, len(sorted_results)):
                    reordered_results.append(sorted_results[i])
                    
                sorted_results = reordered_results
            
            final_results = sorted_results[:10]  # Return top 10 unique results
            self.logger.info(f"@utils.py Returning {len(final_results)} final merged results")
            return final_results
            
        except Exception as e:
            self.logger.error(f"@utils.py Error in merge_search_results: {str(e)}")
            # Return whatever might be available as a fallback
            if raptor_results:
                return raptor_results[:10]
            elif similarity_results:
                return similarity_results[:10]
            return []

    async def perform_database_search(self, query: str, categories: list):
        self.logger.info(f"@utils.py \nPerforming database search with query: {query}")
        self.cached_queries.append(query)
        
        # check whether there are documents present on database or not else skip and return no documents in database
        try:
            if self.vector_store_class.get_document_count() == 0:
                self.logger.info(f"@utils.py No documents found in database")
                return "No documents found in database"
        except Exception as e:
            self.logger.error(f"@utils.py Error checking document count in database {e}")
            return "Error checking document count in database"
        
        self.logger.info(f"@utils.py Performing similarity search")    
        
        try:
            # Perform similarity search
            similarity_results = await self.perform_similarity_search(query, categories)
        except:
            self.logger.error(f"@utils.py Error during similarity search")
            similarity_results = []

        self.logger.info(f"@utils.py Similarity search returned {len(similarity_results)} results")
        
        if type(similarity_results) == str:
            self.logger.info(f"@utils.py \nNo documents found in database")
            return similarity_results
        
        # Perform RAPTOR search
        self.logger.info(f"@utils.py \nPerforming RAPTOR search")
        try:
            raptor_results = self.raptor_retriever.retrieve(query, top_k=5)
        except Exception as e:
            self.logger.error(f"@utils.py Error during RAPTOR search {e}")
            raptor_results = []
        
        self.logger.info(f"@utils.py RAPTOR search returned {len(raptor_results)} results")

       
        
        
        try:
            # Combine and deduplicate results
            self.logger.info(f"@utils.py Combining and deduplicating search results")
            combined_results = self.merge_search_results(raptor_results, similarity_results)
            self.logger.info(f"@utils.py Combined {len(combined_results)} results successfully")
            
            # Process results
            extracted_urls = []
            try:
                self.logger.info(f"@utils.py Clearing cached document IDs")
                self.cached_doc_ids.clear()
                self.logger.debug(f"@utils.py Cached document IDs cleared")
            except Exception as e:
                self.logger.error(f"@utils.py Error clearing cached document IDs: {str(e)}")
                self.cached_doc_ids = []  # Reset if clearing fails
            
            try:
                self.logger.info(f"@utils.py Processing top {min(5, len(combined_results))} results")
                for doc in combined_results[:5]:
                    try:
                        doc_id = doc.get('metadata', {}).get('id')
                        if doc_id:
                            self.cached_doc_ids.append(doc_id)
                            self.logger.debug(f"@utils.py Added document ID to cache: {doc_id}")
                        
                        sources = doc.get('metadata', {}).get('url', [])
                        if isinstance(sources, str):
                            sources = [sources]  # Convert single URL to list
                        extracted_urls.extend(sources)
                        self.logger.debug(f"@utils.py Added sources: {sources}")
                    except Exception as e:
                        self.logger.error(f"@utils.py Error processing individual document: {str(e)}")
                self.logger.info(f"@utils.py Extracted {len(extracted_urls)} URLs from results")
            except Exception as e:
                self.logger.error(f"@utils.py Error processing search results: {str(e)}")
            
            try:
                self.logger.info(f"@utils.py Updating ground sources with {len(extracted_urls)} URLs : {extracted_urls}")
                self.update_ground_sources(extracted_urls)
                self.logger.info(f"@utils.py Ground sources updated successfully")
            except Exception as e:
                self.logger.error(f"@utils.py Error updating ground sources: {str(e)}")
            
            try:
                self.logger.info(f"@utils.py Formatting search results")
                formatted_context = self.format_search_results(combined_results[:5])
                self.logger.info(f"@utils.py Search results formatted successfully")
            except Exception as e:
                self.logger.error(f"@utils.py Error formatting search results: {str(e)}")
                formatted_context = "Error formatting search results."
        except Exception as e:
            self.logger.error(f"@utils.py Error in perform_database_search: {str(e)}")
            formatted_context = "Error performing database search."
        return formatted_context

    def update_ground_sources(self,extracted_urls:list):
        self.ground_sources.extend(extracted_urls)
        self.ground_sources = list(set(self.ground_sources))
    
    def get_ground_sources(self):
        return self.ground_sources
    
    def clear_ground_sources(self):
        self.ground_sources = []
        return True
    
    