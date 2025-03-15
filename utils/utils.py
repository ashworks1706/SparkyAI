from utils.common_imports import *
from rag.raptor import RaptorRetriever
class Utils:
    def __init__(self,vector_store,asu_data_processor,asu_scraper,logger,group_chat):
        """Initialize the Utils from utils.common_imports import *
classwith task tracking and logging."""
        try:
            self.tasks = []
            self.vector_store_class = vector_store
            self.asu_data_processor = asu_data_processor
            self.asu_scraper = asu_scraper
            self.current_content = "Understanding your question"
            self.message = None
            self.cached_doc_ids = []
            self.ground_sources =[]
            # self.scann_store = None
            self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
            self.cached_queries=[]
            self.vector_store = vector_store.get_vector_store()
            self.logger=logger
            self.raptor_retriever = RaptorRetriever(vector_store=self.vector_store_class,logger=self.logger)
            self.group_chat=group_chat
            self.logger.info(f"Group Chat setup successfully {group_chat}")

            self.logger.info("\nUtils instance initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Utils: {e}")

    async def start_animation(self, message):
        """Start the loading animation using Discord's built-in thinking indicator"""
        try:
            self.message = message
            self.logger.info(f"Animation started for message: {message.id}")
        except Exception as e:
            self.logger.error(f"Failed to start animation: {e}")

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
            self.logger.info(f"Message updated with {len(display_lines)} tasks")

        except Exception as e:
            self.logger.error(f"Failed to update text: {e}")
            # Optionally, you could re-raise the exception or handle it differently

    async def stop_animation(self, message=None, final_content=None,View=None):
        """Stop animation and display final content"""
        try:
            # Edit message with final content if provided
            if message and final_content:
                if View:
                    await message.edit(content=final_content,view=View)
                await message.edit(content=final_content)
                self.logger.info(f"Final content set: {final_content}")

            # Reset internal state
            self.tasks = []
            self.current_content = ""
            self.message = None
            self.logger.info("\nAnimation stopped and state reset")

        except Exception as e:
            self.logger.error(f"Error stopping animation: {e}")

    def format_search_results(self, engine_context):
        """Format search results into a readable string."""
        if not engine_context:
            return "No search results found."
        try:
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
            self.logger.error(f"Error formatting search results: {str(e)}")
            return "Error formatting search results."

    async def perform_web_search(self,search_url:str =None,  optional_query : str = None, doc_title : str =None, doc_category : str = None):
        try:
            # Initial search
            self.logger.info("\nPerforming Web Search")

            documents = await self.asu_scraper.engine_search(search_url, optional_query)

            if not documents:
                raise ValueError("No documents found matching the query")
                return "No results found on web"
            
            self.logger.info(documents)
            
            self.logger.info("\nPreprocessing documents...")
            
            processed_docs = await self.asu_data_processor.process_documents(
                documents=documents, 
                search_context= self.group_chat.get_text(),
                title = doc_title, category = doc_category
            )

            store = self.vector_store_class.queue_documents(processed_docs)

            results = []
            extracted_urls=[]
            for doc in processed_docs:
                doc_info = {
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'timestamp': doc.metadata.get('timestamp'),
                    'url': doc.metadata.get('url')
                }
                sources = doc.metadata.get('url')
                extracted_urls.extend(sources)

                results.append(doc_info)

            self.update_ground_sources(extracted_urls)

            results = self.format_search_results(results)

            return results

        except Exception as e:
            self.logger.error(f"Error in web search: {str(e)}")
            return "No results found on web"
       
    async def perform_similarity_search(self, query: str, categories: list):
        try:
            self.logger.info(f"Action Model: Performing similarity search with query: {query}")
            self.vector_store = self.vector_store.get_vector_store()
            if not self.vector_store:
                self.logger.info("\nVector Store not initialized")
                raise ValueError("Vector store not properly initialized")

            # Correct filter construction using Qdrant's Filter class
            filter_conditions = None
            if categories and len(categories) > 0:
                

                filter_conditions = Filter(
                    must=[
                        FieldCondition(
                            key="metadata.category", 
                            match=MatchAny(any=categories)
                        )
                    ]
                )

            # Perform similarity search with optional filtering
            results = self.vector_store.similarity_search(
                query, 
                filter=filter_conditions
            )

            # Check if results are empty
            if not results:
                self.logger.info("\nNo documents found in vector store")
                return None

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

            self.logger.info(f"Retrieved {len(documents)} documents from vector store")
            return documents

        except Exception as e:
            self.logger.error(f"Error during similarity search: {str(e)}")
            return None


    
    def merge_search_results(self, raptor_results, similarity_results):
        combined_results = raptor_results + similarity_results 
        
        deduplicated_results = []
        seen_urls = set()
        
        for result in combined_results:
            url = result.get('metadata', {}).get('url')
            if url not in seen_urls:
                seen_urls.add(url)
                deduplicated_results.append(result)
        
        # Sort by relevance score (assuming higher is better)
        sorted_results = sorted(deduplicated_results, key=lambda x: x.get('score', 0), reverse=True)
        
        return sorted_results[:10]  # Return top 10 unique results

    async def perform_database_search(self, query: str, categories: list):
        self.cached_queries.append(query)
        
        # Perform RAPTOR search
        raptor_results = await self.raptor_retriever.retrieve(query, top_k=5)
        
        # Perform similarity search
        similarity_results = await self.perform_similarity_search(query, categories)
        

        # Combine and deduplicate results
        combined_results = self.merge_search_results(raptor_results, similarity_results)
        
        # Process results
        extracted_urls = []
        self.cached_doc_ids.clear()
        
        for doc in combined_results[:5]:
            doc_id = doc.get('metadata', {}).get('id')
            if doc_id:
                self.cached_doc_ids.append(doc_id)
            sources = doc.get('metadata', {}).get('url', [])
            extracted_urls.extend(sources)
        
        self.update_ground_sources(extracted_urls)
        formatted_context = self.format_search_results(combined_results[:5])
        return formatted_context

    def update_ground_sources(self,extracted_urls:[]):
        self.ground_sources.extend(extracted_urls)
        self.ground_sources = list(set(self.ground_sources))
    
    def get_ground_sources(self):
        return self.ground_sources
    
    def clear_ground_sources(self):
        self.ground_sources = []
        return True
    