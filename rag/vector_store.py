from utils.common_imports import *
import torch
class VectorStore:
    
    """A from utils.common_imports import *
classto manage vector storage operations using Qdrant with enhanced logging and performance."""
    
    def __init__(self, logger,app_config=False,
                 force_recreate: bool = False,
                #  host: str = "10.10.0.9",
                 host: str = os.environ.get("HOST_QDRANT", "localhost"),
                 port: int = 6333,
                 collection_name: str = "asu_docs",
                 model_name: str = "BAAI/bge-large-en-v1.5",
                 batch_size: int = 100,
                 max_retry_attempts: int = 3,
                 retry_delay: int = 2):
        """
        Initialize the VectorStore with specified parameters and enhanced error handling.
        
        Args:
            force_recreate (bool): Whether to recreate the collection if it exists
            host (str): Qdrant server host
            port (int): Qdrant server port
            collection_name (str): Name of the collection
            model_name (str): Name of the embedding model
            batch_size (int): Size of batches for document processing
            max_retry_attempts (int): Maximum number of retry attempts for operations
            retry_delay (int): Delay between retry attempts in seconds
        """    
        try:
            self.vector_store: Optional[QdrantVectorStore] = None
            self.collection_name = collection_name
            self.batch_size = batch_size
            self.max_retry_attempts = max_retry_attempts
            self.retry_delay = retry_delay
            self.corpus = []
            self.app_config = app_config
            self.logger= logger
            
            self.logger.info(f"@vector_store.py Initializing VectorStore with collection: {collection_name}")
            self.logger.info(f"@vector_store.py Configuration: host={host}, port={port}, model={model_name}")
            
            self.client = self._create_qdrant_client(host, port)
            self._initialize_embedding_model(model_name)
            self._setup_collection(force_recreate)
            self._initialize_vector_store()
            
            
        except Exception as e:
            self.logger.error(f"@vector_store.py Critical VectorStore initialization error: {str(e)}", exc_info=True)
            self._log_detailed_error(e)
            raise RuntimeError(f"VectorStore initialization failed: {str(e)}")
    

        
    def get_all_documents(self):
        # Implement method to retrieve all documents from Qdrant
        results = self.client.scroll(
            collection_name=self.collection_name,
            limit=10000  # Adjust as needed
        )
        return [Document(page_content=item.payload["page_content"], metadata=item.payload["metadata"]) for item in results[0]]

    def get_embeddings(self, documents):
        # Implement method to get embeddings for a list of documents
        if not documents:
            self.logger.warning(f"@vector_store.py No documents provided for embedding.")
            return []
        return [self.embedding_model.embed_query(self.get_document_content(doc)) for doc in documents]

    def get_document_content(self, doc):
        #"""Extract the content from a document."""
        self.logger.debug(f"Extracting content from document: {doc}")
        if isinstance(doc, str):
            # Check if the string looks like JSON before trying to parse it
            if doc.strip().startswith('{') and doc.strip().endswith('}'):
                try:
                    doc_dict = json.loads(doc)
                    return doc_dict.get('page_content', '')
                except json.JSONDecodeError:
                    self.logger.warning(f"@vector_store.py Failed to decode JSON from document: {doc}")
                    return doc
            else:
                # Not JSON-formatted, return as is
                return doc
        else:
            return getattr(doc, 'page_content', str(doc))
        

    def _initialize_embedding_model(self, model_name: str) -> None:
        """Initialize the embedding model."""
        self.logger.info(f"@vector_store.py Initializing embedding model: {model_name}")
        try:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': device}
            )
            self.vector_size = len(self.embedding_model.embed_query("test"))
            self.logger.info(f"@vector_store.py Embedding model initialized with vector size: {self.vector_size}")
        except Exception as e:
            self.logger.error(f"@vector_store.py Failed to initialize embedding model: {str(e)}", exc_info=True)
            raise
    
    def _create_qdrant_client(self, host: str, port: int) -> QdrantClient:
        for attempt in range(self.max_retry_attempts):
            try:
                api_key = self.app_config.get_qdrant_api_key()  # Use the Kubernetes API key for authentication
                # client = QdrantClient(
                # url="https://4bfefa3a-9337-4325-9836-5f054c1de8d8.us-east-1-0.aws.cloud.qdrant.io",
                # api_key=api_key,
                # prefer_grpc=False
                # )
                client = QdrantClient(host=host, port=port)

                self.logger.info(f"@vector_store.py Successfully connected to Qdrant at {host}:{port} (Attempt {attempt + 1})")
                return client
            except Exception as e:
                if attempt == self.max_retry_attempts - 1:
                    self.logger.error(f"@vector_store.py Failed to connect to Qdrant after {self.max_retry_attempts} attempts")
                    raise
                self.logger.warning(f"@vector_store.py Qdrant connection attempt {attempt + 1} failed: {str(e)}")
                time.sleep(self.retry_delay)
    
    def _verify_collection_dimensions(self) -> None:
        """Verify that existing collection dimensions match the model."""
        self.logger.info(f"@vector_store.py Verifying collection dimensions for: {self.collection_name}")
        collection_info = self.client.get_collection(self.collection_name)
        existing_size = collection_info.config.params.vectors.size
        
        if existing_size != self.vector_size:
            error_msg = (f"Dimension mismatch: Collection has {existing_size}, "
                        f"model requires {self.vector_size}")
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        self.logger.info(f"@vector_store.py Verified collection dimensions: {existing_size}")
    
    def _log_detailed_error(self, exception: Exception) -> None:
        """
        Log detailed error information for diagnostics.
        
        Args:
            exception (Exception): The exception to log details for
        """
        self.logger.error(f"@vector_store.py Detailed Error Diagnostics:")
        self.logger.error(f"@vector_store.py Error Type: {type(exception).__name__}")
        self.logger.error(f"@vector_store.py Error Message: {str(exception)}")
        self.logger.error(f"@vector_store.py Traceback: {traceback.format_exc()}")
    
    def _setup_collection(self, force_recreate: bool) -> None:
        """Set up the Qdrant collection."""
        try:
            self.logger.info(f"@vector_store.py Setting up collection: {self.collection_name}")
            collections = self.client.get_collections().collections
            collection_exists = any(c.name == self.collection_name for c in collections)
            
            if collection_exists:
                if force_recreate:
                    self.logger.info(f"@vector_store.py Force recreating collection: {self.collection_name}")
                    self.client.delete_collection(self.collection_name)
                    collection_exists = False
                else:
                    self.logger.info(f"@vector_store.py Collection already exists: {self.collection_name}")
                    self._verify_collection_dimensions()
            
            if not collection_exists:
                self._create_collection()
        except Exception as e:
            self.logger.error(f"@vector_store.py Failed to setup collection: {str(e)}", exc_info=True)
    
    def _create_collection(self) -> None:
        """Create a new Qdrant collection."""
        self.logger.info(f"@vector_store.py Creating new collection: {self.collection_name}")
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_size,
                distance=Distance.COSINE
            ),
            optimizers_config=OptimizersConfigDiff(
                default_segment_number=2,
                memmap_threshold=20000
            )
        )
        self.logger.info(f"@vector_store.py  Collection created successfully")
    
    def queue_documents(self, docs: List[Document]) -> None:
        """Queue documents for storage."""
        self.corpus.extend(docs)
        self.logger.info(f"@vector_store.py Queued Processed Documents")
        return True
  
    async def store_to_vector_db(self) -> bool:
        # """Store documents to the vector database."""
        if self.vector_store is None and self.corpus is None:
            self.logger.critical("Vector store not initialized - cannot proceed")
            raise ValueError("Vector store not properly initialized")
                #  model_name: str = "BAAI/bge-small-en-v1.5",

        total_docs = len(self.corpus)
        self.logger.info(f"@vector_store.py Document storage initiated: {total_docs} documents to process")
        performance_start = time.time()
        processed_count = 0
        skipped_count = 0
        error_count = 0
        try:
            for doc in self.corpus:
                self.logger.debug(f"Processing document {processed_count + 1}/{total_docs}")
                try:
                    should_store = self._should_store_document(doc)
                    if should_store:
                        await self.vector_store.aadd_documents([doc])
                        processed_count += 1
                        self.logger.info(f"@vector_store.py Successfully stored document {processed_count}")
                    else:
                        skipped_count += 1
                except Exception as e:
                    self.logger.error(f"@vector_store.py Error processing document {doc.metadata.get('url', 'Unknown')}: {str(e)}")
                    error_count += 1

            performance_end = time.time()
            self.logger.info(f"@vector_store.py Total Documents: {total_docs}")
            self.logger.info(f"@vector_store.py Processed Documents: {processed_count}")
            self.logger.info(f"@vector_store.py Skipped Documents: {skipped_count}")
            self.logger.info(f"@vector_store.py Error Documents: {error_count}")
            self.logger.info(f"@vector_store.py Total Processing Time: {performance_end - performance_start:.2f} seconds")
            return True
        except Exception as e:
            self.logger.error(f"@vector_store.py Catastrophic document storage failure: {str(e)}", exc_info=True)
            self._log_detailed_error(e)
            raise

    def _initialize_vector_store(self) -> None:
        """Initialize the QdrantVectorStore."""
        self.logger.info(f"@vector_store.py  Initializing QdrantVectorStore")
        try:
            self.vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=self.embedding_model,
                content_payload_key="page_content",
                metadata_payload_key="metadata",
                distance=Distance.COSINE
            )
        except Exception as e:
            self.logger.error(f"@vector_store.py Failed to initialize QdrantVectorStore: {str(e)}", exc_info=True)

    # return number of documents present in database
    def get_document_count(self) -> int:
        """Get the number of documents in the vector store."""
        try:
            count = self.client.count(collection_name=self.collection_name).count
            self.logger.info(f"@vector_store.py Document count: {count}")
            return count
        except Exception as e:
            self.logger.error(f"@vector_store.py Failed to get document count: {str(e)}", exc_info=True)
            raise
    
    def get_vector_store(self):
        self.logger.info(f"@vector_store.py Returning vector store instance")
        return self.vector_store

    def _should_store_document(self, doc: Document) -> bool:
        self.logger.info(f"@vector_store.py Evaluating document for storage: {doc.metadata.get('url', 'Unknown')}")
        try:
            # Flatten the URL list if it's a nested list
            url_data = doc.metadata.get('url', 'Unknown')
            if isinstance(url_data, list):
                # Flatten nested lists
                flat_urls = []
                for url in url_data:
                    if isinstance(url, list):
                        flat_urls.extend(url)
                    else:
                        flat_urls.append(url)
                urls = flat_urls
            else:
                urls = [url_data]
                
            existing_docs = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.url", 
                            match=models.MatchAny(any=urls)
                        )
                    ]
                )
            )[0]
            self.logger.info(f"@vector_store.py Existing documents found: {len(existing_docs)}")
            if existing_docs:
                self.logger.info(f"@vector_store.py Found existing Docs ")
                self.logger.info(existing_docs)
                new_timestamp = doc.metadata.get('timestamp')                
                for existing_doc in existing_docs:
                    existing_timestamp = existing_doc.payload.get('metadata', {}).get('timestamp')
                    # Ensure both timestamps are datetime OBJECTs
                    if isinstance(new_timestamp, str):
                        new_timestamp = datetime.fromisoformat(new_timestamp)
                    if isinstance(existing_timestamp, str):
                        existing_timestamp = datetime.fromisoformat(existing_timestamp)
                    # Enhanced timestamp comparison with configurable threshold
                    if (not existing_timestamp or 
                        (new_timestamp and 
                        (new_timestamp - existing_timestamp).total_seconds() >= 60 * 60)):
                        # Delete outdated document
                        self.client.delete(
                            collection_name=self.collection_name,
                            points_selector=models.PointIdsList(points=[existing_doc.id])
                        )
                        self.logger.info(f"@vector_store.py Replaced document: {urls} due to significant changes")
                    
                    self.logger.debug(f"Skipping document with minimal time difference: {urls}")
            self.logger.info(f"@vector_store.py Document evaluation complete: {urls}")
            return True
        except Exception as e:
            self.logger.error(f"@vector_store.py Document evaluation error: {str(e)}")
            raise
    
    def update_document_metadata(self, doc_id: str, metadata_update: dict) -> bool:
        """
        Update the metadata of a document in the vector store.
        
        Args:
            doc_id (str): The ID of the document to update
            metadata_update (dict): Dictionary containing metadata fields to update
            
        Returns:
            bool: True if update was successful
        """
        try:
            self.logger.info(f"@vector_store.py Updating metadata for document {doc_id}")
            self.logger.debug(f"@vector_store.py Metadata update payload: {metadata_update}")
            
            # Get all documents to find the one with the matching ID in metadata
            all_docs = self.get_all_documents()
            self.logger.info(f"@vector_store.py All documents retrieved: {len(all_docs)} documents")
            
            # Find the document with the matching ID in its metadata
            target_doc = None
            point_id = None
            
            # Get all points with their IDs directly from Qdrant
            search_results = self.client.scroll(
                collection_name=self.collection_name,
                with_payload=True,
                with_vectors=False,
                limit=10000
            )[0]  # First element contains the points
            
            for point in search_results:
                if point.payload.get("metadata", {}).get("id") == doc_id:
                    point_id = point.id
                    target_doc = point
                    break
            
            if not target_doc:
                self.logger.error(f"@vector_store.py Document with metadata ID {doc_id} not found")
                return False
                
            try:
                existing_metadata = target_doc.payload.get("metadata", {})
                self.logger.debug(f"@vector_store.py Existing metadata: {existing_metadata}")
                
                # Update metadata while preserving existing fields
                updated_metadata = {**existing_metadata, **metadata_update}
                self.logger.debug(f"@vector_store.py Updated metadata: {updated_metadata}")
                
            except Exception as metadata_error:
                self.logger.error(f"@vector_store.py Error processing metadata: {str(metadata_error)}")
                self._log_detailed_error(metadata_error)
                return False
            
            # Perform the update operation using the point_id
            try:
                self.client.set_payload(
                    collection_name=self.collection_name,
                    payload={"metadata": updated_metadata},
                    points=[point_id]
                )
                self.logger.info(f"@vector_store.py Successfully updated metadata for document {doc_id}")
                return True
                
            except Exception as update_error:
                self.logger.error(f"@vector_store.py Failed to set payload: {str(update_error)}")
                self._log_detailed_error(update_error)
                return False
            
        except Exception as e:
            self.logger.error(f"@vector_store.py Failed to update document metadata: {str(e)}", exc_info=True)
            self._log_detailed_error(e)
            return False

        