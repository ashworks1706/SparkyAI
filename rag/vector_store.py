


from utils.common_imports import *
class VectorStore:
    
    """A from utils.common_imports import *
classto manage vector storage operations using Qdrant with enhanced logging and performance."""
    
    def __init__(self, 
                 force_recreate: bool = False,
                 host: str = "10.10.0.9",
                 port: int = 6333,
                 collection_name: str = "asu_docs",
                 model_name: str = "BAAI/bge-small-en-v1.5",
                #  model_name: str = "BAAI/bge-small-en-v1.5",
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
        self.vector_store: Optional[QdrantVectorStore] = None
        self.collection_name = collection_name
        self.batch_size = batch_size
        self.max_retry_attempts = max_retry_attempts
        self.retry_delay = retry_delay
        self.corpus = []
        self.hnsw_index = None


        
        logger.info(f"Initializing VectorStore with collection: {collection_name}")
        logger.info(f"Configuration: host={host}, port={port}, model={model_name}")
        
        try:
            self.client = self._create_qdrant_client(host, port)
            self._initialize_embedding_model(model_name)
            self._setup_collection(force_recreate)
            self._initialize_vector_store()
            
        except Exception as e:
            logger.error(f"Critical VectorStore initialization error: {str(e)}", exc_info=True)
            self._log_detailed_error(e)
            raise RuntimeError(f"VectorStore initialization failed: {str(e)}")
    
    def mips_search(self, query_vector: List[float], top_k: int = 5):
        try:
            if not self.vector_store:
                logger.error("Vector store not initialized.")
                raise ValueError("Vector store not properly initialized.")
            
            if self.hnsw_index is None:
                self.build_hnsw_index()
            
            labels, distances = self.hnsw_index.knn_query(query_vector, k=top_k)
            results = []
            for label, distance in zip(labels[0], distances[0]):
                doc = self.get_document_by_id(int(label))
                results.append({
                    "id": doc.metadata.get('id'),
                    "score": 1 - distance,  # Convert distance to similarity score
                    "payload": {
                        "page_content": doc.page_content,
                        "metadata": doc.metadata
                    }
                })
            
            logger.info(f"MIPS search retrieved {len(results)} results.")
            return results
        except Exception as e:
            logger.error(f"Error during MIPS search: {str(e)}")
            return []
        
    def get_all_documents(self):
        # Implement method to retrieve all documents from Qdrant
        results = self.client.scroll(
            collection_name=self.collection_name,
            limit=10000  # Adjust as needed
        )
        return [Document(page_content=item.payload["page_content"], metadata=item.payload["metadata"]) for item in results[0]]

    def get_embeddings(self, documents):
        return [self.embedding_model.embed_query(self.get_document_content(doc)) for doc in documents]

    def get_document_content(self, doc):
        if isinstance(doc, str):
            try:
                doc_dict = json.loads(doc)
                return doc_dict.get('page_content', '')
            except json.JSONDecodeError:
                return doc
        else:
            return getattr(doc, 'page_content', str(doc))
        
    def build_hnsw_index(self):
        all_docs = self.get_all_documents()
        all_embeddings = self.get_embeddings(all_docs)
        
        self.hnsw_index = hnswlib.Index(space='cosine', dim=self.vector_size)
        self.hnsw_index.init_index(max_elements=len(all_docs), ef_construction=200, M=16)
        self.hnsw_index.add_items(all_embeddings, np.arange(len(all_docs)))
        self.hnsw_index.set_ef(50)  # Adjust for speed/accuracy trade-off

    def _initialize_embedding_model(self, model_name: str) -> None:
        """Initialize the embedding model."""
        logger.info(f"Initializing embedding model: {model_name}")
        try:
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': 'cpu'}
            )
            self.vector_size = len(self.embedding_model.embed_query("test"))
            logger.info(f"Embedding model initialized with vector size: {self.vector_size}")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {str(e)}", exc_info=True)
            raise
    
    def _create_qdrant_client(self, host: str, port: int) -> QdrantClient:
        for attempt in range(self.max_retry_attempts):
            try:
                api_key = app_config.get_qdrant_api_key()  # Use the Kubernetes API key for authentication
                client = QdrantClient(
                url="https://4bfefa3a-9337-4325-9836-5f054c1de8d8.us-east-1-0.aws.cloud.qdrant.io",
                api_key=api_key,
                prefer_grpc=False
                )

                logger.info(f"Successfully connected to Qdrant at {host}:{port} (Attempt {attempt + 1})")
                return client
            except Exception as e:
                if attempt == self.max_retry_attempts - 1:
                    logger.error(f"Failed to connect to Qdrant after {self.max_retry_attempts} attempts")
                    raise
                logger.warning(f"Qdrant connection attempt {attempt + 1} failed: {str(e)}")
                time.sleep(self.retry_delay)
    
    def _verify_collection_dimensions(self) -> None:
        """Verify that existing collection dimensions match the model."""
        collection_info = self.client.get_collection(self.collection_name)
        existing_size = collection_info.config.params.vectors.size
        
        if existing_size != self.vector_size:
            error_msg = (f"Dimension mismatch: Collection has {existing_size}, "
                        f"model requires {self.vector_size}")
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info(f"Verified collection dimensions: {existing_size}")
    
    def _log_detailed_error(self, exception: Exception) -> None:
        """
        Log detailed error information for diagnostics.
        
        Args:
            exception (Exception): The exception to log details for
        """
        logger.error("Detailed Error Diagnostics:")
        logger.error(f"Error Type: {type(exception).__name__}")
        logger.error(f"Error Message: {str(exception)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
    
    def _setup_collection(self, force_recreate: bool) -> None:
        """Set up the Qdrant collection."""
        try:
            collections = self.client.get_collections().collections
            collection_exists = any(c.name == self.collection_name for c in collections)
            
            if collection_exists:
                if force_recreate:
                    logger.info(f"Force recreating collection: {self.collection_name}")
                    self.client.delete_collection(self.collection_name)
                    collection_exists = False
                else:
                    self._verify_collection_dimensions()
            
            if not collection_exists:
                self._create_collection()
        except Exception as e:
            logger.error(f"Failed to setup collection: {str(e)}", exc_info=True)
    
    def _create_collection(self) -> None:
        """Create a new Qdrant collection."""
        logger.info(f"Creating new collection: {self.collection_name}")
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
        logger.info("\nCollection created successfully")
    
    def queue_documents(self, docs: List[Document]) -> None:
        """Queue documents for storage."""
        self.corpus.extend(docs)
        logger.info("Queued Processed Documents")
        return True
  
    async def store_to_vector_db(self) -> bool:
        if self.vector_store is None and self.corpus is None:
            logger.critical("Vector store not initialized - cannot proceed")
            raise ValueError("Vector store not properly initialized")

        total_docs = len(self.corpus)
        logger.info(f"Document storage initiated: {total_docs} documents to process")
        performance_start = time.time()
        processed_count = 0
        skipped_count = 0
        error_count = 0
        try:
            for doc in self.corpus:
                logger.debug(f"Processing document {processed_count + 1}/{total_docs}")
                try:
                    should_store = self._should_store_document(doc)
                    if should_store:
                        await self.vector_store.aadd_documents([doc])
                        processed_count += 1
                        logger.info(f"Successfully stored document {processed_count}")
                    else:
                        skipped_count += 1
                except Exception as e:
                    logger.error(f"Error processing document {doc.metadata.get('url', 'Unknown')}: {str(e)}")
                    error_count += 1

            performance_end = time.time()
            logger.info(f"Total Documents: {total_docs}")
            logger.info(f"Processed Documents: {processed_count}")
            logger.info(f"Skipped Documents: {skipped_count}")
            logger.info(f"Error Documents: {error_count}")
            logger.info(f"Total Processing Time: {performance_end - performance_start:.2f} seconds")
            return True
        except Exception as e:
            logger.error(f"Catastrophic document storage failure: {str(e)}", exc_info=True)
            self._log_detailed_error(e)
            raise

    def _initialize_vector_store(self) -> None:
        """Initialize the QdrantVectorStore."""
        logger.info("\nInitializing QdrantVectorStore")
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embedding_model,
            content_payload_key="page_content",
            metadata_payload_key="metadata",
            distance=Distance.COSINE
        )
        
    
    def get_vector_store(self):
        return self.vector_store

    def _should_store_document(self, doc: Document) -> bool:
        try:
            urls = doc.metadata['url'] if isinstance(doc.metadata['url'], list) else [doc.metadata['url']]
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
            if existing_docs:
                logger.info("Found existing Docs\n")
                logger.info(existing_docs)
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
                        logger.info(f"Replaced document: {urls} due to significant changes")
                    
                    logger.debug(f"Skipping document with minimal time difference: {urls}")
            return True
        except Exception as e:
            logger.error(f"Document evaluation error: {str(e)}")
            raise
    
    