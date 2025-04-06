from utils.common_imports import *
from agents.data_agent import DataModel

class DataPreprocessor:
    def __init__(self, app_config, genai,
                 chunk_size: int = 1024, 
                 chunk_overlap: int = 200, 
                 max_processing_attempts: int = 3,  logger=False):
        """Initialize DataPreprocessor with configurable text splitting and retry mechanism."""
        self.max_processing_attempts = max_processing_attempts
        self.doc_title = None
        self.doc_category = None
        self.asu_data_agent = DataModel( app_config, genai,logger)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('wordnet', quiet=True)
        self.lemmatizer = WordNetLemmatizer()
        self.WHITESPACE_PATTERN = re.compile(r'\s+')
        self.LINK_PATTERN = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.SPECIAL_CHAR_PATTERN = re.compile(r'[^a-zA-Z0-9\s.,!?;:()\-"\'$]')

        self.logger= logger

        self.logger.info(f"@data_processor.py DataPreprocessor initialized with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")

    async def process_documents(self, 
                                documents: List[Dict[str, str]], 
                                search_context: str, 
                                title: str = None, 
                                category: str = None) -> List[Document]:
        """Process documents with advanced error handling and multiple retry mechanism."""
        self.logger.info(f"@data_processor.py Processing documents with title={title} category={category}")
        self.doc_title = title
        self.doc_category = category

        for attempt in range(self.max_processing_attempts):
            try:
                start_time = time.time()
                self.logger.info(f"@data_processor.py Starting document processing for {len(documents)} documents")
                try:
                    consolidated_text , refined_title = await self._consolidate_documents(documents, search_context)
                    self.doc_title = refined_title != "Undefined" and refined_title or title
                except Exception as e:
                    self.logger.error(f"@data_processor.py Document consolidation failed: {str(e)}")
                    raise
                
                try:
                    document = self._create_processed_document(consolidated_text, documents)
                except Exception as e:
                    self.logger.error(f"@data_processor.py Document creation failed: {str(e)}")
                    raise
                try:
                    processed_documents = self._split_and_annotate_document(document)
                except  Exception as e:
                    self.logger.error(f"@data_processor.py Document splitting failed: {str(e)}")
                    raise
                
                processing_time = time.time() - start_time
                self.logger.info(f"@data_processor.py Document processing completed in {processing_time:.2f} seconds. "
                            f"Generated {len(processed_documents)} document chunks.")

                return processed_documents

            except Exception as e:
                self.logger.error(f"@data_processor.py Document processing attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.max_processing_attempts - 1:
                    return await self._generate_fallback_document(documents, e)

    def clean_and_structure_text(self, text: str) -> str:
        """Enhanced text cleaning for RAG applications."""
        # Remove HTML tags
        text = BeautifulSoup(text, "html.parser").get_text()

        # Remove links
        text = self.LINK_PATTERN.sub('', text)

        # Convert to ASCII and lowercase
        text = unidecode(text.lower())

        # Replace $ with USD
        text = text.replace('$', 'USD ')

        # Remove special characters while preserving important punctuation
        text = self.SPECIAL_CHAR_PATTERN.sub(' ', text)

        # Tokenize
        tokens = word_tokenize(text)

        # Lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]

        # Rejoin tokens and normalize whitespace
        text = ' '.join(tokens)
        text = self.WHITESPACE_PATTERN.sub(' ', text).strip()

        return text
    async def _consolidate_documents(self, documents: List[Dict[str, str]], search_context: str) -> str:
        """Consolidate all documents into a single text corpus and refine the combined content."""
        all_content = "\n\n".join([doc['content'] for doc in documents])
        refined_title = "Undefined"

        try:
            # Use DataModel to refine the combined content
            refine_result = await self.asu_data_agent.refine(search_context, all_content)
            refined_content, refined_title = refine_result
            self.logger.info(f"@data_processor.py Combined text refinement successful")
        except Exception as e:
            self.logger.error(f"@data_processor.py Combined text refinement failed: {str(e)}")
            refined_content = all_content  # Fallback to original combined content if refinement fails

        return refined_content.strip(), refined_title


    def _create_processed_document(self, 
                                   consolidated_text: str, 
                                   documents: List[Dict[str, str]]) -> Document:
        """Create a processed document with comprehensive metadata."""
        
        urls = []
        for doc in documents:
            if 'metadata' in doc and 'url' in doc['metadata']:
                urls.append(doc['metadata']['url'])
        
        return Document(
            page_content=consolidated_text,
            metadata={
                'title': self.doc_title or 'Untitled',
                'category': self.doc_category or "google_results",
                'url': urls[:5],  # Use the extracted URLs
                'timestamp': datetime.now(),
                'total_source_documents': len(documents),
                'cluster_id': None
            }
        )

    def _split_and_annotate_document(self, document: Document) -> List[Document]:
        """Split document into chunks and annotate with metadata."""
        min_chunk_length = 500  # Define the minimum length for chunking

        if len(document.page_content) < min_chunk_length:
            # If the document is too short, don't split it
            document.metadata.update({
                'id': str(uuid.uuid4()),
                'chunk_id': str(uuid.uuid4()),
                'chunk_index': 0,
                'total_chunks': 1  # Indicate that it's a single chunk
            })
            return [document]
        else:
            # Use SemanticTextSplitter to avoid splitting paragraphs abruptly

            # Create semantic splitter (adjust model as needed)
            semantic_splitter = SemanticChunker(
                embeddings=OpenAIEmbeddings(),
                breakpoint_threshold=0.5,  # Adjust as needed
                chunk_size=1024,
                chunk_overlap=200
            )
            
            # Proceed with splitting if the document is long enough
            splits = semantic_splitter.split_documents([document])

            for i, split in enumerate(splits):
                split.metadata.update({
                    'id': str(uuid.uuid4()),
                    'chunk_id': str(uuid.uuid4()),
                    'chunk_index': i,
                    'total_chunks': len(splits)
                })

            return splits

    async def _generate_fallback_document(self, 
                                          documents: List[Dict[str, str]], 
                                          error: Exception) -> List[Document]:
        """Generate a fallback document when all processing attempts fail."""
        fallback_doc = Document(
            page_content=' '.join([doc['content'] for doc in documents]),
            metadata={
                'title': self.doc_title or 'Fallback Document',
                'category': self.doc_category,
                'url': [doc['metadata']["url"] for doc in documents],
                'timestamp': datetime.now(),
                'error_message': str(error),
                'total_source_documents': len(documents)
            }
        )

        return [fallback_doc]