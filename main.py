from utils.common_imports import *

class Main:
    def __init__(self):
        # Initializing common logger
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/data_processor.log'),
                logging.StreamHandler()
            ]
        )
        tracemalloc.start()
        self.logger = logging.getLogger(__name__)
        # Initializing app_config to get prompts, agent details and important secrets
        self.app_config = AppConfig()  
        # Initializing discord state class to dynamically update discord states since the server starts later
        self.discord_state = DiscordState()
        
        self.logger.info("Setting up Vector store @ Main")
        try:
            # initializing vector store for qdrant vector database
            self.vector_store = VectorStore(logger=self.logger, app_config=self.app_config, force_recreate=True)
            self.logger.info("VectorStore initialized successfully in @ Main")
        except Exception as e:
            self.logger.error(f"Failed to initialize VectorStore: {str(e)}")
            self.vector_store = None
            raise e
        # Initializing ASU RAG components
        genai.configure(api_key=self.app_config.get_api_key())
        self.logger.info("Setting up GenAI @ Main")
        self.group_chat = GroupChat("") 
        self.logger.info("Setting up GroupChat @ Main")
        self.asu_data_processor = DataPreprocessor(self.app_config,genai=genai,logger=self.logger)
        self.logger.info("Setting up ASU Data Processor @ Main")
        self.firestore = Firestore(self.discord_state)
        self.logger.info("Setting up ASU Firestore @ Main")
        self.utils = Utils(vector_store_class=self.vector_store, asu_data_processor=self.asu_data_processor, asu_scraper=None, logger=self.logger,group_chat=self.group_chat)
        self.logger.info("Setting up ASU Utils @ Main")
        self.asu_scraper = ASUWebScraper(self.discord_state, self.utils, self.logger)
        self.logger.info("Setting up ASU Web Scraper @ Main")
        self.utils.asu_scraper = self.asu_scraper
        self.logger.info("Setting up ASU Utils @ Main")
        self.agents = Agents(self.vector_store, self.asu_data_processor, self.firestore, genai, self.discord_state, self.utils, self.app_config, self.logger, self.group_chat)
        self.logger.info("Setting up ASU Agents @ Main")

        if self.vector_store:
            self.logger.info("----------------------------------------------------------------")
            self.logger.info("ASU RAG INITIALIZED SUCCESSFULLY")
            self.logger.info("---------------------------------------------------------------")
        else:
            self.logger.warning("\nASU RAG INITIALIZED WITH ERRORS - VectorStore not available")

    async def initialize_scraper(self):
        return True
        # await self.asu_scraper.__login__(self.app_config.get_handshake_user(), self.app_config.get_handshake_pass())

    async def run_discord_bot(self,config: Optional[BotConfig] = None, app_config=None):
        """Run the Discord bot"""
        if not self.vector_store:
            self.logger.error("Cannot start bot: VectorStore not initialized")
            return
        bot = ASUDiscordBot(config,app_config, self.agents, self.firestore, self.discord_state, self.utils, self.vector_store, self.logger)
        
        await self.initialize_scraper()
        
        try:
            await bot.start()
        except KeyboardInterrupt:
            self.logger.info("\nBot shutdown requested")
            await bot.close()
        except Exception as e:
            self.logger.error(f"Bot error: {str(e)}", exc_info=True)
            await bot.close()

if __name__ == "__main__":
    asu_system = Main()
    if asu_system.vector_store:
        config = BotConfig(
            token=asu_system.app_config.get_discord_bot_token(),
            app_config=asu_system.app_config
        )
        asyncio.run(asu_system.run_discord_bot(config,asu_system.app_config))
    else:
        print("Bot initialization failed due to VectorStore error. Check logs for details.")
