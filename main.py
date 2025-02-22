from utils.common_imports import *

class Main:
    def __init__(self):
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
        self.app_config = AppConfig()
        self.discord_state = DiscordState()
        
        try:
            self.asu_store = VectorStore(logger=self.logger, app_config=self.app_config, force_recreate=False)
        except Exception as e:
            self.logger.error(f"Failed to initialize VectorStore: {str(e)}")
            self.asu_store = None
            raise e

        self.asu_data_processor = DataPreprocessor(logger=self.logger)
        self.firestore = Firestore(self.discord_state)
        self.utils = Utils(self.asu_store, self.asu_data_processor, None, logger= self.logger)
        self.asu_scraper = ASUWebScraper(self.discord_state, self.utils, self.logger)
        self.utils.asu_scraper = self.asu_scraper

        self.agents = Agents(self.firestore, genai, self.app_config, self.discord_state, self.utils, self.logger)

        genai.configure(api_key=self.app_config.get_api_key())

        if self.asu_store:
            self.logger.info("\n----------------------------------------------------------------")
            self.logger.info("\nASU RAG INITIALIZED SUCCESSFULLY")
            self.logger.info("\n---------------------------------------------------------------")
        else:
            self.logger.warning("\nASU RAG INITIALIZED WITH ERRORS - VectorStore not available")

    async def initialize_scraper(self):
        await self.asu_scraper.__login__(self.app_config.get_handshake_user(), self.app_config.get_handshake_pass())

    async def run_discord_bot(self, config: Optional[BotConfig] = None):
        """Run the Discord bot"""
        if not self.asu_store:
            self.logger.error("Cannot start bot: VectorStore not initialized")
            return

        bot = ASUDiscordBot(config, self.agents, self.firestore, self.discord_state, self.utils, self.asu_store, self.logger)
        
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
    if asu_system.asu_store:
        config = BotConfig(
            token=asu_system.app_config.get_discord_bot_token(),
            app_config=asu_system.app_config
        )
        asyncio.run(asu_system.run_discord_bot(config))
    else:
        print("Bot initialization failed due to VectorStore error. Check logs for details.")
