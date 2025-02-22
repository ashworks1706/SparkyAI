from utils.common_imports import *


class Main:
    def __init__(self):
        self.app_config = AppConfig()
        self.discord_state = DiscordState()
        self.asu_store = VectorStore(force_recreate=False)
        self.asu_data_processor = DataPreprocessor()
        self.firestore = Firestore(self.discord_state)
        self.utils = Utils(self.asu_store, self.asu_data_processor, None)  # asu_scraper will be set later
        self.asu_scraper = ASUWebScraper(self.discord_state, self.utils)
        self.utils.asu_scraper = self.asu_scraper  # Now set asu_scraper
        
        self.agents = Agents(self.firestore, genai, self.app_config, self.discord_state, self.utils)

        genai.configure(api_key=self.app_config.get_api_key())

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
        self.logger.info("\n----------------------------------------------------------------")
        self.logger.info("\nASU RAG INITIALIZED SUCCESSFULLY")
        self.logger.info("\n---------------------------------------------------------------")

    async def initialize_scraper(self):
        await self.asu_scraper.__login__(self.app_config.get_handshake_user(), self.app_config.get_handshake_pass())

    async def run_discord_bot(self, config: Optional[BotConfig] = None):
        """Run the Discord bot"""
        bot = ASUDiscordBot(config, self.agents, self.firestore, self.discord_state, self.utils, self.asu_store)
        
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
    config = BotConfig(
        token=asu_system.app_config.get_discord_bot_token(),
        app_config=asu_system.app_config
    )
    asyncio.run(asu_system.run_discord_bot(config))
