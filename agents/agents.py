class Agents:
    def __init__(self, firestore, genai, app_config):
        self.firestore = firestore
        self.genai = genai
        self.app_config = app_config
        
        logger.info("\nInitialized ActionCommands")
        self.asu_data_agent = DataModel(firestore,genai,app_config)

        logger.info("\nInitialized DataModel")

        self.asu_live_status_agent = Live_Status_Model(firestore,genai,app_config)
        logger.info("\nInitialized LiveStatusAgent")


        self.asu_search_agent = SearchModel(firestore,genai,app_config)
        logger.info("\nInitialized SearchModel Instance")


        self.asu_discord_agent = DiscordModel(firestore,genai,app_config)
        logger.info("\nInitailized DIscord Model Instance")


        self.asu_action_agent = ActionModel(firestore,genai,app_config)
        logger.info("\nInitialized ActionAgent Global Instance")
