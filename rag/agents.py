from utils.common_imports import *
class Agents:
    def __init__(self, firestore, genai, discord_state, utils, app_config):
        self.firestore = firestore
        self.genai = genai
        self.app_config = app_config
        
        self.asu_data_agent = DataModel( self.firestore,self.genai,self.app_config)

        logger.info("\nInitialized DataModel")

        self.live_status_agent = Live_Status_Model( self.firestore,self.genai,self.app_config)
        logger.info("\nInitialized LiveStatusAgent")

 
        self.rag_rag_search_agent = RagSearchModel( self.firestore,self.genai,self.app_config)
        logger.info("\nInitialized RagSearchModel Instance")


        self.discord_agent = DiscordModel( self.firestore,self.genai,self.app_config)
        logger.info("\nInitailized DIscord Model Instance")


        self.superior_agent = SuperiorModel( self.firestore,self.genai,self.app_config)
        logger.info("\nInitialized ActionAgent Global Instance")
        
        self.agent_tools = AgentTools(firestore,discord_state,utils,app_config, self.live_status_agent, self.rag_rag_search_agent, self.discord_agent)
        logger.info("\nInitialized Agent Tools")
    
    async def execute_function(self, function_call: Any) -> str:
        function_mapping = {
            'access_rag_search_agent': self.agent_tools.access_rag_search_agent,
            'access_google_agent': self.agent_tools.access_google_agent,
            'access_discord_agent': self.agent_tools.access_discord_agent,
            'send_bot_feedback': self.agent_tools.send_bot_feedback,
            'access_live_status_agent': self.agent_tools.access_live_status_agent,
            'get_user_profile_details': self.agent_tools.get_user_profile_details,
            'get_discord_server_info': self.agent_tools.get_discord_server_info,
            'notify_moderators': self.agent_tools.notify_moderators,
            'notify_discord_helpers': self.agent_tools.notify_discord_helpers,
            'create_discord_forum_post': self.agent_tools.create_discord_forum_post,
            'create_discord_announcement': self.agent_tools.create_discord_announcement,
            'create_discord_poll': self.agent_tools.create_discord_poll,
            'search_discord': self.agent_tools.search_discord,
             'get_live_library_status': self.agent_tools.get_live_library_status,
            'get_live_shuttle_status': self.agent_tools.get_live_shuttle_status,
             'get_latest_club_information': self.agent_tools.get_latest_club_information,
            'get_latest_event_updates': self.agent_tools.get_latest_event_updates,
            'get_latest_news_updates': self.agent_tools.get_latest_news_updates,
            'get_latest_social_media_updates': self.agent_tools.get_latest_social_media_updates,
            'get_latest_sport_updates': self.agent_tools.get_latest_sport_updates,
           'get_library_resources': self.agent_tools.get_library_resources,
              'get_latest_scholarships': self.agent_tools.get_latest_scholarships,
            'get_latest_job_updates': self.agent_tools.get_latest_job_updates,
            'get_latest_class_information': self.agent_tools.get_latest_class_information
        }

        function_name = function_call.name
        function_args = function_call.args

        if function_name not in function_mapping:
            raise ValueError(f"Unknown function: {function_name}")
        
        function_to_call = function_mapping[function_name]
        return await function_to_call(**function_args)
    async def process_question(self, question: str) -> str:
        return self.superior_agent.process_question(question)