from utils.common_imports import *
from agents.data_agent import DataModel
from agents.live_status_agent import Live_Status_Model
from agents.rag_search_agent import RagSearchModel
from agents.discord_agent import DiscordModel
from agents.superior_agent import SuperiorModel
from agent_tools.discord_agent_tools import Discord_Agent_Tools
from agent_tools.rag_search_agent_tools import Rag_Search_Agent_Tools
from agent_tools.live_status_agent_tools import Live_Status_Agent_Tools
from agent_tools.superior_agent_tools import Superior_Agent_Tools

class Agents:
    def __init__(self, firestore, genai, discord_state, utils, app_config,logger,group_chat):
        self.firestore = firestore
        self.genai = genai
        self.utils=utils
        self.discord_state = discord_state
        self.app_config = app_config 
        self.logger = logger
        self.group_chat = group_chat
        self.discord_agent_tools = Discord_Agent_Tools(firestore,discord_state,utils,logger)
        self.rag_search_agent_tools = Rag_Search_Agent_Tools(firestore,utils, app_config,logger)
        self.live_status_agent_tools = Live_Status_Agent_Tools(firestore, utils,logger)
        
        self.logger.info("\nInitialized Agent Tools")
        
        
        self.asu_data_agent = DataModel( self.genai,logger)

        self.logger.info("\nInitialized DataModel")

        self.live_status_agent = Live_Status_Model( self.firestore,self.genai,self.app_config,logger,self.live_status_agent_tools,discord_state)
        self.logger.info("\nInitialized LiveStatusAgent")

        self.rag_search_agent = RagSearchModel( self.firestore,self.genai,self.app_config,logger,self.rag_search_agent_tools,discord_state)
        self.logger.info("\nInitialized RagSearchModel Instance")


        self.discord_agent = DiscordModel( self.firestore,self.genai,self.app_config,logger,self.discord_agent_tools,discord_state)
        self.logger.info("\nInitailized DIscord Model Instance")


        self.superior_agent_tools = Superior_Agent_Tools(firestore,discord_state,utils, app_config, self.live_status_agent, self.rag_search_agent, self.discord_agent,logger,self.group_chat)
        self.superior_agent = SuperiorModel( self.firestore,self.genai,self.app_config,self.logger,self.superior_agent_tools)
        self.logger.info("\nInitialized ActionAgent Global Instance")
    
    async def process_question(self, question: str) -> str:
        return await self.superior_agent.determine_action(question)