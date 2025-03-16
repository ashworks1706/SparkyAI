from utils.common_imports import *
from agents.data_agent import DataModel
from agents.live_status_agent import Live_Status_Model
# from agents.rag_search_agent import RagSearchModel
from agents.discord_agent import DiscordModel
from agents.superior_agent import SuperiorModel
from agents.courses_agent import CoursesModel
from agents.events_agent import EventsModel
from agents.library_agent import LibraryModel
from agents.news_agent import NewsModel
from agents.scholarship_agent import ScholarshipModel
from agents.social_media_agent import SocialMediaModel
from agents.sports_agent import SportsModel
from agents.student_club_agent import StudentClubModel
from agents.student_jobs_agent import StudentJobsModel

from agent_tools.discord_agent_tools import Discord_Agent_Tools
# from agent_tools.rag_search_agent_tools import Rag_Search_Agent_Tools
from agent_tools.live_status_agent_tools import Live_Status_Agent_Tools
from agent_tools.superior_agent_tools import Superior_Agent_Tools
from agent_tools.courses_agent_tools import Courses_Agent_Tools
from agent_tools.events_agent_tools import Events_Agent_Tools
from agent_tools.library_agent_tools import Library_Agent_Tools
from agent_tools.news_agent_tools import News_Agent_Tools
from agent_tools.scholarship_agent_tools import Scholarship_Agent_Tools
from agent_tools.social_media_agent_tools import Social_Media_Agent_Tools
from agent_tools.sports_agent_tools import Sports_Agent_Tools
from agent_tools.student_club_agent_tools import Student_Club_Agent_Tools
from agent_tools.student_jobs_agent_tools import Student_Jobs_Agent_Tools

class Agents:
    def __init__(self, firestore, genai, discord_state, utils, app_config, logger, group_chat):
        self.firestore = firestore
        self.genai = genai
        self.utils = utils
        self.discord_state = discord_state
        self.app_config = app_config
        self.logger = logger
        self.group_chat = group_chat

        self.discord_agent_tools = Discord_Agent_Tools(firestore, discord_state, utils, logger)
        # self.rag_search_agent_tools = Rag_Search_Agent_Tools(firestore, utils, app_config, logger)
        self.live_status_agent_tools = Live_Status_Agent_Tools(firestore, utils, logger)
        self.courses_agent_tools = Courses_Agent_Tools(firestore, utils, logger)
        self.events_agent_tools = Events_Agent_Tools(firestore, utils, logger)
        self.library_agent_tools = Library_Agent_Tools(firestore, utils, logger)
        self.news_agent_tools = News_Agent_Tools(firestore, utils, logger)
        self.scholarship_agent_tools = Scholarship_Agent_Tools(firestore, utils, logger)
        self.social_media_agent_tools = Social_Media_Agent_Tools(firestore, utils, logger)
        self.sports_agent_tools = Sports_Agent_Tools(firestore, utils, logger)
        self.student_club_agent_tools = Student_Club_Agent_Tools(firestore, utils, logger)
        self.student_jobs_agent_tools = Student_Jobs_Agent_Tools(firestore, utils, logger)

        self.logger.info("Initialized Agent Tools")

        self.asu_data_agent = DataModel(self.genai, logger)
        self.logger.info("Initialized DataModel")

        self.live_status_agent = Live_Status_Model(self.firestore, self.genai, self.app_config, logger, self.live_status_agent_tools, discord_state)
        self.logger.info("Initialized LiveStatusAgent")

        # self.rag_search_agent = RagSearchModel(self.firestore, self.genai, self.app_config, logger, self.rag_search_agent_tools, discord_state)
        # self.logger.info("Initialized RagSearchModel Instance")

        self.discord_agent = DiscordModel(self.firestore, self.genai, self.app_config, logger, self.discord_agent_tools, discord_state)
        self.logger.info("Initialized Discord Model Instance")

        self.courses_agent = CoursesModel(self.firestore, self.genai, self.app_config, logger, self.courses_agent_tools, discord_state)
        self.logger.info("Initialized CoursesModel Instance")

        self.events_agent = EventsModel(self.firestore, self.genai, self.app_config, logger, self.events_agent_tools, discord_state)
        self.logger.info("Initialized EventsModel Instance")

        self.library_agent = LibraryModel(self.firestore, self.genai, self.app_config, logger, self.library_agent_tools, discord_state)
        self.logger.info("Initialized LibraryModel Instance")

        self.news_agent = NewsModel(self.firestore, self.genai, self.app_config, logger, self.news_agent_tools, discord_state)
        self.logger.info("Initialized NewsModel Instance")

        self.scholarship_agent = ScholarshipModel(self.firestore, self.genai, self.app_config, logger, self.scholarship_agent_tools, discord_state)
        self.logger.info("Initialized ScholarshipModel Instance")

        self.social_media_agent = SocialMediaModel(self.firestore, self.genai, self.app_config, logger, self.social_media_agent_tools, discord_state)
        self.logger.info("Initialized SocialMediaModel Instance")

        self.sports_agent = SportsModel(self.firestore, self.genai, self.app_config, logger, self.sports_agent_tools, discord_state)
        self.logger.info("Initialized SportsModel Instance")

        self.student_club_agent = StudentClubModel(self.firestore, self.genai, self.app_config, logger, self.student_club_agent_tools, discord_state) 
        self.logger.info("Initialized StudentClubModel Instance")

        self.student_jobs_agent = StudentJobsModel(self.firestore, self.genai, self.app_config, logger, self.student_jobs_agent_tools, discord_state)
        self.logger.info("Initialized StudentJobsModel Instance")

        self.superior_agent_tools = Superior_Agent_Tools(
            firestore, discord_state, utils, app_config, self.live_status_agent, 
            self.discord_agent, self.courses_agent, self.events_agent, self.library_agent, self.news_agent, 
            self.scholarship_agent, self.social_media_agent, self.sports_agent, self.student_club_agent, 
            self.student_jobs_agent, logger, self.group_chat
        )
        
        self.superior_agent = SuperiorModel(self.firestore, self.genai, self.app_config, self.logger, self.superior_agent_tools)
        self.logger.info("Initialized ActionAgent Global Instance")

    async def process_question(self, question: str) -> str:
        return await self.superior_agent.determine_action(question)