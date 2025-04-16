from utils.common_imports import *
from agents.shuttle_status_agent import Shuttle_Status_Model
from agents.discord_agent import DiscordModel
from agents.superior_agent import SuperiorModel
from agents.courses_agent import CoursesModel
from agents.library_agent import LibraryModel
from agents.news_media_agent import NewsMediaModel
from agents.scholarship_agent import ScholarshipModel
from agents.sports_agent import SportsModel
from agents.student_clubs_events_agent import StudentClubsEventsModel
from agents.student_jobs_agent import StudentJobsModel

from agent_tools.discord_agent_tools import Discord_Agent_Tools
from agent_tools.shuttle_status_agent_tools import Shuttle_Status_Agent_Tools
from agent_tools.superior_agent_tools import Superior_Agent_Tools
from agent_tools.courses_agent_tools import Courses_Agent_Tools
from agent_tools.library_agent_tools import Library_Agent_Tools
from agent_tools.news_media_agent_tools import News_Media_Agent_Tools
from agent_tools.scholarship_agent_tools import Scholarship_Agent_Tools
from agent_tools.sports_agent_tools import Sports_Agent_Tools
from agent_tools.student_clubs_events_tools import Student_Clubs_Events_Agent_Tools
from agent_tools.student_jobs_agent_tools import Student_Jobs_Agent_Tools

class Agents:
    def __init__(self, vector_store, asu_data_processor, middleware, genai, utils, app_config, logger, group_chat):
        self.vector_store = vector_store
        self.asu_data_processor = asu_data_processor
        self.middleware = middleware
        self.genai = genai
        self.utils = utils
        self.app_config = app_config
        self.logger = logger 
        self.group_chat = group_chat

        self.discord_agent_tools = Discord_Agent_Tools(middleware, utils, logger)
        self.shuttle_status_agent_tools = Shuttle_Status_Agent_Tools(middleware, utils, logger)
        self.courses_agent_tools = Courses_Agent_Tools(middleware, utils, logger)
        self.library_agent_tools = Library_Agent_Tools(middleware, utils, logger)
        self.scholarship_agent_tools = Scholarship_Agent_Tools(middleware, utils, logger)
        self.news_media_agent_tools = News_Media_Agent_Tools(middleware, utils, logger)
        self.sports_agent_tools = Sports_Agent_Tools(middleware, utils, logger)
        self.student_clubs_events_agent_tools = Student_Clubs_Events_Agent_Tools(middleware, utils, logger)
        self.student_jobs_agent_tools = Student_Jobs_Agent_Tools(middleware, utils, logger)

        self.logger.info(f"@agents.py Initialized Agent Tools")


        self.shuttle_status_agent = Shuttle_Status_Model(self.middleware, self.genai, self.app_config, logger, self.shuttle_status_agent_tools)
        self.logger.info(f"@agents.py Initialized ShuttleStatusAgent")


        self.discord_agent = DiscordModel(self.middleware, self.genai, self.app_config, logger, self.discord_agent_tools)
        self.logger.info(f"@agents.py Initialized Discord Model Instance")

        self.courses_agent = CoursesModel(self.middleware, self.genai, self.app_config, logger, self.courses_agent_tools)
        self.logger.info(f"@agents.py Initialized CoursesModel Instance")

        self.library_agent = LibraryModel(self.middleware, self.genai, self.app_config, logger, self.library_agent_tools)
        self.logger.info(f"@agents.py Initialized LibraryModel Instance")

        self.news_media_agent = NewsMediaModel(self.middleware, self.genai, self.app_config, logger, self.news_media_agent_tools)
        self.logger.info(f"@agents.py Initialized NewsMediaModel Instance")

        self.scholarship_agent = ScholarshipModel(self.middleware, self.genai, self.app_config, logger, self.scholarship_agent_tools)
        self.logger.info(f"@agents.py Initialized ScholarshipModel Instance")


        self.sports_agent = SportsModel(self.middleware, self.genai, self.app_config, logger, self.sports_agent_tools)
        self.logger.info(f"@agents.py Initialized SportsModel Instance")

        self.student_clubs_events_agent = StudentClubsEventsModel(self.middleware, self.genai, self.app_config, logger, self.student_clubs_events_agent_tools) 
        self.logger.info(f"@agents.py Initialized StudentClubsEventsModel Instance")

        self.student_jobs_agent = StudentJobsModel(self.middleware, self.genai, self.app_config, logger, self.student_jobs_agent_tools)
        self.logger.info(f"@agents.py Initialized StudentJobsModel Instance")

        self.superior_agent_tools = Superior_Agent_Tools(self.vector_store, self.asu_data_processor,middleware, utils, app_config, self.shuttle_status_agent,self.discord_agent, self.courses_agent, self.library_agent, self.news_media_agent,self.scholarship_agent, self.sports_agent, self.student_clubs_events_agent,self.student_jobs_agent, logger, self.group_chat
        )
        
        self.superior_agent = SuperiorModel(self.middleware, self.genai, self.app_config, self.logger, self.superior_agent_tools)
        self.logger.info(f"@agents.py Initialized ActionAgent Global Instance")

    async def process_question(self, question: str) -> str:
        return await self.superior_agent.determine_action(question)