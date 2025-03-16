from utils.common_imports import *

class AppConfig:
    def __init__(self, config_file='config/appConfig.json'):
        try:
            with open(config_file, 'r') as file:
                config_data = json.load(file)
        except FileNotFoundError:
            print(f"Error: Config file '{config_file}' not found.")
            config_data = {}
        except json.JSONDecodeError:
            print(f"Error: Config file '{config_file}' contains invalid JSON.")
            config_data = {}
        
        os.environ['NUMEXPR_MAX_THREADS'] = config_data.get('NUMEXPR_MAX_THREADS', '16')
        os.environ['HUGGINGFACEHUB_API_TOKEN'] = config_data.get('HUGGINGFACEHUB_API_TOKEN', '')
        self._api_key = config_data.get('API_KEY', '')
        self.handshake_user = config_data.get('HANDSHAKE_USER', '')
        self.handshake_pass = config_data.get('HANDSHAKE_PASS', '')
        self.gmail = config_data.get('GMAIL', '')
        self.gmail_pass = config_data.get('GMAIL_PASS', '')
        self.main_agent_prompt = config_data.get('MAIN_AGENT_PROMPT', '')
        self.live_status_agent_prompt = config_data.get('LIVE_STATUS_AGENT_PROMPT', '')
        self.live_status_agent_instruction = config_data.get('LIVE_STATUS_AGENT_INSTRUCTION', '')
        self.discord_agent_prompt = config_data.get('DISCORD_AGENT_PROMPT', '')
        self.discord_agent_instruction = config_data.get('DISCORD_AGENT_INSTRUCTION', '')
        # self.rag_search_agent_prompt = config_data.get('RAG_SEARCH_AGENT_PROMPT', '')
        # self.rag_search_agent_instruction = config_data.get('RAG_SEARCH_AGENT_INSTRUCTION', '')
        self.superior_agent_prompt = config_data.get('SUPERIOR_AGENT_PROMPT', '')
        self.superior_agent_instruction = config_data.get('SUPERIOR_AGENT_INSTRUCTION', '')
        
        
        self.discord_agent_prompt = config_data.get('DISCORD_AGENT_PROMPT', '')
        self.discord_agent_instruction = config_data.get('DISCORD_AGENT_INSTRUCTION', '')
        
        self.google_agent_prompt = config_data.get('GOOGLE_AGENT_PROMPT', '')
        self.google_agent_instruction = config_data.get('GOOGLE_AGENT_INSTRUCTION', '')
        self.data_agent_prompt = config_data.get('DATA_AGENT_PROMPT', '')
        self.discord_bot_token = config_data.get('DISCORD_BOT_TOKEN', '')
        self.kubernetes_api_key = config_data.get('KUBERNETES_SECRET', '')
        self.qdrant_api_key = config_data.get('QDRANT_API_KEY', '')
        self.discord_target_guild_id = config_data.get('TARGET_GUILD_ID', '')
        self.discord_allowed_chat_id = config_data.get('ALLOWED_CHAT_ID', '')
        self.discord_verification_id = config_data.get('VERIFY_CHANNEL_ID', '')
        self.discord_feedback_id = config_data.get('FEEDBACK_CHANNEL_ID', '')
        self.discord_mod_role_name = config_data.get('DISCORD_MOD_ROLE_NAME', '')
        self.discord_post_channel_name = config_data.get('DISCORD_POST_CHANNEL_NAME', '')
        self.student_club_agent_prompt = config_data.get('STUDENT_CLUB_AGENT_PROMPT', '')
        self.student_club_agent_instruction = config_data.get('STUDENT_CLUB_AGENT_INSTRUCTION', '')
        self.events_agent_prompt = config_data.get('EVENTS_AGENT_PROMPT', '')
        self.events_agent_instruction = config_data.get('EVENTS_AGENT_INSTRUCTION', '')
        self.news_agent_prompt = config_data.get('NEWS_AGENT_PROMPT', '')
        self.news_agent_instruction = config_data.get('NEWS_AGENT_INSTRUCTION', '')
        self.sports_agent_prompt = config_data.get('SPORTS_AGENT_PROMPT', '')
        self.sports_agent_instruction = config_data.get('SPORTS_AGENT_INSTRUCTION', '')
        self.social_media_agent_prompt = config_data.get('SOCIAL_MEDIA_AGENT_PROMPT', '')
        self.social_media_agent_instruction = config_data.get('SOCIAL_MEDIA_AGENT_INSTRUCTION', '')
        self.library_agent_prompt = config_data.get('LIBRARY_AGENT_PROMPT', '')
        self.library_agent_instruction = config_data.get('LIBRARY_AGENT_INSTRUCTION', '')
        self.scholarship_agent_prompt = config_data.get('SCHOLARSHIP_AGENT_PROMPT', '')
        self.scholarship_agent_instruction = config_data.get('SCHOLARSHIP_AGENT_INSTRUCTION', '')
        self.student_jobs_agent_prompt = config_data.get('STUDENT_JOBS_AGENT_PROMPT', '')
        self.student_jobs_agent_instruction = config_data.get('STUDENT_JOBS_AGENT_INSTRUCTION', '')
        self.courses_agent_prompt = config_data.get('COURSES_AGENT_PROMPT', '')
        self.courses_agent_instruction = config_data.get('COURSES_AGENT_INSTRUCTION', '')

    def get_numexpr_max_threads(self):
        return os.environ['NUMEXPR_MAX_THREADS']
    def get_huggingfacehub_api_token(self):
        return os.environ['HUGGINGFACEHUB_API_TOKEN']
    def get_qdrant_api_key(self):
        return self.qdrant_api_key
    def get_discord_target_guild_id(self):
        return int(self.discord_target_guild_id)
    def get_discord_mod_role_name(self):
        return self.discord_mod_role_name
    def get_discord_post_channel_name(self):
        return self.discord_post_channel_name
    def get_discord_allowed_chat_id(self):
        return int(self.discord_allowed_chat_id)
    def get_discord_verification_id(self):
        return int(self.discord_verification_id)
    def get_discord_feedback_id(self):
        return int(self.discord_feedback_id)
    def get_discord_bot_token(self):
        return self.discord_bot_token
    def get_kubernetes_api_key(self):
        return self.kubernetes_api_key
    def get_data_agent_prompt(self):
        return self.data_agent_prompt
    def get_live_status_agent_prompt(self):
        return self.live_status_agent_prompt
    def get_live_status_agent_instruction(self):
        return self.live_status_agent_instruction
    def get_discord_agent_prompt(self):
        return self.discord_agent_prompt
    def get_discord_agent_instruction(self):
        return self.discord_agent_instruction
    # def get_rag_search_agent_prompt(self):
    #     return self.rag_search_agent_prompt
    # def get_rag_search_agent_instruction(self):
    #     return self.rag_search_agent_instruction
    def get_superior_agent_prompt(self):
        return self.superior_agent_prompt
    def get_superior_agent_instruction(self):
        return self.superior_agent_instruction
    def get_google_agent_prompt(self):
        return self.google_agent_prompt
    def get_google_agent_instruction(self):
        return self.google_agent_instruction
    def get_api_key(self):
        return self._api_key
    def get_handshake_user(self):
        return self.handshake_user
    def get_handshake_pass(self):
        return self.handshake_pass
    def get_gmail(self):
        return self.gmail
    def get_gmail_pass(self):
        return self.gmail_pass
    
    def get_student_club_agent_prompt(self):
        return self.student_club_agent_prompt
    def get_student_club_agent_instruction(self):
        return self.student_club_agent_instruction
    
    def get_events_agent_prompt(self):
        return self.events_agent_prompt
    def get_events_agent_instruction(self):
        return self.events_agent_instruction
    
    def get_news_agent_prompt(self):
        return self.news_agent_prompt
    def get_news_agent_instruction(self):
        return self.news_agent_instruction
    
    def get_sports_agent_prompt(self):
        return self.sports_agent_prompt
    def get_sports_agent_instruction(self):
        return self.sports_agent_instruction
    
    def get_social_media_agent_prompt(self):
        return self.social_media_agent_prompt
    def get_social_media_agent_instruction(self):
        return self.social_media_agent_instruction
    
    def get_library_agent_prompt(self):
        return self.library_agent_prompt
    def get_library_agent_instruction(self):
        return self.library_agent_instruction
    
    def get_scholarship_agent_prompt(self):
        return self.scholarship_agent_prompt
    def get_scholarship_agent_instruction(self):
        return self.scholarship_agent_instruction
    
    def get_student_jobs_agent_prompt(self):
        return self.student_jobs_agent_prompt
    def get_student_jobs_agent_instruction(self):
        return self.student_jobs_agent_instruction
    
    def get_courses_agent_prompt(self):
        return self.courses_agent_prompt
    def get_courses_agent_instruction(self):
        return self.courses_agent_instruction
