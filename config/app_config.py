from utils.common_imports import *
class AppConfig:
    def __init__(self, config_file='config/appConfig.json'):
        with open(config_file, 'r') as file:
            config_data = json.load(file)
        
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
        
        self.rag_search_agent_prompt = config_data.get('RAG_SEARCH_AGENT_PROMPT', '')
        self.rag_search_agent_instruction = config_data.get('RAG_SEARCH_AGENT_INSTRUCTION', '')
        
        self.superior_agent_prompt = config_data.get('superior_agent_PROMPT', '')
        self.superior_agent_instruction = config_data.get('superior_agent_INSTRUCTION', '')
        
        
        self.rag_search_agent_prompt = config_data.get('RAG_SEARCH_AGENT_PROMPT', '')
        self.rag_search_agent_instruction = config_data.get('RAG_SEARCH_AGENT_INSTRUCTION', '')
        
        self.discord_agent_prompt = config_data.get('DISCORD_AGENT_PROMPT', '')
        self.discord_agent_instruction = config_data.get('DISCORD_AGENT_INSTRUCTION', '')
        
        self.google_agent_prompt = config_data.get('GOOGLE_AGENT_PROMPT', '')
        self.google_agent_instruction = config_data.get('GOOGLE_AGENT_INSTRUCTION', '')
        self.data_agent_prompt = config_data.get('DATA_AGENT_PROMPT', '')
        self.discord_bot_token = config_data.get('DISCORD_BOT_TOKEN', '')
        self.kubernetes_api_key = config_data.get('KUBERNETES_SECRET', '')
        self.qdrant_api_key = config_data.get('QDRANT_API_KEY', '')

    def get_numexpr_max_threads(self):
        return os.environ['NUMEXPR_MAX_THREADS']

    def get_huggingfacehub_api_token(self):
        return os.environ['HUGGINGFACEHUB_API_TOKEN']

    def get_qdrant_api_key(self):
        return self.qdrant_api_key
    
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
    
    def get_rag_search_agent_prompt(self):
        return self.rag_search_agent_prompt
    def get_rag_search_agent_instruction(self):
        return self.rag_search_agent_instruction
    
    
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
