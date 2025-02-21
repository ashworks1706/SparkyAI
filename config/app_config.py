class AppConfig:
    def __init__(self, config_file='appConfig.json'):
        with open(config_file, 'r') as file:
            config_data = json.load(file)
        
        os.environ['NUMEXPR_MAX_THREADS'] = config_data.get('NUMEXPR_MAX_THREADS', '16')
        os.environ['HUGGINGFACEHUB_API_TOKEN'] = config_data.get('HUGGINGFACEHUB_API_TOKEN', '')
        self._api_key = config_data.get('API_KEY', '')
        self.handshake_user = config_data.get('HANDSHAKE_USER', '')
        self.handshake_pass = config_data.get('HANDSHAKE_PASS', '')
        self.gmail = config_data.get('GMAIL', '')
        self.gmail_pass = config_data.get('GMAIL_PASS', '')
        self.spreadsheet_id = config_data.get('SPREADSHEET_ID', '')
        
        self.main_agent_prompt = config_data.get('MAIN_AGENT_PROMPT', '')
        
        self.live_status_agent_prompt = config_data.get('LIVE_STATUS_AGENT_PROMPT', '')
        self.live_status_agent_instruction = config_data.get('LIVE_STATUS_AGENT_INSTRUCTION', '')
        
        self.discord_agent_prompt = config_data.get('DISCORD_AGENT_PROMPT', '')
        self.discord_agent_instruction = config_data.get('DISCORD_AGENT_INSTRUCTION', '')
        
        self.search_agent_prompt = config_data.get('SEARCH_AGENT_PROMPT', '')
        self.search_agent_instruction = config_data.get('SEARCH_AGENT_INSTRUCTION', '')
        
        self.action_agent_prompt = config_data.get('ACTION_AGENT_PROMPT', '')
        self.action_agent_instruction = config_data.get('ACTION_AGENT_INSTRUCTION', '')
        
        
        self.search_agent_prompt = config_data.get('SEARCH_AGENT_PROMPT', '')
        self.search_agent_instruction = config_data.get('SEARCH_AGENT_INSTRUCTION', '')
        
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
    
    def get_search_agent_prompt(self):
        return self.search_agent_prompt
    def get_search_agent_instruction(self):
        return self.search_agent_instruction
    
    
    def get_action_agent_prompt(self):
        return self.action_agent_prompt
    def get_action_agent_instruction(self):
        return self.action_agent_instruction
    
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
    def get_spreadsheet_id(self):
        return self.spreadsheet_id
    def get_gmail_pass(self):
        return self.gmail_pass
