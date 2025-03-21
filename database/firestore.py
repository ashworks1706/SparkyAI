from utils.common_imports import *
class Firestore:
    def __init__(self,discord_state):
        if not firebase_admin._apps:
            cred = credentials.Certificate("config/firebase_secret.json")
            firebase_admin.initialize_app(cred)
        self.db = firestore.client()
        self.collection = None
        self.document = {
            "superior_agent_message": [],
            "discord_agent_message": [],
            "google_agent_message": [],
            "shuttle_status_agent_message": [],
            "courses_agent_message": [],
            "student_club_events_agent_message": [],
            "library_agent_message": [],
            "news_agent_message": [],
            "sports_agent_message": [],
            "scholarship_agent_message": [],
            "social_media_agent_message": [],
            "student_club_agent_message": [],
            "student_jobs_agent_message": [],
            "user_id": "",
            "user_message": "",
            "time": "",
            "category": []
        }
        self.discord_state = discord_state

    def update_collection(self, collection):
        self.collection = collection
    
    def update_message(self, property, value): 
        if property in self.document:
            if isinstance(self.document[property], list):
                self.document[property].append(f"{value}")
            else:
                self.document[property] = f"{value}"
        else:
            raise ValueError(f"Invalid property: {property}")
    
    async def push_message(self):
        if not self.collection:
            raise ValueError("Collection not set. Use update_collection() first.")
        
        self.document["user_id"] = self.discord_state.get('user_id')
        self.document["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        doc_ref = self.db.collection(self.collection).document()
        doc_ref.set(self.document)
        
        self.check_and_add_user() 
        
        return doc_ref.id  # Return the document ID for reference

    async def check_and_add_user(self):
        user_id = self.discord_state.get('user_id')
        if not user_id:
            raise ValueError("User ID not found in discord_state.")
        
        users_collection = self.db.collection("users")
        query = users_collection.where("user_id", "==", user_id).get()
        
        if not query:  # If no document with the user_id exists
            new_user_doc = {
                "user_id": user_id,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "user_has_mod_role": self.discord_state.get('user_has_mod_role'),
                "user_in_voice_channel": self.discord_state.get('user_in_voice_channel'),
                "request_in_dm": self.discord_state.get('request_in_dm'),
                "guild_user": self.discord_state.get('guild_user'),
                "user_voice_channel_id": self.discord_state.get('user_voice_channel_id'),
            }
            users_collection.add(new_user_doc)
