from utils.common_imports import *
class Firestore:
    def __init__(self,discord_state):
        if not firebase_admin._apps:
            cred = credentials.Certificate("firebase_secret.json")
            firebase_admin.initialize_app(cred)
        self.db = firestore.client()
        self.collection = None
        self.document = {
            "superior_agent_message": [],
            "discord_agent_message": [],
            "google_agent_message": [],
            "live_status_agent_message": [],
            "rag_search_agent_message": [],
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
        
        return doc_ref.id  # Return the document ID for reference
