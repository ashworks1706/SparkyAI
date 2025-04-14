from datetime import timedelta
from utils.common_imports import *
class Firestore:
    
    
    # This class is responsible for interacting with Firestore database.
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
            "sports_agent_message": [],
            "scholarship_agent_message": [],
            "news_media_agent_message": [],
            "student_club_agent_message": [],
            "student_jobs_agent_message": [],
            "user_id": "",
            "user_message": "",
            "time": "",
            "category": []
        }
        
        self.discord_state = discord_state
        
        print("@firestore.py Firestore initialized @ Firestore")
    
    
    
    def update_collection(self, collection):
        # get existing collections
        collections = self.db.collections()
        collection_names = [col.id for col in collections]
        print(f"@firestore.py Existing collections: {collection_names}")
        
        print(f"@firestore.py Updating Firestore collection to: {collection}")
        if not collection in collection_names:
            raise ValueError(f"@firestore.py Collection '{collection}' does not exist.")
        
        self.collection = collection
        
        print(f"@firestore.py Firestore collection updated to: {self.collection}")
        
        
    
    
    def update_message(self, property, value): 
        if property in self.document:
            if isinstance(self.document[property], list):
                print(f"@firestore.py Appending to {property}: {value}")
                self.document[property].append(f"{value}")
            else:
                self.document[property] = f"{value}"
                print(f"@firestore.py Setting {property} to: {value}")
        else:
            raise ValueError(f"@firestore.py Invalid property: {property}")
        
        
        
    async def check_user_session_timeout(self, user_id):
            """
            Check if a user's session has timed out and delete credentials if expired
            
            Args:
                user_id (str): Discord user ID to check
                
            Returns:
                bool: True if session is valid, False if timed out or user not found
            """
            users_collection = self.db.collection("users")
            query_results = users_collection.where("user_id", "==", user_id).get()
            
            user_docs = list(query_results)
            if not user_docs:
                print(f"@firestore.py User {user_id} not found in database")
                return False
                
            user_doc = user_docs[0]
            user_data = user_doc.to_dict()
            
            # Check if user has session timeout field
            if "user_session_timeout" not in user_data:
                print(f"@firestore.py User {user_id} has no session timeout")
                return False
                
            # Parse timeout and check against current time
            timeout_str = user_data.get("user_session_timeout")
            timeout_time = datetime.strptime(timeout_str, "%Y-%m-%d %H:%M:%S")
            
            if datetime.now() > timeout_time:
                # Session timed out, remove credentials
                print(f"@firestore.py User {user_id} session timed out, removing credentials")
                users_collection.document(user_doc.id).update({
                    "user_session_timeout": None
                })
                self.discord_state.update(user_session_id= None)                
                return False
            
            return True
            
    # This method is used to push the message to Firestore.
    async def push_message(self):
        
        if not self.collection:
            raise ValueError("@firestore.py Collection not set. Use update_collection() first.")
        
        print(f"@firestore.py Pushing message to Firestore collection: {self.collection}")
        self.document["user_id"] = self.discord_state.get('user_id')
        
        self.document["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        doc_ref = self.db.collection(self.collection).document()
        
        doc_ref.set(self.document)
        
        print(f"@firestore.py Document written with ID: {doc_ref.id}")
        
        await self.check_and_add_user() 
        
        await self.check_user_session_timeout(self.document["user_id"])
        
        return doc_ref.id  # Return the document ID for reference
    
    # This method is used to check if the user exists in Firestore and add them if not.
    async def check_and_add_user(self):
        # Check if the user already exists in the Firestore database
        user_id = self.discord_state.get('user_id')
        if not user_id:
            raise ValueError("@firestore.py User ID not found in discord_state.")
        
        users_collection = self.db.collection("users")
        # Use the where method for filtering in Firestore
        query_results = users_collection.where("user_id", "==", user_id).get()
        print(f"Query result: {query_results}")
        
        # Check if the user already exists in the collection
        if not list(query_results):  # Convert to list to check if empty
            new_user_doc = {
                "user_id": user_id,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "user_has_mod_role": self.discord_state.get('user_has_mod_role'),
                "request_in_dm": self.discord_state.get('request_in_dm'),
                "guild_user": str(self.discord_state.get('guild_user')),
            }
            users_collection.add(new_user_doc)
            print(f"@firestore.py New user added to Firestore: {new_user_doc}")
    
    
    
    async def login_user_credentials(self, user_id, asurite_id, password):
        # Check if the user already exists in the Firestore database
        users_collection = self.db.collection("users")
        query_results = users_collection.where("user_id", "==", user_id).get()
        
        user_docs = list(query_results)
        if not user_docs:  # User doesn't exist
            new_user_doc = {
            "user_id": user_id,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user_session_timeout": (datetime.now() + timedelta(hours=6)).strftime("%Y-%m-%d %H:%M:%S"),
            "user_has_mod_role": self.discord_state.get('user_has_mod_role'),
            "request_in_dm": self.discord_state.get('request_in_dm'),
            "guild_user": str(self.discord_state.get('guild_user')),
            "asurite_id": asurite_id
            }
            doc_ref = users_collection.add(new_user_doc)
            doc_id = doc_ref[1].id
            print(f"@firestore.py New user credentials added to Firestore: {new_user_doc}")
        else:
            doc = user_docs[0]
            user_data = doc.to_dict()
            
            # Check if user is already signed in (has valid session)
            if "user_session_timeout" in user_data:
                timeout_str = user_data.get("user_session_timeout")
                if timeout_str:
                    timeout_time = datetime.strptime(timeout_str, "%Y-%m-%d %H:%M:%S")
                    if datetime.now() < timeout_time:
                        # User is already signed in, just update discord state
                        print(f"@firestore.py User {user_id} already signed in, updating discord state only")
                        doc_id = doc.id
                        asurite_id = user_data.get("asurite_id", asurite_id)
                        self.discord_state.update(user_session_id=doc_id)
                        self.discord_state.update(user_asu_rite=asurite_id)
                        self.discord_state.update(user_password=password)
                        return
            
            # Otherwise update the session timeout
            users_collection.document(doc.id).update({
                "user_session_timeout": (datetime.now() + timedelta(hours=6)).strftime("%Y-%m-%d %H:%M:%S"),
                "asurite_id": asurite_id
            })
            doc_id = doc.id
            print(f"@firestore.py User credentials updated in Firestore: {doc_id}")
        
        self.discord_state.update(user_session_id=doc_id)    
        self.discord_state.update(user_asu_rite=asurite_id)    
        self.discord_state.update(user_password=password)    

    async def logout_user_credentials(self, user_id):
        # Check if the user already exists in the Firestore database
        users_collection = self.db.collection("users")
        query_results = users_collection.where("user_id", "==", user_id).get()
        
        # Check if the user already exists in the collection
        if not list(query_results):
            print(f"@firestore.py User {user_id} not found in Firestore")
            return False