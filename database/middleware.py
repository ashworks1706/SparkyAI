from utils.common_imports import *

class Middleware:
    # This class is used to manage the state of the Discord bot and its interactions, and interact with Firestore.
    def __init__(self):
        # Discord State initialization
        try:
            nest_asyncio.apply()
            self.intents = discord.Intents.default()
            self.intents.message_content = True
            self.intents.members = True
            self.user = False
            self.logged_in_sessions = {}
            self.target_guild = None
            self.user_id = None
            self.user_has_mod_role = False
            self.request_in_dm = False
            self.guild_user = None
            self.discord_client = discord.Client(intents=self.intents)
            self.task_message = None
            self.discord_post_channel_name = None
            self.discord_mod_role_name = None
            print("@middleware.py DiscordState initialized @ DiscordState")
        except Exception as e:
            print(f"@middleware.py Error initializing Discord state: {e}")

        # Firestore initialization
        try:
            if not firebase_admin._apps:
                print("@middleware.py Initializing Firebase app...")
                cred = credentials.Certificate("config/firebase_secret.json")
                firebase_admin.initialize_app(cred)
                print("@middleware.py Firebase app initialized.")

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

            print("@middleware.py Firestore initialized @ Firestore")
        except Exception as e:
            print(f"@middleware.py Error initializing Firestore: {e}")

    # Discord State methods
    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"Middleware has no attribute '{key}'")

    def get(self, attr):
        if hasattr(self, attr):
            return getattr(self, attr)
        else:
            raise AttributeError(f"Middleware has no attribute '{attr}'")

    def __str__(self):
        return "\n".join([f"{attr}: {getattr(self, attr)}" for attr in vars(self) if not attr.startswith('__')])

    # Firestore methods
    def update_collection(self, collection):
        collections = self.db.collections()
        collection_names = [col.id for col in collections]
        print(f"@middleware.py Existing collections: {collection_names}")

        print(f"@middleware.py Updating Firestore collection to: {collection}")
        if not collection in collection_names:
            raise ValueError(f"@middleware.py Collection '{collection}' does not exist.")

        self.collection = collection
        print(f"@middleware.py Firestore collection updated to: {self.collection}")

    def update_message(self, property, value):
        if property in self.document:
            if isinstance(self.document[property], list):
                print(f"@middleware.py Appending to {property}: {value}")
                self.document[property].append(f"{value}")
            else:
                self.document[property] = f"{value}"
                print(f"@middleware.py Setting {property} to: {value}")
        else:
            raise ValueError(f"@middleware.py Invalid property: {property}")

    async def check_user_session_timeout(self, user_id):
        try:
            users_collection = self.db.collection("users")
            query_results = users_collection.where("user_id", "==", user_id).get()

            user_docs = list(query_results)
            if not user_docs:
                print(f"@middleware.py User {user_id} not found in database")
                return False

            user_doc = user_docs[0]
            user_data = user_doc.to_dict()

            if "user_session_timeout" not in user_data:
                print(f"@middleware.py User {user_id} has no session timeout")
                return False

            timeout_str = user_data.get("user_session_timeout")
            if not timeout_str:
                print(f"@middleware.py User {user_id} has null session timeout value")
                return False

            timeout_time = datetime.strptime(timeout_str, "%Y-%m-%d %H:%M:%S")

            if datetime.now() > timeout_time:
                print(f"@middleware.py User {user_id} session timed out, removing credentials")
                users_collection.document(user_doc.id).update({
                    "user_session_timeout": None
                })
                del self.logged_in_sessions[user_id]
                return False

            print(f"@middleware.py User {user_id} session is valid until {timeout_str}")
            return True

        except ValueError as e:
            print(f"@middleware.py ValueError in check_user_session_timeout: {str(e)}")
            return False
        except Exception as e:
            print(f"@middleware.py Unexpected error in check_user_session_timeout: {str(e)}")
            return False

    async def push_message(self):
        if not self.collection:
            raise ValueError("@middleware.py Collection not set. Use update_collection() first.")

        print(f"@middleware.py Pushing message to Firestore collection: {self.collection}")
        self.document["user_id"] = self.get('user_id')
        self.document["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        doc_ref = self.db.collection(self.collection).document()
        doc_ref.set(self.document)

        print(f"@middleware.py Document written with ID: {doc_ref.id}")

        await self.check_and_add_user()
        await self.check_user_session_timeout(self.document["user_id"])

        return doc_ref.id

    async def check_and_add_user(self):
        user_id = self.get('user_id')
        if not user_id:
            raise ValueError("@middleware.py User ID not found in middleware.")

        users_collection = self.db.collection("users")
        query_results = users_collection.where("user_id", "==", user_id).get()
        print(f"Query result: {query_results}")

        if not list(query_results):
            new_user_doc = {
                "user_id": user_id,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "user_has_mod_role": self.get('user_has_mod_role'),
                "request_in_dm": self.get('request_in_dm'),
                "guild_user": str(self.get('guild_user')),
            }
            users_collection.add(new_user_doc)
            print(f"@middleware.py New user added to Firestore: {new_user_doc}")

    async def login_user_session_credentials(self, user_id, asurite_id, driver):
        try:
            users_collection = self.db.collection("users")
            query_results = users_collection.where("user_id", "==", user_id).get()

            user_docs = list(query_results)
            if not user_docs:
                new_user_doc = {
                    "user_id": user_id,
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "user_session_timeout": (datetime.now() + timedelta(hours=6)).strftime("%Y-%m-%d %H:%M:%S"),
                    "user_has_mod_role": self.get('user_has_mod_role'),
                    "request_in_dm": self.get('request_in_dm'),
                    "guild_user": str(self.get('guild_user')),
                    "asurite_id": asurite_id
                }
                try:
                    doc_ref = users_collection.add(new_user_doc)
                    # doc_id = doc_ref[1].id # this line is problematic
                    self.logged_in_sessions[user_id] = driver
                    print(f"@middleware.py New user credentials added to Firestore: {new_user_doc}")
                except Exception as e:
                    print(f"@middleware.py Error adding new user to Firestore: {e}")
                    return False
            else:
                doc = user_docs[0]
                user_data = doc.to_dict()

                if "user_session_timeout" in user_data:
                    timeout_str = user_data.get("user_session_timeout")
                    if timeout_str:
                        timeout_time = datetime.strptime(timeout_str, "%Y-%m-%d %H:%M:%S")
                        if datetime.now() < timeout_time:
                            print(f"@middleware.py User {user_id} already signed in, updating discord state only")
                            self.logged_in_sessions[user_id] = driver
                            return True

                try:
                    users_collection.document(doc.id).update({
                        "user_session_timeout": (datetime.now() + timedelta(hours=6)).strftime("%Y-%m-%d %H:%M:%S"),
                        "asurite_id": asurite_id
                    })
                    doc_id = doc.id
                    self.logged_in_sessions[user_id] = driver
                    print(f"@middleware.py User credentials updated in Firestore: {doc_id}")
                except Exception as e:
                    print(f"@middleware.py Error updating user credentials in Firestore: {e}")
                    return False
            return True
        except Exception as e:
            print(f"@middleware.py Unexpected error in login_user_session_credentials: {e}")
            return False

    async def logout_user_session_credentials(self, user_id):
        users_collection = self.db.collection("users")
        query_results = users_collection.where("user_id", "==", user_id).get()

        user_docs = list(query_results)
        if not user_docs:
            print(f"@middleware.py User {user_id} not found in Firestore")
            return False

        doc = user_docs[0]
        users_collection.document(doc.id).update({
            "user_session_timeout": None
        })
        
        del self.logged_in_sessions[user_id]


        print(f"@middleware.py User {user_id} logged out successfully")
        return True
