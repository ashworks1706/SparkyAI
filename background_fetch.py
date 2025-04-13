from utils.common_imports import *


class Background_Fetch:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.app_config = AppConfig()
        self.asu_data_processor = DataPreprocessor(self.app_config,genai=genai,logger=self.logger)
        self.vector_store = VectorStore(logger=self.logger, app_config=self.app_config, force_recreate=False)
        self.shuttles = Shuttles(self.vector_store)
        self.events = Events()
        self.news = News()
        self.clubs = Clubs()
        self.study_rooms = Study_Rooms()
        self.courses_catalog = Courses_Catalog()
        self.social_media_instagram = Social_Media_Instagram()
        self.social_media_x = Social_Media_X()
        self.social_media_facebook = Social_Media_Facebook()
        self.scholarships_goglobal = Scholarships_GoGlobal()
        self.scholarships_onsa = Scholarships_Onsa()
        self.library_catalog = Library_Catalog()
        self.shuttle_docs = []
        self.event_docs = []
        self.news_docs = []
        self.club_docs = []
        self.study_room_docs = []
        self.course_docs = []
        self.social_media_instagram_docs = []
        self.social_media_x_docs = []
        self.social_media_facebook_docs = [] 
        self.scholarships_goglobal_docs = []
        self.scholarships_onsa_docs = []
        self.library_catalog_docs = []
        self.library_ = []
        
        
        
    async def start_background_process(self):
        """
        Process the documents in the background.
        
        Example schema for self.shuttle_docs or any self.docs
        
        self..._docs = [
            {
                "documents": [
                    {
                        "id": "123",
                        "title": "Shuttle Schedule Update",
                        "content": "The shuttle schedule has been updated for the fall semester.",
                        "url": "https://example.com/shuttle-update",
                        "timestamp": "2023-10-01T10:00:00Z"
                    },
                    {
                        "id": "124",
                        "title": "New Shuttle Route",
                        "content": "A new shuttle route has been added to serve the west campus.",
                        "url": "https://example.com/new-shuttle-route",
                        "timestamp": "2023-10-02T12:00:00Z"
                    }
                ],
                "search_context": "shuttle information",
                "title": "Shuttle Updates"
            }
        ]

        
        """
        try:
            for doc_category, doc_list in {
                "shuttles_status": self.shuttle_docs,
                "events_info": self.event_docs,
                "news_info": self.news_docs,
                "clubs_info": self.club_docs,
                "study_rooms_status": self.study_room_docs,
                "courses_catalog": self.course_docs,
                "social_media_updates": self.social_media_instagram_docs,
                "social_media_updates": self.social_media_x_docs,
                "social_media_updates": self.social_media_facebook_docs,
                "scholarships_info": self.scholarships_goglobal_docs,
                "scholarships_info": self.scholarships_onsa_docs,
                "library_catalog": self.library_catalog_docs
            }.items():
                for doc in doc_list:
                    processed_docs= await self.asu_data_processor.process_documents(
                        documents=doc["documents"],
                        search_context=doc["search_context"],
                        title=doc["title"],
                        category=doc_category
                    )
                    self.vector_store.queue_documents(processed_docs)
            self.logger.info("Document processing completed successfully.")
            
            self.logger.info("Storing documents in vector store")

            await self.vector_store.store_to_vector_db()               
        except Exception as e:
            self.logger.error(f"@background_fetch.py Error during document processing: {str(e)}")
            raise ValueError("Failed to process documents")
    
    async def start_background_search(self):
        """
        Perform a web search in the background.
        """
        try:
            self.shuttle_docs = await self.shuttles.perform_web_search()            
            self.news_docs = await self.news.perform_web_search()
            self.club_docs = await self.clubs.perform_web_search()
            self.study_room_docs = await  self.study_rooms.perform_web_search()
            self.course_docs = await  self.courses_catalog.perform_web_search()
            self.social_media_instagram_docs = await  self.social_media_instagram.perform_web_search()
            self.social_media_x_docs = await  self.social_media_x.perform_web_search()
            self.social_media_facebook_docs = await  self.social_media_facebook.perform_web_search()
            self.scholarships_goglobal_docs = await  self.scholarships_goglobal.perform_web_search()
            self.scholarships_onsa_docs = await  self.scholarships_onsa.perform_web_search()
            self.library_docs = await  self.library_catalog.perform_web_search()
            self.event_docs = await self.events.perform_web_search()
            await self.logger.info("Web search completed successfully.")
            
        except Exception as e:
            self.logger.error(f"@background_fetch.py Error during web search: {str(e)}")
            raise ValueError("Failed to perform web search")
        

background_fetch = Background_Fetch()

async def main():
    await background_fetch.start_background_search()
    await background_fetch.start_background_process()

# Run the main function every 24 hours
if __name__ == "__main__":
    import asyncio
    import time
    
    async def scheduled_task():
        while True:
            try:
                await main()
                # Sleep for 24 hours (86400 seconds)
                await asyncio.sleep(86400)
            except Exception as e:
                print(f"Error in scheduled task: {e}")
                # If there's an error, wait 1 hour before retrying
                await asyncio.sleep(3600)
    
    asyncio.run(scheduled_task())

