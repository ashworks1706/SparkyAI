from utils.common_imports import *

class Courses_Agent_Tools:
    def __init__(self,firestore,utils,logger):
        self.firestore = firestore
        self.utils = utils
        self.visited_urls = set()
        self.max_depth = 2
        self.max_links_per_page = 3
        
   
    async def get_latest_class_information(self,search_bar_query: Optional[str] = None,class_term: Optional[str] = None,subject_name: Optional[Union[str, List[str]]] = None, 
    num_of_credit_units: Optional[Union[str, List[str]]] = None, 
    class_level: Optional[Union[str, List[str]]] = None,
    class_session: Optional[Union[str, List[str]]] = None,
    class_days: Optional[Union[str, List[str]]] = None,
    class_location: Optional[Union[str, List[str]]] = None,
    class_seat_availability : Optional[str] = None,
    ) -> str:
        """
        Optimized function to generate a search URL for ASU class catalog with flexible input handling.
        
        Args:
            Multiple optional parameters for filtering class search
        
        Returns:
            Constructed search URL for class catalog
        """
        
        # Helper function to convert input to query string
        
        
        
        DAYS_MAP = {
            'Monday': 'MON',
            'Tuesday': 'TUES', 
            'Wednesday': 'WED', 
            'Thursday': 'THURS', 
            'Friday': 'FRI', 
            'Saturday': 'SAT', 
            'Sunday': 'SUN'
        }
        
        
        CLASS_LEVEL_MAP = {
        'Lower division': 'lowerdivision',
        'Upper division': 'upperdivision', 
        'Undergraduate': 'undergrad',
        'Graduate': 'grad',
        '100-199': '100-199',
        '200-299': '200-299',
        '300-399': '300-399',
        '400-499': '400-499'
        }
        
        SESSION_MAP = {
            'A': 'A',
            'B': 'B', 
            'C': 'C',
            'Other': 'DYN'
        }
        
       

        TERM_MAP= {
            'Spring 2025': '2251',
            'Fall 2024': '2247', 
            'Summer 2024': '2244',
            'Spring 2024': '2241',
            'Fall 2023': '2237', 
            'Summer 2023': '2234'
        }
        
        CREDIT_UNITS_MAP = {
            '0': 'Less than 1',
            '1': '1',
            '2': '2',
            '3': '3',
            '4': '4',
            '5': '5',
            '6': '6',
            '7': '7 or more'
        }


        
        unmapped_items = []
        
        def _convert_to_query_string(input_value: Optional[Union[str, List[str]]], mapping: Dict[str, str]) -> str:
            global unmapped_items
            unmapped_items = []
            
            # Handle None input
            if input_value is None:
                return ''
            
            # Ensure input is a list
            if isinstance(input_value, str):
                input_value = [input_value]
            
            # Process each input value
            mapped_values = []
            for value in input_value:
                # Check if value exists in mapping
                if value in mapping:
                    mapped_values.append(mapping[value])
                else:
                    # Add unmapped items to global list
                    unmapped_items.append(value)
            
            # Join mapped values with URL-encoded comma
            return '%2C'.join(mapped_values) if mapped_values else ''
        
        
        
        search_bar_query = (search_bar_query or '') + ' ' + ' '.join(unmapped_items)
        search_bar_query+=subject_name
        search_bar_query = search_bar_query.strip().replace(" ", "%20")
        
        
        params = {
            'advanced': 'true',
            'campus': _convert_to_query_string(class_location, LOCATION_MAP),
            'campusOrOnlineSelection': 'A',
            'daysOfWeek': _convert_to_query_string(class_days, DAYS_MAP),
            'honors': 'F',
            'keywords': search_bar_query,
            'level': _convert_to_query_string(class_level, CLASS_LEVEL_MAP),
            'promod': 'F',
            'searchType': "open" if class_seat_availability == "Open" else "all",
            'session': _convert_to_query_string(class_session, SESSION_MAP),
            'term': _convert_to_query_string(class_term, TERM_MAP),
            'units': _convert_to_query_string(num_of_credit_units, CREDIT_UNITS_MAP)
        }
        
        self.logger.info(params)

        # Remove None values and construct URL
        search_url = 'https://catalog.apps.asu.edu/catalog/classes/classlist?' + '&'.join(
            f'{key}={value}' 
            for key, value in params.items() 
            if value is not None and value != ''
        )
        
        doc_title=""
        if search_bar_query:
            doc_title = search_bar_query
        elif subject_name:
            doc_title = " ".join(subject_name) if isinstance(subject_name, list) else subject_name
        elif class_term:
            doc_title = class_term
        elif class_level:
            doc_title = " ".join(class_level) if isinstance(class_level, list) else class_level
        elif class_location:
            doc_title = " ".join(class_location) if isinstance(class_location, list) else class_location
        elif class_session:
            doc_title = " ".join(class_session) if isinstance(class_session, list) else class_session
        elif num_of_credit_units:
            doc_title = " ".join(num_of_credit_units) if isinstance(num_of_credit_units, list) else num_of_credit_units
        elif class_days:
            doc_title = " ".join(class_days) if isinstance(class_days, list) else class_days

        elif class_seat_availability:
            doc_title = class_seat_availability
        else:
            doc_title = None

        return await self.utils.perform_web_search(search_url,doc_title=doc_title, doc_category ="classes_info")