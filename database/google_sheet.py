class UpdateTask:
    def __init__(self, user_id: str, column: str, value: Any):
        self.user_id = user_id
        self.column = column
        self.value = value

class GoogleSheet:
    def __init__(self, credentials_file: str, spreadsheet_id: str) -> None:
        self.credentials = Credentials.from_service_account_file(
            credentials_file,
            scopes=['https://www.googleapis.com/auth/spreadsheets']
        )
        self.spreadsheet_id = spreadsheet_id
        self.service = build('sheets', 'v4', credentials=self.credentials, cache_discovery=False)
        self.sheet = self.service.spreadsheets()
        self.logger = logging.getLogger(__name__)
        self.update_tasks: List[UpdateTask] = []
        self.user_row_cache: Dict[str, int] = {}

    async def get_all_users(self, range_name: str = 'SparkyVerify!A:C'):
        try:
            result = self.sheet.values().get(
                spreadsheetId=self.spreadsheet_id,
                range=range_name
            ).execute()
            return result.get('values', [])
        except Exception as e:
            self.logger.error(f"Error getting all users: {str(e)}")
            return []

    async def increment_function_call(self, user_id: str, column: str):
        self.update_tasks.append(UpdateTask(user_id, column, None))

    async def update_user_column(self, user_id: str, column: str, value):
        self.update_tasks.append(UpdateTask(user_id, column, value))

    async def perform_updates(self):
        try:
            batch_update_data = []
            for task in self.update_tasks:
                row = await self.get_user_row(task.user_id)
                if row:
                    range_name = f'SparkyVerify!{task.column}{row}'
                    if task.value is None:  # Increment function call
                        current_value = await self.get_cell_value(range_name)
                        new_value = int(current_value) + 1 if current_value else 1
                    else:
                        new_value = task.value
                    batch_update_data.append({
                        'range': range_name,
                        'values': [[new_value]]
                    })
                else:
                    self.logger.warning(f"User {task.user_id} not found for updating")

            if batch_update_data:
                body = {
                    'valueInputOption': 'USER_ENTERED',
                    'data': batch_update_data
                }
                self.sheet.values().batchUpdate(
                    spreadsheetId=self.spreadsheet_id,
                    body=body
                ).execute()
                self.logger.info(f"Batch update completed for {len(batch_update_data)} tasks")
            
            self.update_tasks.clear()
        except Exception as e:
            self.logger.error(f"Error performing batch updates: {str(e)}")

    async def get_user_row(self, user_id: str):
        if user_id in self.user_row_cache:
            return self.user_row_cache[user_id]

        try:
            range_name = 'SparkyVerify!A:A'
            result = self.sheet.values().get(
                spreadsheetId=self.spreadsheet_id,
                range=range_name
            ).execute()
            rows = result.get('values', [])
            for idx, row in enumerate(rows):
                if row and row[0] == str(user_id):
                    row_number = idx + 1
                    self.user_row_cache[user_id] = row_number
                    return row_number
            return None
        except Exception as e:
            self.logger.error(f"Error finding user row: {str(e)}")
            return None

    async def get_cell_value(self, range_name: str):
        try:
            result = self.sheet.values().get(
                spreadsheetId=self.spreadsheet_id,
                range=range_name
            ).execute()
            return result.get('values', [[0]])[0][0]
        except Exception as e:
            self.logger.error(f"Error getting cell value: {str(e)}")
            return 0

    async def add_new_user(self, user, email):
        user_data = [str(user.id), user.name, email, datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        existing_row = await self.get_user_row(user.id)
        
        try:
            if existing_row:
                range_name = f'SparkyVerify!A{existing_row}:Z{existing_row}'
                body = {'values': [user_data]}
                self.sheet.values().update(
                    spreadsheetId=self.spreadsheet_id,
                    range=range_name,
                    valueInputOption='USER_ENTERED',
                    body=body
                ).execute()
                self.logger.info(f"Updated existing user: {user.id}")
            else:
                range_name = 'SparkyVerify!A:Z'
                body = {'values': [user_data]}
                self.sheet.values().append(
                    spreadsheetId=self.spreadsheet_id,
                    range=range_name,
                    valueInputOption='USER_ENTERED',
                    body=body
                ).execute()
                self.logger.info(f"Added new user: {user.id}")
            
            self.user_row_cache[user.id] = await self.get_user_row(user.id)
        except Exception as e:
            self.logger.error(f"Error adding/updating user: {str(e)}")
