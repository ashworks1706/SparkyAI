class GroupChat:
    def __init__(self, text):
        self.text = text
    def update_text(self, text):
        self.text = text
    def get_text(self):
        return self.text
    
# TODO: Implement the GroupChat class

# How this is going to work-
# It should be always called from the start from user to agents
# any functions called in this group chat will be executed and returned to the agent