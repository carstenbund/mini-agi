# Thread manager to orchestrate multiple chat threads
class ChatThread:
    def __init__(self, name):
        self.name = name
        self.history = []

    def add_message(self, message):
        self.history.append(message)

    def get_history(self):
        return self.history
