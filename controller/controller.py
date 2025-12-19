
from service.nodes import Node_Service


class Controller:
    def __init__(self, model):
        self.llm = model
        self.service = Node_Service(model)

    def ask(self, question: str):
        state = {"question": question, "data_source": "", "answer": ""}
        result = self.service.ask(state)
        return result





