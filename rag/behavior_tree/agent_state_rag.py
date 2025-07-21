# Define the Agent class to track state
class Agent:
    def __init__(self, name, start_location):
        self.name = name
        self.current_location = start_location
        self.has_item = False
        # task_start and task_end define the current mission (pickup or delivery)
        self.task_start = start_location
        self.task_end = None  # will be set when a pickup task is assigned
        self.current_path = []
        self.path_index = 0

    def set_pickup_task(self, item_location):
        """Assign a new pickup task for the agent (go from current location to item_location)."""
        self.has_item = False
        self.task_start = self.current_location
        self.task_end = item_location
        self.current_path = []
        self.path_index = 0

    def set_deliver_task(self):
        """Switch to a delivery task after picking up an item (deliver from current location to E)."""
        self.has_item = True
        self.task_start = self.current_location  # pickup location
        self.task_end = "E"                      # deliver to central hub
        self.current_path = []
        self.path_index = 0

# Initialize the RAG database with initial paths (bidirectional for simplicity)
rag_db = {
    ("E", "A"): ["E", "X", "Y", "A"],
    ("A", "E"): ["A", "X", "Y", "E"],
    ("E", "B"): ["E", "Z", "B"],
    ("B", "E"): ["B", "Z", "E"],
    ("E", "C"): ["E", "W", "V", "C"],
    ("C", "E"): ["C", "V", "W", "E"],
    ("E", "D"): ["E", "U", "T", "D"],
    ("D", "E"): ["D", "T", "U", "E"]
}

# Predefined alternate paths that agents will "discover" if certain routes are blocked
new_path_E_C = ["E", "W", "P", "Q", "C"]  # alternate path from E to C (bypassing V)
new_path_C_E = ["C", "Q", "P", "W", "E"]  # alternate path from C to E (reverse route)
new_path_A_E = ["A", "X", "M", "N", "E"]  # alternate path from A to E (bypassing Y)
new_path_E_A = ["E", "N", "M", "X", "A"]  # alternate path from E to A (reverse route)
