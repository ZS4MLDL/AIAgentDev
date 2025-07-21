import py_trees
import time
from py_trees.common import Status
from agent_state_rag import rag_db, new_path_A_E, new_path_C_E, new_path_E_A, new_path_E_C
from py_trees.blackboard import Blackboard

# Condition: checks if the agent is currently carrying an item
class HasItem(py_trees.behaviour.Behaviour):
    def __init__(self, agent):
        super().__init__(name=f"{agent.name}_HasItem")
        self.agent = agent

    def update(self):
        return Status.SUCCESS if self.agent.has_item else Status.FAILURE

# Condition: checks if the agent does NOT have an item
class NoItem(py_trees.behaviour.Behaviour):
    def __init__(self, agent):
        super().__init__(name=f"{agent.name}_NoItem")
        self.agent = agent

    def update(self):
        return Status.SUCCESS if not self.agent.has_item else Status.FAILURE

# Action: Query the RAG for a path from task_start to task_end
class QueryRAG(py_trees.behaviour.Behaviour):
    def __init__(self, agent):
        super().__init__(name=f"{agent.name}_QueryRAG")
        self.agent = agent

    def update(self):
        
        if not self.agent.current_path:
            start, end = self.agent.task_start, self.agent.task_end
            path = rag_db[(start,end)]
            self.agent.current_path = path[:]
            self.agent.path_index = self.agent.current_path.index(self.agent.current_location)
            print(f"{self.agent.name}: Queries RAG for {start} -> {end}, gets {path}")
        return Status.SUCCESS

# Action: Follow the current path one step at a time
class FollowPath(py_trees.behaviour.Behaviour):
    def __init__(self, agent):
        super().__init__(name=f"{agent.name}_FollowPath")
        self.agent = agent
        

    def update(self):
        try:
            current_time = Blackboard.get("current_time")
        except KeyError:
            return Status.FAILURE
        # If no path or already at end, nothing to do
        if not self.agent.current_path or self.agent.path_index >= len(self.agent.current_path) - 1:
            return Status.SUCCESS
        # Determine the next node on the path
        next_index = self.agent.path_index + 1
        next_node = self.agent.current_path[next_index]
        # Check for dynamic blockage scenarios:
        if (self.agent.name == "Agent3" and next_node == "V" and current_time == 2):
            # Agent3 trying to go through V at time 1 -> blocked
            time.sleep(2)
            print(f"{self.agent.name}: Finds {next_node} is blocked.")
            return Status.FAILURE
        if (self.agent.name == "Agent1" and next_node == "Y"  and not self.agent.has_item  and current_time == 2):
            # Agent1 trying to return via Y at time 4 -> blocked
            time.sleep(2)
            print(f"{self.agent.name}: Finds {next_node} is blocked.")
            return Status.FAILURE
        # If not blocked, move to the next node
        time.sleep(2)
        print(f"{self.agent.name}: Moves from {self.agent.current_location} to {next_node}")
        self.agent.current_location = next_node
        self.agent.path_index = next_index
        # If this move reached the end of the path, we're at the destination
        if self.agent.current_location == self.agent.task_end:
            return Status.SUCCESS
        else:
            return Status.RUNNING

# Action: Pick up item at current location (executes when agent reaches an item location)
class PickUpItem(py_trees.behaviour.Behaviour):
    def __init__(self, agent):
        super().__init__(name=f"{agent.name}_PickUpItem")
        self.agent = agent

    def update(self):
        try:
            current_time = Blackboard.get("current_time")
        except KeyError:
            return Status.FAILURE
        # Simulate picking up the item.
        time.sleep(2)
        print(f"{self.agent.name}: Picks up item at {self.agent.current_location}")
        # Update agent state to start delivery task
        self.agent.set_deliver_task()
        return Status.SUCCESS

# Action: Drop off item at E (executes when agent reaches the delivery hub with an item)
class DropOffItem(py_trees.behaviour.Behaviour):
    def __init__(self, agent):
        super().__init__(name=f"{agent.name}_DropOffItem")
        self.agent = agent

    def update(self):
        time.sleep(2)
        print(f"{self.agent.name}: Drops off item at {self.agent.current_location}")
        # Update agent state to having no item; task is complete
        self.agent.has_item = False
        # (In a more complex scenario, we might assign a new pickup task here if available)
        return Status.SUCCESS

# Action: Explore to find a new path when blocked
class ExploreAction(py_trees.behaviour.Behaviour):
    def __init__(self, agent):
        super().__init__(name=f"{agent.name}_Explore")
        self.agent = agent
        self.explored = False

    def update(self):
        if self.explored:
            return Status.SUCCESS

        ct = Blackboard.get("current_time")
        print(f"{self.agent.name}: !!! Entering ExploreAction at time {ct}")

        # Agent3 outbound E→C
        if (self.agent.name=="Agent3"
            and self.agent.task_start=="E"
            and self.agent.task_end=="C"):
            rag_db[("E","C")] = new_path_E_C[:]
            rag_db[("C","E")] = new_path_C_E[:]
            print(f"{self.agent.name}: Explores and discovers new path {rag_db[('E','C')]} for E -> C")

        # Agent1 **inbound** E→A
        elif (self.agent.name=="Agent1"
              and self.agent.task_start=="E"
              and self.agent.task_end=="A"):
            # assign the new E→A path
            rag_db[("E","A")] = new_path_E_A[:]   # bypassing Y inbound
            rag_db[("A","E")] = new_path_A_E[:]
            print(f"{self.agent.name}: Explores and discovers new path {rag_db[('E','A')]} for E -> A")

        # Agent1 **return** A→E
        elif (self.agent.name=="Agent1"
              and self.agent.task_start=="A"
              and self.agent.task_end=="E"):
            rag_db[("A","E")] = new_path_A_E[:]
            rag_db[("E","A")] = new_path_E_A[:]
            print(f"{self.agent.name}: Explores and discovers new path {rag_db[('A','E')]} for A -> E")

        else:
            print(f"{self.agent.name}: ExploreAction ran but no matching case.")

        # clear path so QueryRAG will re‑fetch
        self.agent.current_path = []
        self.agent.path_index = 0
        self.explored = True
        return Status.SUCCESS
