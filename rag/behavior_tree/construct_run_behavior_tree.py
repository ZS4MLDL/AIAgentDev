import py_trees
import time
from agent_state_rag import Agent
from behavior_tree_nodes import HasItem,QueryRAG,FollowPath, DropOffItem, NoItem, PickUpItem, ExploreAction
from py_trees.blackboard import Blackboard

# Create agents and their behavior trees
agents = []
for name, pickup_location in [("Agent1", "A"), ("Agent2", "B"), ("Agent3", "C")]:
    agent = Agent(name, start_location="E")
    agent.set_pickup_task(pickup_location)  # initialize each agent with its pickup target
    # Build the behavior tree for this agent
    root = py_trees.composites.Selector(name=f"{name}_Root", memory=False)

    # Deliver sequence (runs when agent has an item)
    deliver_seq = py_trees.composites.Sequence(name=f"{name}_DeliverSeq", memory=True)
    deliver_seq.add_children([
        HasItem(agent),
        QueryRAG(agent),      # query path from current pickup location to E
        FollowPath(agent),    # follow the path towards E
        DropOffItem(agent)    # drop off item at E when reached
    ])

    # PickUp sequence (runs when agent does not have an item)
    pickup_seq = py_trees.composites.Sequence(name=f"{name}_PickUpSeq",memory=True)
    pickup_seq.add_children([
        NoItem(agent),
        QueryRAG(agent),     # query path from current location (E) to target item location
        FollowPath(agent),   # follow the path towards item location
        PickUpItem(agent)    # pick up the item when reached
    ])

    # Explore sequence (fallback if above fails due to blockage)
    explore_seq = py_trees.composites.Sequence(name=f"{name}_ExploreSeq", memory=True)
    explore_seq.add_children([ExploreAction(agent)])  
    # (We could include a condition here if we only want to explore under certain failure contexts, 
    #  but in this design, ExploreAction is only reached when preceding sequence fails.)

    # Assemble the tree
    root.add_children([deliver_seq, pickup_seq, explore_seq])
    # Attach the tree to a BehaviorTree runner for ticking
    agent.tree = py_trees.trees.BehaviourTree(root)
    agents.append(agent)


# Simulate 5 time steps
time.sleep(7) 
for t in range(1, 6):   # using 1-indexed time steps for logging clarity
    # update your BT’s notion of current_time
    time.sleep(2) 
    Blackboard.set("current_time", t)
    print(f"\nTime {t}:", flush=True)

    # tick each agent and pause briefly afterward
    for agent in agents:
        agent.tree.tick()
        time.sleep(0.1)          # half‑second pause between agents
    # optional extra pause before the next time step
    time.sleep(1.0)  