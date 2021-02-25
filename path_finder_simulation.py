from room import Room
from path_finder_agents import  IterativeDeepeningAgent, BFSAgent, DFSAgent, AStarAgent, BBAgent, GreedyAgent

# env = Room(room=[[0, 0, 1], [0, 0, 1], [0, 0, 0]], target=[2, 2])
env = Room(prob=0.2, n=25, plot_on=True)

# agent = BFSAgent(env)
# agent = DFSAgent(env)
agent = AStarAgent(env)
# agent = GreedyAgent(env)

# agent = BBAgent(env, bound=20)
# agent = IterativeDeepeningAgent(env)


agent.run()
input('Press ENTER to get out of here')
