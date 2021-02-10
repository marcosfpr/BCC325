from definitions import Agent
import numpy as np
import heapq as heap
from scipy.spatial import distance


class HeapPath(object):
    def __init__(self, value, path):
        self.path = path
        self.value = value
    
    def get_path(self):
        return self.path

    # função de comparação da heap minima
    def __lt__(self, other):
        return self.value < other.value # compara apenas em relação ao valor chave


class PathFinderAgent(Agent):
    """
    This class implements a path finder agent base class.
    """
    def __init__(self, env):
        """Connects to the next available port.

        Args:
            env: A reference to an environment.

        """

        # Make a connection to the environment using the superclass constructor
        Agent.__init__(self,env)
        
        # Get initial percepts
        self.percepts = env.initial_percepts()
        
        # Initializes the frontier with the initial postion 
        self.frontier = [[self.percepts['current_position']]]
        
        # Initializes list of visited nodes for multiple path prunning
        self.visited = []


    def act(self):
        """Implements the agent action
        """

        # seleciona um caminho da fronteira
        path = self.remove_from_frontier()
        
        # visita o caminho e retorna seus vizinhos viaveis
        viable_neighbors = self.visit_neighbours(path)
        
        # atualiza a fronteira
        self.update_frontier(path, viable_neighbors)


    def visit_neighbours(self, path):
        # Visit the last node in the path
        action = {'visit_position': path[-1], 'path': path} 
        # The agente sends a position and the full path to the environment, the environment can plot the path in the room 
        self.percepts = self.env.signal(action)

        # Add visited node 
        self.visited.append(path[-1])

        # From the list of viable neighbors given by the environment
        # Select a random neighbor that has not been visited yet
        
        return self.percepts['neighbors']

    
    def remove_from_frontier(self):
        """
        Removes one path from the frontier
        """
        return self.frontier.pop(0)

    def update_frontier(self, path, viable_neighbors):
        """
        Updates the frontier based on current path
        """
        # If the agent is not stuck
        if viable_neighbors: 
            # Select random neighbor
            visit = viable_neighbors[np.random.randint(0,len(viable_neighbors))]
            
            # Append neighbor to the path and add it to the frontier
            self.frontier = [path + [visit]] + self.frontier

    def run(self):
        """Keeps the agent acting until it finds the target
        """
        while (self.percepts['current_position'] != self.percepts['target']).any() and self.frontier:
            self.act()
        print(self.percepts['current_position'])
    

    def is_cicle(self, neighbor, path) -> bool:
        return any(map(lambda node: (node == neighbor).all(), path))
    
    def is_explored(self, neighbor) -> bool:
        if self.visited:
            return any(map(lambda row: all(row), (neighbor == self.visited)))
        return False


class BFSAgent(PathFinderAgent):
    """
    A unica mudança da BFS para o PathFinderAgent está na forma como ela adiciona os nós na fronteira => FIFO
    """
    def update_frontier(self, path, viable_neighbors):
        # If the agent is not stuck
        if viable_neighbors:         
            for neighbor in viable_neighbors:            
                # Append neighbor to the path and add it to the frontier

                #  colocar a poda de multiplos caminhos aqui nao vai evitar
                #  que o mesmo vizinho seja adicionado na fronteira mais de uma vez
                if self.is_cicle(neighbor, path) or self.is_explored(neighbor):
                    continue

                self.frontier = self.frontier + [path + [neighbor]]

class DFSAgent(PathFinderAgent):
    """
    A unica mudança da DFS para o PathFinderAgent está na forma como ela adiciona os nós na fronteira => LIFO
    """
    def update_frontier(self, path, viable_neighbors):
        # If the agent is not stuck
        if viable_neighbors:         
            for neighbor in viable_neighbors:            
                # Append neighbor to the path and add it to the frontier

                if self.is_cicle(neighbor, path) or self.is_explored(neighbor):
                    continue
                self.frontier = [path + [neighbor]] + self.frontier 


class GreedyAgent(PathFinderAgent):
    """
        Utiliza-se de uma heap mínima baseada no valor da heurística h
    """
    def __init__(self, env):
        PathFinderAgent.__init__(self, env)
        self.frontier = []

        initial_pos = self.percepts['current_position']

        h = distance.euclidean(initial_pos, self.percepts['target'])

        heap.heappush(self.frontier, HeapPath(h, [initial_pos]))

    def remove_from_frontier(self):
        return heap.heappop(self.frontier).get_path()

    def update_frontier(self, path, viable_neighbors):
        if viable_neighbors:
            for neighbor in viable_neighbors:            
                # Append neighbor to the path and add it to the frontier
                # avoiding cicles
                if self.is_cicle(neighbor, path) or self.is_explored(neighbor):
                    continue
                
                new_path =  path + [neighbor]

                h = distance.euclidean(neighbor, self.percepts['target'])

                heap.heappush(self.frontier, HeapPath(h, new_path))


class AStarAgent(PathFinderAgent):
    """
        Utiliza-se de uma heap mínima baseada no valor de f = custo +  heuristica
    """
    def __init__(self, env):
        PathFinderAgent.__init__(self, env)
        self.frontier = []

        initial_cost = 0
        initial_pos = self.percepts['current_position']
        path = {'path': [initial_pos], 'cost': initial_cost}

        f = initial_cost + distance.euclidean(initial_pos, self.percepts['target'])

        heap.heappush(self.frontier, HeapPath(f, path))


    def remove_from_frontier(self):
        return heap.heappop(self.frontier).get_path() # recupero apenas o caminho, o valor de f não preciso mais

    def visit_neighbours(self, path):
        return super().visit_neighbours(path['path'])


    def update_frontier(self, path, viable_neighbors):
        cost = path['cost']
        actual_path = path['path']
        if viable_neighbors:
            for neighbor in viable_neighbors:            
                # Append neighbor to the path and add it to the frontier
                # avoiding cicles
                if self.is_cicle(neighbor, actual_path) or self.is_explored(neighbor):
                    continue
                
                new_cost =  cost + distance.euclidean(actual_path[-1], neighbor)
                new_path = {'path': actual_path + [neighbor], 'cost': new_cost}

                f = new_cost + distance.euclidean(neighbor, self.percepts['target'])

                heap.heappush(self.frontier, HeapPath(f, new_path))

            
class BBAgent(PathFinderAgent):
    def __init__(self, env, bound=100):
        PathFinderAgent.__init__(self, env)

        self.bound = bound 
        self.cost = [0]
        self.best_path = []

        self._last_cost = None

    def remove_from_frontier(self):
        self._last_cost = self.cost.pop(0)
        return super().remove_from_frontier()

    def visit_neighbours(self, path):

        last_path_cost = self._last_cost # custo do caminho que foi retirado da fronteira

        # print(f"==> cost: {last_path_cost}, target: {self.percepts['target']}, node: {path[-1]}, target:{self.percepts['target']}, compare:  {last_path_cost + distance.euclidean(path[-1], self.percepts['target']) < self.bound}")
        if last_path_cost + distance.euclidean(path[-1], self.percepts['target']) < self.bound:
            # Visit the last node in the path
            action = {'visit_position': path[-1], 'path': path} 
            # The agente sends a position and the full path to the environment, the environment can plot the path in the room 
            self.percepts = self.env.signal(action)

            # Add visited node 
            self.visited.append(path[-1])

            if (self.percepts['current_position'] == self.percepts['target']).all():
                self.best_path = path 
                self.bound = last_path_cost
                print(self.bound)

        return self.percepts['neighbors'] # retornar vizinhos viáveis


    def update_frontier(self, path, viable_neighbors):
        last_path_bost = self._last_cost
        # If the agent is not stuck
        if viable_neighbors:         
            for neighbor in viable_neighbors:            
                # Append neighbor to the path and add it to the frontier
                # avoiding cicles
                if self.is_cicle(neighbor, path):
                    continue
                self.frontier = [path + [neighbor]] + self.frontier 
                self.cost = [last_path_bost + distance.euclidean(path[-1], [neighbor])] + self.cost
    
    def run(self):
        
        while self.frontier:
            self.act()
        print(self.percepts['current_position'])

        if self.best_path:
            for i in range(1000):
                action = {'visit_position': self.best_path[-1], 'path': self.best_path} 
                self.percepts = self.env.signal(action)


class IterativeDeepeningAgent(PathFinderAgent):
    """
    This class implements an agent that explores the environmente randomly
    until it reaches the target
    """
    def __init__(self, env):
        PathFinderAgent.__init__(self, env)
        self.bound = None

    def update_frontier(self, path, viable_neighbors):
        # If the agent is not stuck
        if viable_neighbors and len(path) < self.bound: 
            
            for neighbor in viable_neighbors:            
                # Append neighbor to the path and add it to the frontier
                # avoiding cicles
                if self.is_cicle(neighbor, path) or self.is_explored(neighbor):
                    continue

                self.frontier = [path + [neighbor]] + self.frontier

    def run(self):
        """Keeps the agent acting until it finds the target
        """

        # Run agent
        self.bound = 1
        initial_position = self.frontier[0]

        while (self.percepts['current_position'] != self.percepts['target']).any():
            self.frontier = [initial_position]
            self.visited = []
            while (self.percepts['current_position'] != self.percepts['target']).any() and self.frontier:
                self.act()
            self.bound += 1
            

        print(self.percepts['current_position'])
