from definitions import Environment
import numpy as np
import matplotlib.pyplot as plt

class Room(Environment):

    """ This class implements an an environment that represents a room with obstacles. 
    """

    def __init__(self, room=[], initial_position=[0, 0], target=[], prob=0.3, n=10, plot_on=False):
        
        """ Room constructor

        Args:
            room: A matrix representing the free spaces, 0, and obstacles 1.
            initial_position: The position where the agents startes
            target: The position ehre the agent has to go
            room: If a room is not provided, prob is the probability the a given position will have an obstacle in the generated room
            n: size of the genrated room is a room is not given
            plot_on: Whether the environment and agent behaviour will be plotted

        """
        
        if not room:
            self.room = np.zeros((n, n))

            for i in range(len(self.room)):
                for j in range(len(self.room[0])):
                    self.room[i][j] = 1 if np.random.random() < prob else 0
        else:
            self.room = np.array(room)

        self.target = np.array(self.room.shape) - 1 if not target else np.array(target)

        # Makes sure that the target position does not have an obstacle
        self.room[self.target[0]][self.target[1]] = 0 

        self.initial_position = np.array(initial_position)
        self.current_position = np.array(initial_position)

        # For plotting purposes
        self.counter = 0
        self.fig = []
        self.ax = []
        self.prev = []
        self.plot_on = plot_on

    def initial_percepts(self):
        """ Tells the agent where it is as soon as it is born
        
        Returns:
            A set of percepts, i.e., the current position of the agent, the list of viable neighboors and the target.

        """

        if self.plot_on:
            # plot room
            self.init_plot()

            # plot initial position
            self.plot_position(self.current_position, marker_string='go')

            # plot target
            self.plot_position(self.target, marker_string='yv')

        return {'current_position': self.current_position,
                'neighbors': self.get_neighbors(self.current_position),
                'target': self.target}

    def signal(self, action):

        """ Signals the agent about the new state

        Args:
            action: An action from an Agent which contains the postion to ehre the agent desires 
            to move and the path the agent took to get there.
        
        Returns:
            A set of percepts, i.e., the current position of the agent, the list of viable neighboors and the target.

        """

        self.current_position = action['visit_position']
        neighbors = self.get_neighbors(action['visit_position'])
        self.counter += 1

        if 'path' in action:
            self.plot_path(action['path'], marker_string='-r')
        else:
            self.plot_position(self.current_position)
            self.plot_position(self.current_position, marker_string='co')

        return {'current_position': self.current_position,
                'neighbors': neighbors,
                'target': self.target}

    def get_neighbors(self, position):
        """ Signals the agent about the new state

        Args:
            position: Any position in the room         
        
        Returns:
            A set of viable positions, i.e., the neighboring positions which do not contain an obstacle 
        """

        neighborhood = np.array([[0, -1], [1, 0], [-1, 0], [0, 1], [-1, 1], [1, 1], [-1, -1], [1, -1]])
        neighbors = [position + n for n in neighborhood]

        return [n for n in neighbors if ((0 <= n[0] < self.room.shape[0]) and (0 <= n[1] < self.room.shape[1])) and (
                self.room[n[0]][n[1]] == 0)]

    def init_plot(self):
        # Plot room
        plt.ion()
        self.fig = plt.figure()
        plt.ylim(-0.1, len(self.room) - 0.9)
        plt.xlim(-0.1, len(self.room) - 0.9)
        self.ax = self.fig.add_subplot(111)

        for i in range(len(self.room)):
            for j in range(len(self.room[0])):
                if self.room[i][j] == 1:
                    self.ax.plot(j, len(self.room[0]) - 1 - i, 'ko', markersize=10)

        self.fig.canvas.draw()

    def plot_path(self, path, marker_string='-o'):
        x = np.array([p[0] for p in path])
        y = np.array([p[1] for p in path])

        lines = self.ax.plot(y, - x + len(self.room[0]) - 1, marker_string, markersize=10)

        self.fig.canvas.draw()

        l = lines.pop()
        l.remove()

    def plot_position(self, pos, marker_string='ro'):
        self.ax.plot(pos[1], len(self.room[0]) - 1 - pos[0], marker_string, markersize=10)
        self.fig.canvas.draw()
