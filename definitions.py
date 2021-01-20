"""definitions.py"""

import abc

class Environment(abc.ABC):
 
    @abc.abstractmethod
    def signal(self, action):
        """
            Produces the enviroment percepts after action is executed.
        """
        pass

    @abc.abstractmethod
    def initial_percepts(self):
        """
            Produces the enviroment initial percepts.
        """
        pass

class Agent(abc.ABC):
    """
        Implements the Agent class
    """

    def __init__(self, env: Environment):
        """
            Constructor for the agent class.

            Args:
                env: a reference to an enviroment
        """
        self.env = env

    @abc.abstractmethod
    def act(self):
        """
            Defines the agent action.
        """
        pass


