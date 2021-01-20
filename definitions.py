"""definitions.py"""

import abc

class Environment(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def signal(self, action):
        pass

    @abc.abstractmethod
    def initial_percepts(self):
        pass

class Agent(abc.ABC):

    def __init__(self, env: Environment):
        self.env = env

    @abc.abstractmethod
    def act(self):
        pass
   
