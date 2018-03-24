import cv2
from enduro.agent import Agent
from enduro.action import Action
from enduro.state import EnvironmentState
import numpy as np
from collections import defaultdict
import dill as pickle


class QAgent(Agent):
    def __init__(self, alpha = 0.01, gamma = 0.9, epsilon = 0.01):
        super(QAgent, self).__init__()

        self.total_reward = 0
        self.alpha = alpha
        self.gamma = gamma
        self.e = epsilon
        self.policy = defaultdict(lambda: Action.ACCELERATE)
        self.q = defaultdict(float)
        self.prev_state = None
        self.curr_state = None
        self.last_reward = None
        self.last_action = None

    def initialise(self, grid):
        """ Called at the beginning of an episode
        """
        self.total_reward = 0
        self.prev_state = None
        self.curr_state = None
        self.sense(grid)
        self.last_reward = None
        self.last_action = None

        if not self.learning:
            cv2.imshow("Enduro", self._image)
            cv2.imshow("Environment Grid", EnvironmentState.draw(grid))
            with open('policy.p', 'rb') as handle:
                self.policy = pickle.load(handle)


    def act(self):

        if self.learning:
            action_val = {}

            for a in self.getActionsSet():
                action_val[a] = self.q[(self.curr_state, a)]
            a_max = max(action_val, key=action_val.get)

            self.policy[self.curr_state] = a_max
            action = np.random.choice([a_max, np.random.choice(self.getActionsSet())],  p=[1-self.e, self.e])

        else:
            action = self.policy[self.curr_state]

        self.last_action = action
        self.last_reward = self.move(action)
        self.total_reward += self.last_reward

    def sense(self, grid):
        """ Constructs the next state from sensory signals.

        gird -- 2-dimensional numpy array containing the latest grid
                representation of the environment
        """
        self.prev_state = self.curr_state

        position2 = np.where(grid == 2)
        x2 = position2[1][0]
        position1 = np.where(grid == 1)
        x1_1 = None
        y1_1 = None
        x1_2 = None
        y1_2 = None
        dist = None

        for x, y in zip(position1[1], position1[0]):
            if (dist is None) or abs(x - x2) + y < dist:
                x1_1 = x
                y1_1 = y
                dist = max(abs(x - x2) - 1, 0) + y

        x1_1 = -1 if x1_1 is None else -1 if dist > 9 or abs(x1_1 - x2) > 2 else x1_1
        dist = None
        for x, y in zip(position1[1], position1[0]):
            if (x != x1_1 or y != y1_1) and ((dist is None) or abs(x - x2) + y < dist):
                x1_2 = x
                y1_2 = y
                dist = max(abs(x - x2) - 1, 0) + y
        x1_2 = -1 if x1_2 is None else -1 if dist > 9 or abs(x1_2 - x2) > 2 else x1_2
        a = 100 if x1_1 == -1 else x2 - x1_1
        b = 100 if x1_2 == -1 else x2 - x1_2
        self.curr_state = (a, b, x2 >= 7, x2 <= 2)

        if not self.learning:
            # Visualise the environment grid
            cv2.imshow("Environment Grid", EnvironmentState.draw(grid))

    def learn(self):
        """ Performs the learning procudre. It is called after act() and
        sense() so you have access to the latest tuple (s, s', a, r).
        """
        vals = []
        for a in self.getActionsSet():
            vals.append(self.q[(self.curr_state, a)])
        val_max = max(vals)

        self.q[(self.prev_state, self.last_action)] = self.q[(self.prev_state, self.last_action)] + \
        self.alpha * (self.last_reward + self.gamma * val_max - self.q[(self.prev_state, self.last_action)])


    def callback(self, episode, iteration):
        """ Called at the end of each timestep for reporting/debugging purposes.
        """
        print("{0}/{1}: {2}".format(episode, iteration, self.total_reward))
        # Show the game frame only if not learning
        if self.learning:
            if iteration == 6500:
                with open('policy.p', 'wb') as handle:
                    pickle.dump(self.policy, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            cv2.imshow("Enduro", self._image)
            cv2.waitKey(40)

if __name__ == "__main__":
    a = QAgent()
    a.run(True, episodes=500, draw=False)
    print('Total reward: ' + str(a.total_reward))
