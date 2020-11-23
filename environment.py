import gym
import minerl
from gym import spaces
import numpy as np

class MyEnv(gym.Env):
    """Custom Neurocar Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(MyEnv, self).__init__()
        self.action_space = spaces.Discrete(10)
        self.env = gym.make("MineRLTreechop-v0")
        self.observation_space = self.env.observation_space
        self.env_name = "MineRLTreechop-v0"


    def step(self, a_idx):
        # return treechop_env.step(actions_arr[action])
        a = self.env.action_space.noop()
        a["attack"] = 1
        if a_idx == 0:
            # pitch + 5
            a["camera"] = [5, 0]
        elif a_idx == 1:
            # pitch -5
            a["camera"] = [-5, 0]
        elif a_idx == 2:
            # yaw +5
            a["camera"] = [0, 5]
        elif a_idx == 3:
            # yaw -5
            a["camera"] = [0, -5]
        elif a_idx == 4:
            # forward
            a["forward"] = 1
        elif a_idx == 5:
            # forward & jump
            a["forward"] = 1
            a["jump"] = 1
        elif a_idx == 6:
            # left
            a["left"] = 1
        elif a_idx == 7:
            # right
            a["right"] = 1
        elif a_idx == 8:
            # back
            a["back"] = 1
        elif a_idx == 9:
            # jump
            a["jump"] = 1
        return self.env.step(a)

    def reset(self):
        return self.env.reset()

    def render(self, mode='human'):
        return self.env.render(mode)

    def close (self):
        self.env.close()