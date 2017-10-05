import sys, gym
import ppaquette_gym_doom
from doom_utils import ToDiscrete
from utils import ReplayMemory, PreprocessImage, Transition, calculate_nsteps
from datetime import date
import time
import pickle
import torch

GAMMA = 0.99
FRAMESKIP = 4

class TransitionSaver:
    def __init__(self):
        self.processor = PreprocessImage(None)
        self.memory = ReplayMemory()
        self.transitions = []

    def new_episode(self, first_state):
        self.state = self.processor._observation(first_state)

    def add_transition(self, action, next_state, reward):
        if next_state is not None:
            next_state = self.processor._observation(next_state)
            self.transitions.insert(0, Transition(state=self.state, action=self.add_noop(action),
                next_state=next_state, reward=torch.FloatTensor([reward]), n_reward=torch.zeros(1)))

            self.transitions = calculate_nsteps(self.transitions, reward)
        else:
            for trans in reversed(self.transitions):
                self.memory.push(trans)
            self.transitions = []
        self.state = next_state
    
    def add_noop(self, actions):
        actions.insert(0, 0)
        actions = torch.LongTensor(actions)
        actions[0] = (1 - actions[1:].max(0)[0])[0]
        return actions.max(0)[1]

    def save(self, fname):
        with open(fname, 'wb') as memory_file:
            pickle.dump(self.memory, memory_file)


saver = TransitionSaver()
def _play_human_mode(self):
    state = self.game.get_state().image_buffer.copy()
    saver.new_episode(state)
    while not self.game.is_episode_finished():
        self.game.advance_action(FRAMESKIP)
        img = self.game.get_state().image_buffer
        if img is not None:
            state = img.copy()
        if self.game.is_episode_finished():
            state = None
        action = self.game.get_last_action()
        reward = self.game.get_last_reward()
        saver.add_transition(action, state, reward)
        time.sleep(0.02857)  # 35 fps = 0.02857 sleep between frames
    return

ppaquette_gym_doom.doom_env.DoomEnv._play_human_mode = _play_human_mode


DOOM_ENV = 'DoomMyWayHome-v0'

for i in range(10):
    env = gym.make('ppaquette/' + DOOM_ENV)
    env = ToDiscrete("minimal")(env)
    env.unwrapped._mode = 'human'
    env.reset()

timestring = str(date.today()) + '_' + time.strftime("%Hh-%Mm-%Ss", time.localtime(time.time()))
saver.save('demos/' + DOOM_ENV + '_demo_' + 'fs_'  + str(FRAMESKIP) + '_' + timestring + '.p')