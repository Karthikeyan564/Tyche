import gym
from gym import error, spaces, utils, logger
from gym.utils import seeding
import cityflow
import numpy as np
import os
import matplotlib.pyplot as plt
class City(gym.Env):

    def __init__(self):
        self.config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dtt")
        self.cityflow = cityflow.Engine(os.path.join(self.config_dir, "config.json"), thread_num=1)
        self.intersection_id = "intersection_1_1"

        self.sec_per_step = 15.0

        self.steps_per_episode = 200
        self.current_step = 0
        self.is_done = False
        self.all_lane_ids = list(self.cityflow.get_lane_vehicle_count())
        self.start_lane_ids = list(self.cityflow.get_lane_vehicle_count())
        print(self.all_lane_ids)
        self.action_space = spaces.Discrete(9)
        self.ss=[]
        # self.ns=0
        self.observation_space = spaces.MultiDiscrete([100]*len(self.start_lane_ids))

    def step(self, action):
        # action = int(self.cityflow.get_current_time()//15)%9
        # print(self.cityflow.get_current_time(),action)
        # action = np.random.randint(9)
        # print(action,end = "")
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        self.cityflow.set_tl_phase(self.intersection_id, action)
        self.cityflow.next_step()

        state = self._get_state()
        reward = self._get_reward()

        self.current_step += 1

        if self.is_done:
            logger.warn("You are calling 'step()' even though this environment has already returned done = True. "
                        "You should always call 'reset()' once you receive 'done = True' "
                        "-- any further steps are undefined behavior.")
            reward = 0.0
        # self.ns+=1
        # if self.ns%100==0:
        #     self.ss.append(self.cityflow.get_average_travel_time())  

        if self.current_step + 1 == self.steps_per_episode:
            # print()
            # self.plot_rewards()
            kk= self.cityflow.get_average_travel_time()
            self.ss.append(kk)
            print(kk)
            self.is_done = True

        return state, reward, self.is_done, {}


    def reset(self):
        self.cityflow.reset()
        self.is_done = False
        self.current_step = 0

        return self._get_state()

    def render(self,mode = "human"):
        # print("Current time: " ,self.cityflow.get_current_time())
        di = self.cityflow.get_lane_waiting_vehicle_count()
        print("  ",di['road_1_2_3_1'],di['road_1_2_3_0'])
        print(di['road_0_1_0_0'],end="       ");print(di['road_2_1_2_1'])
        print(di['road_0_1_0_1'],end="       ");print(di['road_2_1_2_0'])
        print("  ",di['road_1_0_1_0'],di['road_1_0_1_1'])

    def _get_state(self):
        lane_vehicles_dict = self.cityflow.get_lane_vehicle_count()
        lane_waiting_vehicles_dict = self.cityflow.get_lane_waiting_vehicle_count()

        state = np.zeros(len(self.start_lane_ids), dtype=np.float32)
        for i in range(len(self.start_lane_ids)):
            state[i] = lane_waiting_vehicles_dict[self.start_lane_ids[i]]

        return state

    def _get_reward(self):
        lane_waiting_vehicles_dict = self.cityflow.get_lane_waiting_vehicle_count()
        reward = 0.0

        for (road_id, num_vehicles) in lane_waiting_vehicles_dict.items():
            if road_id in self.start_lane_ids:
                reward -= self.sec_per_step * num_vehicles
        # reward = -self.cityflow.get_average_travel_time()
        return reward

    def set_replay_path(self, path):
        self.cityflow.set_replay_file(path)

    def seed(self, seed=None):
        self.cityflow.set_random_seed(seed)
        
    def set_save_replay(self, save_replay):
        self.cityflow.set_save_replay(save_replay)

    def plot_rewards(self):
        plt.clf()
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        # for ed in self.ss:
        #     plt.plot(ed)
        plt.plot(self.ss)
        plt.show()# Pause a bit so that the graph is updated



# env = City()
# print(env.reset())
# env.render()
# for i in range(200):
#     env.step(2)
#     env.render()



from keras.callbacks import TensorBoard
# tb =  TensorBoard(log_dir='./keras-rl')

import numpy as np
import gym
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


env = City()


# Get the environment and extract the number of actions.
# env = gym.make(mi)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n


print(nb_actions, env.observation_space.shape)

# Next, we build a very simple model.
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
# model.add(Dense(16))
# model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())
# env.set_replay_path("dataa")
# env.set_save_replay(True)
# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=100000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=200,gamma=.99,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=.00025), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
visualize = False
dqn.fit(env, nb_steps=50000, visualize=visualize, verbose=2)
env.plot_rewards()
# # After training is done, we save the final weights.
# dqn.save_weights('dqn_{}_weights.h5f'.format(Flow), overwrite=True)

# # Finally, evaluate our algorithm for 5 episodes.
# dqn.test(env, nb_episodes=5, visualize=False)







# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense

# import matplotlib.pyplot as plt
# import gym

# import huskarl as hk

# if __name__ == '__main__':

#     # Setup gym environment
#     create_env = lambda: City()
#     dummy_env = create_env()

#     # Build a simple neural network with 3 fully connected layers as our model
#     model = Sequential([
#         Dense(16, activation='relu', input_shape=dummy_env.observation_space.shape),
#         Dense(16, activation='relu'),
#         Dense(16, activation='relu'),
#     ])

#     # Create Deep Q-Learning Network agent
#     agent = hk.agent.DQN(model, actions=dummy_env.action_space.n, nsteps=2)

#     def plot_rewards(episode_rewards, episode_steps, done=False):
#         plt.clf()
#         plt.xlabel('Step')
#         plt.ylabel('Reward')
#         for ed, steps in zip(episode_rewards, episode_steps):
#             plt.plot(steps, ed)
#         plt.show() if done else plt.pause(0.001) # Pause a bit so that the graph is updated

#     # Create simulation, train and then test
#     sim = hk.Simulation(create_env, agent)
#     sim.train(max_steps=3000, visualize=False, plot=plot_rewards)
#     sim.test(max_steps=1000)