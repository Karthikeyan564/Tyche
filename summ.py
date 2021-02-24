import gym
from gym import error, spaces, utils, logger
from gym.utils import seeding
import cityflow
import numpy as np
import os

class City(gym.Env):

    def __init__(self):
        self.config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataa")
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

        if self.current_step + 1 == self.steps_per_episode:
            # print()
            print(self.cityflow.get_average_travel_time())
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





# env = City()
# print(env.reset())
# env.render()
# for i in range(200):
#     env.step(2)
#     env.render()



from keras.callbacks import TensorBoard
tb =  TensorBoard(log_dir='./keras-rl')

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
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100,gamma=.9,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=.00025), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
visualize = False
dqn.fit(env, nb_steps=50000, visualize=visualize, verbose=2,callbacks=[tb])

# After training is done, we save the final weights.
dqn.save_weights('dqn_{}_weights.h5f'.format(Flow), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=5, visualize=False)



# import gym
# import numpy as np
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# from keras.models import Sequential
# from keras.layers import Dense, Activation, Flatten
# from keras.optimizers import Adam

# env = City()
# nb_actions = env.action_space.n
# num_actions =nb_actions
# def md():
#     model = Sequential()
#     model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
#     model.add(Dense(16))
#     model.add(Activation('relu'))
#     model.add(Dense(16))
#     model.add(Activation('relu'))
#     model.add(Dense(nb_actions))
#     model.add(Activation('linear'))
#     return model

# model=md()
# model_target=md()
# # print(model.summary())


# gamma = 0.99
# epsilon = 1.0  
# epsilon_min = 0.1  # Minimum epsilon greedy parameter
# epsilon_max = 1.0  # Maximum epsilon greedy parameter
# epsilon_interval = (
#     epsilon_max - epsilon_min
# )  # Rate at which to reduce chance of random action being taken
# batch_size = 32  # Size of batch taken from replay buffer
# max_steps_per_episode = 10000




# optimizer = keras.optimizers.Adam(learning_rate=0.001)
# hisp=[]
# hisp2=[]
# hisp3=[]
# action_history = []
# state_history = []
# state_next_history = []
# rewards_history = []
# done_history = []
# episode_reward_history = []
# running_reward = 0
# episode_count = 0
# frame_count = 0


# # Number of frames to take random action and observe output
# epsilon_random_frames = 100
# # Number of frames for exploration
# epsilon_greedy_frames = 1000.0

# max_memory_length = 100000
# # How often to update the target network
# update_target_network = 400
# # Using huber loss for stability
# loss_function = keras.losses.MeanSquaredError()  #jfjf

# while True:  # Run until solved
#     state = np.array(env.reset())
#     episode_reward = 0


#     for timestep in range(1, max_steps_per_episode):
#         # env.render(); Adding this line would show the attempts
#         # of the agent in a pop up window.
#         frame_count += 1

#         # Use epsilon-greedy for exploration
#         if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
#             # Take random action
#             action = np.random.choice(num_actions)
#         else:
#             # Predict action Q-values
#             # From environment state
#             state_tensor = tf.convert_to_tensor(state)
#             state_tensor = tf.expand_dims(state_tensor, 0)
#             action_probs = model(state_tensor, training=False)
#             # Take best action
#             action = tf.argmax(action_probs[0]).numpy()

#         # Decay probability of taking random action
#         epsilon -= epsilon_interval / epsilon_greedy_frames
#         epsilon = max(epsilon, epsilon_min)

#         # Apply the sampled action in our environment
#         state_next, reward, done, _ = env.step(action)
#         state_next = np.array(state_next)

#         episode_reward += reward

#         # Save actions and states in replay buffer
#         action_history.append(action)
#         state_history.append(state)
#         state_next_history.append(state_next)
#         done_history.append(done)
#         rewards_history.append(reward)
#         state = state_next


#         if frame_count%1000==0:
#           print("                                   record:",np.mean(episode_reward_history[-10:]))





#         # Update every fourth frame and once batch size is over 32
#         if len(done_history) > batch_size:

#             # Get indices of samples for replay buffers
#             indices = np.random.choice(range(len(done_history)), size=batch_size)

#             # Using list comprehension to sample from replay buffer
#             state_sample = np.array([state_history[i] for i in indices])
#             state_next_sample = np.array([state_next_history[i] for i in indices])
#             rewards_sample = [rewards_history[i] for i in indices]
#             action_sample = [action_history[i] for i in indices]
#             done_sample = tf.convert_to_tensor(
#                 [float(done_history[i]) for i in indices]
#             )


#             # Build the updated Q-values for the sampled future states
#             # Use the target model for stability
#             future_rewards = model_target.predict(state_next_sample)  #re
#             # Q value = reward + discount factor * expected future reward
#             updated_q_values = rewards_sample + gamma * tf.reduce_max(future_rewards, axis=1)

#             updated_q_values = updated_q_values * (1 - done_sample) - done_sample




#             # Create a mask so we only calculate loss on the updated Q-values
#             masks = tf.one_hot(action_sample, num_actions)

#             with tf.GradientTape() as tape:
#                 # Train the model on the states and updated Q-values
#                 q_values = model(state_sample)

#                 # Apply the masks to the Q-values to get the Q-value for action taken
#                 q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
#                 # Calculate loss between new Q-value and old Q-value
#                 loss = loss_function(updated_q_values, q_action)

#             # Backpropagation
#             grads = tape.gradient(loss, model.trainable_variables)
#             optimizer.apply_gradients(zip(grads, model.trainable_variables))

#         if frame_count % update_target_network == 0:
#             # update the the target network with new weights
#             model_target.set_weights(model.get_weights())
#             # Log details
#             # template = "running reward: {:.2f} at episode {}, frame count {}"
#             # print(template.format(running_reward, episode_count, frame_count))
#         # Limit the state and reward history
#         if len(rewards_history) > max_memory_length:
#             del rewards_history[:1]
#             del state_history[:1]
#             del state_next_history[:1]
#             del action_history[:1]
#             del done_history[:1]

#         if done:
#             break

#     # Update running reward to check condition for solving
#     episode_reward_history.append(episode_reward)
#     if len(episode_reward_history) > 100:
#         del episode_reward_history[:1]
#     running_reward = np.mean(episode_reward_history)
#     print("episode {}   {}     {}    {}".format(episode_count,running_reward,episode_reward,frame_count))
#     episode_count += 1
#     hisp.append(np.mean(episode_reward_history[-10:]))
#     hisp2.append(running_reward)
#     if running_reward > 300:  # Condition to consider the task solved
#         print("Solved at episode {}!".format(episode_count))
#         break
