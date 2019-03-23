# TODO: your agent here!
from past.builtins import basestring
import numpy as np
from task import Task
import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Lambda, Activation, BatchNormalization
from keras.optimizers import Adam
from keras.layers.merge import Add
from collections import deque, defaultdict
from keras import backend as K
from tensorflow.python.util import nest
from keras.layers.advanced_activations import LeakyReLU as leaky_relu


class DeepLearnAgent(object):
    def __init__(
            self, 
            task, 
            n_inputs=12,
            actor_optimizer=tf.train.AdamOptimizer,
            actor_initializer='truncated_normal',
            critic_initializer='truncated_normal',
            actor_final_activation='tanh',
            actor_neuron_counts=[16,32,16],        
            critic_state_neuron_counts=[32,16], 
            critic_action_neuron_counts=[24,16], 
            critic_combined_neuron_counts=[16,24],
            critic_update_rate=0.02, 
            actor_update_rate=0.02,
            actor_lr=0.001,
            critic_lr=0.001,
            gamma=0.99,
            epsilon=0.01,
            memory_size=250,
            target_increment=0.05,
            verbose=True,
            noise_ratio = 0.25,
            clipnorm=0.05,
            **kwargs
          
        ):
        
        K.set_learning_phase(True) 
        self.gamma = gamma
        self._initialize_hyperparams()
        self.noise_ratio = noise_ratio
        self.critic_update_rate=critic_update_rate 
        self.actor_update_rate=actor_update_rate
        self.n_inputs = n_inputs
        self._set_task_attributes(task)
        
        self.clipnorm = clipnorm
        self.memory_size = memory_size
        self.clear_memory()
        
        self.verbose = verbose
        self.actor_initializer = actor_initializer
        self.actor_optimizer = actor_optimizer
        self.actor_lr, self.critic_lr = actor_lr, critic_lr
        self.actor_neuron_counts = actor_neuron_counts
        # going to preserve critics, so don't need to modularize their init function right now
        self.init_actors()
        
        self.critic, self.critic_state_in, self.critic_action_in = self.create_critic(
            lr=actor_lr, initializer=critic_initializer,
            state_neuron_counts=critic_state_neuron_counts,
            action_neuron_counts=critic_action_neuron_counts,
            combined_neuron_counts=critic_combined_neuron_counts,)
        self.target_critic = self.create_critic(
            lr=critic_lr, initializer=critic_initializer,
            state_neuron_counts=critic_state_neuron_counts,
            action_neuron_counts=critic_action_neuron_counts,
            combined_neuron_counts=critic_combined_neuron_counts,)[0]
         
    def init_actors(self):
        # keep starting up new actors and see if the critic can get a headstart
        self.actor, self.actor_state_in = self.create_actor(
            lr=self.actor_lr, initializer=self.actor_initializer, 
            neuron_counts=self.actor_neuron_counts,)
        self.target_actor = self.create_actor(
            lr=self.actor_lr, initializer=self.actor_initializer,
            neuron_counts=self.actor_neuron_counts,)[0]
   
        
        
    def _set_task_attributes(self, task):
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high
        self.action_range = self.action_high - self.action_low
        
    def _initialize_hyperparams(self):
        self.epsilon_increment = 5
        
    @property
    def epsilon(self):
        return 1/self.epsilon_increment
    
    def clear_memory(self):
        self.memory = deque([], maxlen=self.memory_size)
    
    def act_many_times(self, task, state, n=100):
        # print("I am acting many times for the first time with an input state with shape {}.".format(state.shape))
        for nn in range(n):
       
            rotor_speeds, next_state, reward, done = self.act_and_store(task, state)
            # print("I made these rotor speeds for round {}: {}".format(nn, rotor_speeds))
            self.total_reward += reward
            rotor_mean = np.mean(rotor_speeds)
            self.total_speed += np.mean(rotor_speeds) 
            self.total_height += state[2]
            self.total_height_error += np.abs(state[2] - task.hover_height)
            self.count += 1
            self.best_reward = max(self.best_reward, reward)
            if done:
#                 print("Done with:")
#                 print("State: {}".format(state))
                
#                 print("Rotor Speeds: {}".format(rotor_speeds))
#                 print("Reward: {}".format(reward))
#                 print("Next State: {}".format(next_state))
                
                self.report_episodic_progress()
                break
            else:
                state = next_state
                
        return state
    
    
    def report_periodic_progress(self):
        pass
        
    def reset_episode(self):
        self.total_reward = 0.0
        self.total_speed = 0.0
        self.total_height = 0.0
        self.total_height_error = 0.0
        self.count = 0
        self.best_reward = -np.Inf
        state = self.task.reset()
        return state
    
    def report_episodic_progress(self):
        if self.count==1:
            print(self.last_action)
            print("Looks like model is dead.")
            raise ValueError("Model is dead.")
        if not self.verbose:
            return
        print("My total reward was: {} over {} timesteps, an average of {} reward.".format(
                self.total_reward, self.count, self.total_reward/self.count, ))
        
        print("I tried an average rotor speed of {}.".format(
                self.total_speed/self.count, ))
        print("My average height was {}, and my average height error was.".format(
                self.total_height/self.count, self.total_height_error, self.count ))
        print("My best reward in a single timestep was {}".format(self.best_reward))
        
    
    
    def do_periodic_update(self, **kwargs):
        self.epsilon_increment += 1
        self.update_target_models(**kwargs)
        self.report_periodic_progress()
    
    
    
    def act_and_store(self, task, state):
        # print("I am acting and storing with a state of shape {}.".format(state.shape))
        action = self.act(state=state, exploring=True)
       
        rotor_speeds = self._scale_output_to_action(action)
        # print("I made these rotor speeds: {}".format(rotor_speeds))
        # RL take action and get next observation and reward
        try:
            # print("Stepping task with these rotor speeds: {}".format(rotor_speeds))
            next_state, reward, done = task.step(rotor_speeds)
        except (ValueError, TypeError) as ex:
            print("""
               !!!!
               {}
            """.format(ex))
            print(rotor_speeds)
        
        self.store_experience(
            state=state, 
            action=action, 
            reward=reward, 
            next_state=next_state)
        

        return rotor_speeds, next_state, reward, done

    def store_experience(self, state, action, reward, next_state):
        self.memory.appendleft(dict(
            state=state, action=action, reward=reward, 
            next_state=next_state))
        

        
        
        
    def _add_dense_layers(
        self,
        inputs, 
        neuron_counts, 
        dropout_rate=0.50,
        activation=leaky_relu,
        initializer='truncated_normal',
        ):
        layer = inputs
        for num in neuron_counts:
            layer = Dense(
                num, 
                # activation=activation, 
                kernel_initializer=initializer)(layer)
            layer = activation()(layer)
            if dropout_rate:
                layer = Dropout(dropout_rate)(layer)
        return layer
        
    
    def learn_from_experiences(self):
#         self.batch_learn()
#         return
        
        for experience in self.memory:
            self.learn_from_one(experience)
    
    
    
    def batch_learn(self):

        all_states = np.array([ex["state"] for ex in self.memory])
        
        all_actions = np.array([ex["action"] for ex in self.memory])
        all_reward = np.array([ex["reward"] for ex in self.memory])
        all_next_states = np.array([ex["next_state"] for ex in self.memory])
#         print(all_states.shape)
#         print(all_actions.shape)
#         print(all_reward.shape)
#         print(all_next_states.shape)
        
        actor_target_action = self.act_many(states=all_next_states, model=self.target_actor)
#         print(actor_target_action.shape)
        critic_inputs = [all_next_states, actor_target_action]
        critic_target_next_reward = self.target_critic.predict(x=critic_inputs)
        
        discounted_reward = all_reward + critic_target_next_reward * self.gamma
        print(discounted_reward.shape)
        self.critic.fit(critic_inputs, discounted_reward, verbose=0)
        
        # these shapes don't work
        
        actor_action = self.act(state=all_states, model=self.actor)
        predicted_reward = self.critic_predict_many(
            state=all_states, action=actor_action, model=self.target_critic)
        actor_error = discounted_reward - predicted_reward
        actor_next_action = self.act(state=state, model=self.target_actor)
        self.actor.fit(all_states, actor_error, verbose=0)
        
    
    def update_target_models(
        self, 
        critic_update_rate=None, 
        actor_update_rate=None,):
        
        if critic_update_rate is None:
            critic_update_rate = self.critic_update_rate
            
        if actor_update_rate is None:
            actor_update_rate = self.actor_update_rate
        """
        update rate is value between 0 and 1, 0 does not update, 1 changes weights completely to target weights
        """
        def update(current_val, update_val, update_rate):
            return current_val + (update_rate * (update_val - current_val))
        
        def update_all_weights(model, target_model, update_rate):
            all_model_weights = model.get_weights()
            all_target_weights = target_model.get_weights()
            new_weights = []
            for model_weights, target_weights in zip(all_model_weights, all_target_weights):
                new_weights.append(update(target_weights, model_weights, update_rate))
                
            target_model.set_weights(new_weights)
            
        for model, target_model, update_rate in [
            (self.actor, self.target_actor, actor_update_rate),
            (self.critic, self.target_critic, critic_update_rate),
            ]:
            update_all_weights(model, target_model, update_rate)
        
            
    
    
        
        
        
    def act(self, state, model=None, exploring=False):
        
        model = model or self.actor
        # single state as input
        # print("Acting singly {} ".format(state.shape))
        self.last_action = model.predict(state.reshape(1,self.state_size))
        # print("State: {} \n\n-- yielded --\n\n Action: {}".format(state, self.last_action))
#         self.last_action = self.act_many(
#             state[np.newaxis, :], 
#             model=model, exploring=exploring)[0, :]
        return self.last_action

        
    def act_many(self, states, model=None, exploring=False):
        model = model or self.actor
        
        actions = model.predict(states)
        
        if exploring and np.random.uniform() < self.epsilon:
            action_noise = np.random.uniform(size=states.shape)
            actions = actions + self.noise_ratio * (action_noise - actions)
#             print("Acting with random noise:")
#             print(self._scale_output_to_action(actions))
#         else:
#             print("Acting nonrandomly:")
#             print(self._scale_output_to_action(actions))
        
        return actions
       




    def step(self, reward, done):
        # Save experience / reward
        self.total_reward += reward
        self.count += 1

        # Learn, if at end of episode
        if done:
            self.learn()

    
    def _scale_output_to_action(self, output):
        return self.action_low + (output * self.action_range)
   
    
    
    
    
#     def create_dqn(self):
#         state = Input((self.state_size,))
#         layer = self._add_dense_layers(
#             inputs=state, neuron_counts=[16,32,16])
#         layer = Dense(self.action_size, activation='tanh')(layer)
#         output = Lambda(lambda x: (x+1)/2)(layer)
#         model = Model(inputs=state, outputs=output)
#         return model
    
    
    
    
    
    def learn_from_one(self, experience):
        state = experience["state"].reshape((1,self.state_size))
        action = experience["action"]
        reward = experience["reward"]
        next_state = experience["next_state"].reshape((1,self.state_size))
#         self.critic_learn(
#             next_state=next_state, 
#             reward=reward)
        
        # estimate discounted reward from critic's perspective, using target model
        
        # print("My next_state has a shape of {}.".format(next_state.shape))
        actor_target_action = self.act(state=next_state, model=self.target_actor)
        # print("My actor target action has a shape of {}.".format(actor_target_action.shape))
        critic_inputs = [next_state, actor_target_action]
        critic_target_next_reward = self.target_critic.predict(x=critic_inputs)
        
        discounted_reward = reward + critic_target_next_reward * self.gamma
        self.critic.fit(critic_inputs, discounted_reward, verbose=0)
        
        # get update gradients for actor based on target critic's estimates
        # of actor's action
        
        actor_action = self.act(state=state, model=self.actor)
        predicted_reward = self.critic_predict(
            state=state, action=actor_action, model=self.target_critic)
        actor_error = discounted_reward - predicted_reward
        actor_next_action = self.act(state=state, model=self.target_actor)
        self.actor.fit(state, [actor_error], verbose=0)
        
        
        
    def critic_predict_many(self, states, actions, model=None):
        model = model or self.critic
        actor_target_action = self.act_many(state=states, model=self.target_actor)
        critic_inputs = np.concatenate([states, actions])
        return model.predict(critic_inputs)
    
    def critic_predict(self, state, action, model=None):
        model = model or self.critic
        state = state.reshape((1,self.state_size))
        actor_target_action = self.act(state=state, model=self.target_actor)
        critic_inputs = [state, action]
        return model.predict(critic_inputs)
            
            
    def create_actor(
        self, 
        final_activation='tanh',
        initializer='truncated_normal',
        neuron_counts=[16,32,16], 
        dropout_rate=0.50,
        optimizer=Adam,
        lr=0.001,
        loss='mse',
        ):
        state = Input((self.state_size,))
        layer = self._add_dense_layers(
            state, neuron_counts=neuron_counts, 
            dropout_rate=dropout_rate,
            initializer=initializer)
                
        layer = Dense(self.action_size, activation=final_activation)(layer)
        output = Lambda(lambda x: (x+1)/2)(layer)
        actor = Model(inputs=state, outputs=output)
        opt = optimizer(lr=lr, clipnorm=self.clipnorm)
        actor.compile(loss=loss, optimizer=opt)
        actor._make_predict_function()
        return actor, state
        
        
    def create_critic(
        self, 
        final_activation='linear',
        initializer='truncated_normal',
        state_neuron_counts=[32,16], 
        
        action_neuron_counts=[24,16], 
        combined_neuron_counts=[16,24],
        dropout_rate=0.50,
        optimizer=Adam,
        lr=0.001,
        loss='mse',
    ):
        
        state = Input((self.state_size,))
        state_handle = Lambda(lambda x: x)
        state_out = self._add_dense_layers(
            inputs=state, 
            neuron_counts=state_neuron_counts, 
            dropout_rate=dropout_rate, 
            initializer=initializer)
                        
        action = Input((self.action_size,))
        action_out = self._add_dense_layers(
            inputs=state, 
            neuron_counts=action_neuron_counts, 
            dropout_rate=dropout_rate, 
            initializer=initializer)
        
        combined = Add()([state_out, action_out])
        combined_out =  self._add_dense_layers(
            inputs=combined, 
            neuron_counts=combined_neuron_counts, 
            dropout_rate=dropout_rate,
            initializer=initializer)
        
        output = Dense(self.action_size, activation=final_activation)(combined_out)
        
        critic = Model(inputs=[state, action], outputs=output)
        opt = optimizer(lr=lr, clipnorm=self.clipnorm)
        critic.compile(loss=loss, optimizer=opt)
        critic._make_predict_function()
        return critic, state_handle, action

    
    
    