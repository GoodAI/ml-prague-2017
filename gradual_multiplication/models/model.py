import math
import time

from collections import namedtuple

import numpy as np
import tensorflow as tf

from controller.controller import RecurrentAgent
from stacks.stacks import StackEnvironment, StackMachine, DataEntry

TrainingStats = namedtuple('TrainingStats', ['name', 'accuracy', 'labels'])


class Model:

    """The model encapsulates an example run of the gradual program learning agent.
    
    The subclasses of this model detail the the specific data generation methods and stack actions
    used in each example. The model initialises everything required to run an example and allows it
    to be trained and tested.
    """
    def __init__(self, name, num_episodes, batch_size, input_size, stack_size, num_cells, train_size, min_divisor=1,
                 max_divisor=5, debug_interval=10):
        self.name = name

        self.num_episodes = num_episodes
        self.batch_size = batch_size
        self.min_divisor = min_divisor
        self.max_divisor = max_divisor
        self.input_size = input_size
        self.stack_size = stack_size
        self.num_cells = num_cells

        self._debug_interval = debug_interval

        self.reward = StackMachine.Rewards.strong_supervision

        self.actions = self._setup_actions()

        self.dataset = self._setup_dataset(self.actions, train_size)
        self.dataset.generate_all()

        with tf.variable_scope(self.name+"_training") as scope:
            training_agent = self._setup_agent(batch_size=self.batch_size, scope=scope)
        with tf.variable_scope(self.name+"_inference") as scope:
            inference_agent = self._setup_agent(batch_size=1, scope=scope)

        self.training_agent = training_agent
        self.inference_agent = inference_agent

        self.environments = self._setup_environments()

    def try_load(self, session):
        return self.inference_agent.try_load(session)

    def _setup_actions(self):
        raise NotImplementedError

    def _setup_dataset(self, actions, train_size):
        raise NotImplementedError

    def _setup_agent(self, batch_size, scope):
        return RecurrentAgent(name=self.name, env_state_size=self.input_size, env_stack_size=self.stack_size,
                              num_actions=len(self.actions), hidden_size=self.num_cells,
                              num_stacks=self.dataset.stack_count, batch_size=batch_size, scope=scope)

    def _setup_environments(self):
        environments = []
        for k in range(self.batch_size):
            env = StackEnvironment(self.actions, self.dataset.stack_count, self.stack_size, self.reward, reshuffle=True)
            env.use_dataset(list(self.dataset.training_data))
            environments.append(env)

        return environments

    @staticmethod
    def _get_init_state(session, agent):
        init_state = session.run(agent.zero_state)
        return np.array([init_state[0], init_state[1]])

    def _get_divisors(self):
        action_divisors = np.zeros((self.dataset.timesteps, 1))
        action_divisors.fill(self.min_divisor)
        stack_divisors = np.zeros((self.dataset.timesteps, 1))
        stack_divisors.fill(self.min_divisor)

        return action_divisors, stack_divisors

    def _get_init_environment(self):
        inputs = []
        stacks = []
        for k in range(self.batch_size):
            input_symbol, _ = self.environments[k].reset()
            inputs.append(self.environments[k].encode_symbol(input_symbol))
            flattened_stacks = [symbol for stack in self.environments[k].encoded_stacks() for symbol in stack]
            stacks.append(np.reshape(np.array(flattened_stacks), -1))

        return inputs, stacks

    @staticmethod
    def _feed_forward_feed_dict(agent, input_symbols, input_stacks, state, action_divisors, stack_divisors):
        return {agent.env_state_input: input_symbols,
                agent.env_stack_input: input_stacks,
                agent.rnn_state_ph: state,
                agent.action_softmax_div: action_divisors,
                agent.stack_softmax_div: stack_divisors}

    def train(self, session):
        """ The basic outline of this training procedure was taken from:
        https://medium.com/@awjuliani/super-simple-reinforcement-learning-tutorial-part-2-ded33892c724 

        The main training procedure. This function trains the controller
        to execute the specified algorithm via a strong supervision method.

        At each timestep, the environment is run one step and the next input symbol and
        state of the stacks is determined, these inputs are passed into the controller,
        the instruction is calculated and the environment is run again for the next timestep.

        For each episode, a histroy of the inputs and outputs is saved. Each timestep is rewarded
        based upon correctness and how confident the controller is in the selection. Low confidence
        correct answers are rewarded more than high confidence correct answers. Incorrect selections are
        always punished with -1.

        When an episode has concluded, the history and rewards are fed into the controller as a
        mult-timestep batch, credit is apportioned and weights are updated.

        The divisors are softmax scalars used to encourage exploration in timesteps that are consistently
        punished by scaling the values by a factor which increases as the controller gets the selection wrong,
        thereby flattening the softmax distribution. As the controller makes the correct selection at some
        timestep, the divisor is reduced to a minimum level of 1 which removes its effect."""
        stats = TrainingStats(name=self.name, accuracy=[], labels=[])

        i = 0
        window_reward = []
        agent = self.training_agent
        init_state = self._get_init_state(session, agent)

        # We set up the divisors for all timesteps and adjust them as rewards come in
        action_divisors, stack_divisors = self._get_divisors()

        action_divisor_rewards = np.zeros(self.dataset.timesteps)
        stack_divisor_rewards = np.zeros(self.dataset.timesteps)

        start_time = epoch_start_time = time.time()
        r_time = []
        positive_reward = 0
        while i < self.num_episodes:
            inputs, stacks = self._get_init_environment()

            ep_history = []
            current_timestep = 0
            for k in range(self.batch_size):
                ep_history.append([])

            # Rewards for all examples in the batch. We use two rewards (action selection, stack selection).
            running_reward = [np.zeros(self.batch_size), np.zeros(self.batch_size)]

            episode_actions = np.zeros(len(self.actions))

            state = init_state

            while inputs[0] is not None:
                input_symbols = np.expand_dims(inputs, 1)
                input_stacks = np.expand_dims(stacks, 1)
                feed_forward_feed_dict = self._feed_forward_feed_dict(agent, input_symbols, input_stacks, state,
                                                                      [action_divisors[current_timestep]],
                                                                      [stack_divisors[current_timestep]])
                action_choice, stack_choice, new_state = session.run([agent.output, agent.out_stack,
                                                                      agent.hidden_state],
                                                                     feed_dict=feed_forward_feed_dict)

                inputs = []
                stacks = []
                action_choice = np.squeeze(action_choice, axis=1)
                stack_choice = np.squeeze(stack_choice, axis=1)
                for k in range(self.batch_size):
                    # Select a random action based upon the action distribution from the controller
                    a_c = action_choice[k]
                    s_c = stack_choice[k]
                    a_conf = np.random.choice(a_c, p=a_c)
                    a = np.argmax(a_c == a_conf)

                    s_conf = np.random.choice(s_c, p=s_c)
                    s = np.argmax(s_c == s_conf)

                    episode_actions[a] += 1
                    debug = (i + 1) % self._debug_interval == 0
                    action_debug = ' '.join("{:.2f}".format(f) for f in action_choice[0])
                    stack_debug = ' '.join("{:.2f}".format(f) for f in stack_choice[0])
                    if debug:
                        print("soft selections - action: {}, stack: {}".format(action_debug, stack_debug))

                    new_symbol, _, reward = self.environments[k].step(a, a_conf, s, s_conf, debug)

                    new_symbol = self.environments[k].encode_symbol(new_symbol)
                    flattened_stacks = [symbol for stack in self.environments[k].encoded_stacks() for symbol in stack]
                    stacks.append(np.reshape(np.array(flattened_stacks), -1))

                    # Write the input symbol, stack states, action, reward for action, reward for stacks, and stack
                    # to the history of the episode
                    ep_history[k].append([input_symbols[k][0], input_stacks[k][0], a, reward[0], reward[1], s])
                    running_reward[0][k] += reward[0]
                    running_reward[1][k] += reward[1]

                    positive_reward += 1 if reward[0] > 0 else 0
                    positive_reward += 1 if reward[1] > 0 else 0

                    inputs.append(new_symbol)
                    # Count the number of positive or negative rewards (not the values themselves)
                    action_divisor_rewards[current_timestep] += math.ceil(reward[0])
                    stack_divisor_rewards[current_timestep] += math.ceil(reward[1])

                state = np.array([new_state[0], new_state[1]])
                inputs = np.array(inputs)
                stacks = np.array(stacks)
                current_timestep += 1

                if inputs[0] is None:
                    # If it is the end of the episode, update the agent according to
                    # The rewards that it has accumulated throughout the episode
                    ep_history = np.array(ep_history)
                    in_hist = ep_history[:, :, 0]
                    stack_hist = ep_history[:, :, 1]
                    state_in = []
                    stack_in = []
                    # Stack the histories in n timesteps
                    for k in range(self.batch_size):
                        state_in.append(np.stack(in_hist[k], 0))
                        stack_in.append(np.stack(stack_hist[k], 0))

                    state_in = np.stack(state_in, 0)
                    stack_in = np.stack(stack_in, 0)

                    action_reward = ep_history[:, :, 3]
                    stack_reward = ep_history[:, :, 4]

                    # Feed in entire history of the episode with the rewards and divisors
                    feed_dict = {agent.action_reward_holder: action_reward,
                                 agent.stack_reward_holder: stack_reward,
                                 agent.action_holder: ep_history[:, :, 2],

                                 agent.env_state_input: state_in,
                                 agent.env_stack_input: stack_in,
                                 agent.stack_holder: ep_history[:, :, 5],

                                 agent.stack_softmax_div: stack_divisors,
                                 agent.action_softmax_div: action_divisors,

                                 agent.rnn_state_ph: init_state}

                    _ = session.run(agent.update, feed_dict=feed_dict)

                    # Here we update the divisors based upon the rewards accumulated
                    # If there were more positive rewards, we decrease the divisor towards min_divisor,
                    # If there were more negative rewards, we increase the divisor towards max_divisor
                    for k in range(self.dataset.timesteps):
                        stack_divisors[k][0] = np.clip(-np.tanh(stack_divisor_rewards[k]) + stack_divisors[k][0],
                                                       self.min_divisor, self.max_divisor)
                        action_divisors[k][0] = np.clip(-np.tanh(action_divisor_rewards[k]) + action_divisors[k][0],
                                                        self.min_divisor, self.max_divisor)

                    stack_divisor_rewards.fill(0)
                    action_divisor_rewards.fill(0)

                    window_reward.append(np.mean(running_reward))
                    break

            # Print debug information every so often...
            if (i + 1) % self._debug_interval == 0:
                duration = time.time() - epoch_start_time
                window_mean_reward = np.mean(window_reward)
                window_reward = []
                accuracy = positive_reward / (2 * self._debug_interval * self.dataset.timesteps * self.batch_size)
                stats.accuracy.append(accuracy)
                stats.labels.append(i+1)
                positive_reward = 0

                print("Episode {}: Duration: {:.2f}s, Mean reward of prev {}: {}".format(
                    i, duration, self._debug_interval, window_mean_reward))

                r_time.append(duration)
                print("Est Time Remaining: {:.2f}s".format(np.mean(np.array(r_time)) *
                                                           ((self.num_episodes-i+1)/self._debug_interval)))
                epoch_start_time = time.time()

                for (action, num) in zip(self.actions, episode_actions):
                    print("Action: {} Count: {}".format(action, num))

                episode_actions.fill(0)
            i += 1

        print("Total training time: ", time.time() - start_time)

        agent.training_finished(session)
        self.inference_agent.apply_variables(session, agent.trained_variables)
        self.inference_agent.save(session)

        return stats

    def run_single(self, session, sequence, answer):
        sequence = sequence + '=' * (self.dataset.timesteps - len(sequence))

        self.dataset.testing_data = [DataEntry(sequence, answer)]
        self.test(session, debug=True)

    def test(self, session, debug=False):
        num_oks = 0

        agent = self.inference_agent

        init_state = self._get_init_state(session, agent)

        environment = self.environments[0]
        environment.random = False
        environment.use_dataset(self.dataset.testing_data)

        for i in range(len(self.dataset.testing_data)):
            state = init_state
            symbol, _ = environment.reset()

            symbol = environment.encode_symbol(symbol)
            flattened_stacks = [symbol for stack in environment.encoded_stacks() for symbol in stack]
            shaped_stacks = np.reshape(np.array(flattened_stacks), -1)

            while symbol is not None:
                symbol = np.expand_dims([symbol], 1)
                stacks_step = np.expand_dims([shaped_stacks], 1)
                a_select, s_select, state = session.run([agent.chosen_action, agent.chosen_stack, agent.hidden_state],
                                                        feed_dict={agent.env_state_input: symbol,
                                                                   agent.env_stack_input: stacks_step,
                                                                   agent.rnn_state_ph: state,
                                                                   agent.action_softmax_div: [[1]],
                                                                   agent.stack_softmax_div: [[1]]})

                a_conf = s_conf = 1.

                symbol, _, _ = environment.step(np.squeeze(a_select), a_conf, np.squeeze(s_select), s_conf)

                symbol = environment.encode_symbol(symbol)
                flattened_stacks = [symbol for stack in environment.encoded_stacks() for symbol in stack]
                shaped_stacks = np.reshape(np.array(flattened_stacks), -1)

            data_entry = environment.current_data_entry
            result = environment.result(length=len(data_entry.answer))

            if debug:
                print('Sequence: {}, correct answer: {}, model answer: {}'.format(data_entry.sequence,
                                                                                  data_entry.answer, result))
                return

            if result == data_entry.answer:
                num_oks += 1
            else:
                print('sequence: {}, correct answer: {}, model answer: {}'.format(data_entry.sequence,
                                                                                  data_entry.answer, result))
                print("INCORRECT")

        print("Correct: {}/{}".format(num_oks, len(self.dataset.testing_data)))
