import os
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.python.ops import rnn
from tensorflow.contrib.rnn import core_rnn_cell, LSTMStateTuple
from tensorflow.python.framework.errors import NotFoundError, InvalidArgumentError


class RecurrentAgent:
    """The controller for learning algorithms in a gradual manner.
    This network is an LSTM with a specified number of cells feeding into two softmax selectors in order to
    produce distributions for action and stack selections at a particular timestep.
    
    At each timestep, the controller is fed the state of the stack machine and the input symbol in the forms
    (batchsize, n_timesteps, input_size) where n_timesteps = 1. When the episode is completed, the computation
    history of the stacks is fed in as input (n_timesteps = n), the selected action indices and corresponding rewards
    are also fed in via the training placeholders. For each member of the batch, the responsible outputs are determined,
    rewarded/punished accordingly, and losses calculated. These losses are summed, gradients obtained, and weights
    updated accordingly.
    """
    def __init__(self, name, env_state_size, env_stack_size, num_actions, hidden_size, num_stacks, batch_size, scope):
        self.name = name

        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.lstm = core_rnn_cell.LSTMCell(num_units=hidden_size)
        self.zero_state = self.lstm.zero_state(batch_size=self.batch_size, dtype=tf.float32)

        self.rnn_state_ph = tf.placeholder(shape=[2, batch_size, hidden_size], dtype=tf.float32)
        self.rnn_state = self.to_lstm_state(self.rnn_state_ph)

        self.env_state_input = tf.placeholder(shape=[batch_size, None, env_state_size], dtype=tf.float32)
        self.env_stack_input = tf.placeholder(shape=[batch_size, None, num_stacks*env_stack_size*env_state_size],
                                              dtype=tf.float32)
        self.action_softmax_div = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        self.stack_softmax_div = tf.placeholder(shape=[None, 1], dtype=tf.float32)

        self.env_input = tf.concat((self.env_state_input, self.env_stack_input), axis=2)

        hidden, self.hidden_state = rnn.dynamic_rnn(self.lstm, self.env_input, initial_state=self.rnn_state)
        out_act = slim.fully_connected(hidden, num_actions, activation_fn=None, biases_initializer=None)
        self.output = tf.nn.softmax(out_act/self.action_softmax_div)
        self.chosen_action = tf.argmax(self.output, axis=2)

        out_sta = slim.fully_connected(hidden, num_stacks, activation_fn=None, biases_initializer=None)
        self.out_stack = tf.nn.softmax(out_sta/self.stack_softmax_div)
        self.chosen_stack = tf.argmax(self.out_stack, axis=2)

        # The next six lines establish the training procedure. We feed the reward and chosen action into the network
        # to compute the loss, and use it to update the network.
        self.action_reward_holder = tf.placeholder(shape=[batch_size, None], dtype=tf.float32)
        self.stack_reward_holder = tf.placeholder(shape=[batch_size, None], dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[batch_size, None], dtype=tf.int32)
        self.stack_holder = tf.placeholder(shape=[batch_size, None], dtype=tf.int32)

        loss_actions, loss_stacks = tf.scan(self.determine_input_seq, elems=[self.output, self.action_holder,
                                                                             self.out_stack, self.stack_holder,
                                                                             self.action_reward_holder,
                                                                             self.stack_reward_holder],
                                            initializer=(tf.constant(0.0), tf.constant(0.0)))

        self.loss = loss_actions+loss_stacks

        self.trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)
        self.gradients = tf.gradients(self.loss, self.trainable_variables)

        optimizer = tf.train.AdamOptimizer()
        self.update = optimizer.apply_gradients(zip(self.gradients, self.trainable_variables))

        self.trained_variables = []

        self._saver = tf.train.Saver(self.trainable_variables)

    @staticmethod
    def determine_input_seq(_, elements):
        output = elements[0]
        action_holder = elements[1]
        out_stack = elements[2]
        stack_holder = elements[3]
        action_reward = elements[4]
        stack_reward = elements[5]

        indexes_actions = tf.range(0, tf.shape(output)[0]) * tf.shape(output)[1] + action_holder
        indexes_stacks = tf.range(0, tf.shape(out_stack)[0]) * tf.shape(out_stack)[1] + stack_holder

        responsible_outputs_actions = tf.gather(tf.reshape(output, [-1]), indexes_actions)
        responsible_outputs_stacks = tf.gather(tf.reshape(out_stack, [-1]), indexes_stacks)

        loss_actions = -tf.reduce_mean(tf.log(responsible_outputs_actions) * action_reward)
        loss_stacks = -tf.reduce_mean(tf.log(responsible_outputs_stacks) * stack_reward)

        return loss_actions, loss_stacks

    @staticmethod
    def to_lstm_state(tensor):
        return LSTMStateTuple(tensor[0], tensor[1])

    def try_load(self, session):
        try:
            self._saver.restore(session, self._get_model_path())
            self._save_trained_variables(session)
            return True
        except NotFoundError:
            print("Saved model not found")
        except InvalidArgumentError:
            print("Saved variables in wrong format")

        return False

    def save(self, session):
        self._saver.save(session, self._get_model_path())

    def training_finished(self, session):
        self._save_trained_variables(session)

    def apply_variables(self, session, trained_variables):
        self.trained_variables = trained_variables
        ops = [tf.assign(variable, value) for variable, value in zip(self.trainable_variables, trained_variables)]
        session.run(ops)

    def _save_trained_variables(self, session):
        self.trained_variables = []
        keys = self.trainable_variables
        for value in session.run(keys):
            self.trained_variables.append(value)

    def _get_model_path(self):
        return os.path.join("saved_models", "Model_trained-{}".format(self.name))

    def _restore(self, session):
        self._saver.restore(session, self._get_model_path())

    def get_zero_state(self):
        return self.zero_state


