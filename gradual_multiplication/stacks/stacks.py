import numpy as np
from functools import wraps
import random


class Alphabet:
    def __init__(self, non_digit_symbols='+-=', error_symbol='E'):
        self.digit_symbols = list(''.join(str(i) for i in range(10)))
        self.non_digit_symbols = list(non_digit_symbols + error_symbol)
        self.error_symbol = error_symbol

        self.all_symbols = self.digit_symbols + self.non_digit_symbols


class DataEntry:
    def __init__(self, sequence, answer=None, selections=None):
        self.sequence = sequence
        self.answer = answer
        self.selections = selections


def check_dataset(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if self._dataset is None:
            raise RuntimeError("Call use_dataset() first")

        return method(self, *args, **kwargs)

    return wrapper


class StackEnvironment:
    """ The stack environment is a middleman between the training/inference methods of the
         model and the stack machine. The environment encapsulates a running stack machine, brokers
         information between the machine and model, and supplies the input symbols to the model.
        """

    def __init__(self, actions, stack_count, stack_size, reward_function=None, alphabet=None, dtype=np.float32,
                 reshuffle=True):

        self._alphabet = alphabet or Alphabet()
        self._stack = StackMachine(actions, stack_count, stack_size, self._alphabet)
        self._action_count = len(actions)
        self._stack_count = stack_count
        self._reward_function = reward_function
        self._dtype = dtype
        self._stack_size = stack_size
        self._reshuffle = reshuffle

        self._dataset = None

        self._timestep = -1
        self._data_entry_idx = -1
        self.current_sequence = None
        self.current_data_entry = None

    @check_dataset
    def reset(self):
        self._timestep = 0
        self._stack.reset()

        self._data_entry_idx += 1
        if self._data_entry_idx >= len(self._dataset):
            self.reset_dataset()
            self._data_entry_idx += 1

        self.current_data_entry = self._dataset[self._data_entry_idx]
        self.current_sequence = self.current_data_entry.sequence

        return self._current_symbol(), self._stack.stacks

    def use_dataset(self, dataset):
        self._dataset = dataset
        self.reset_dataset()

    @check_dataset
    def reset_dataset(self):
        self._data_entry_idx = -1
        if self._reshuffle:
            random.shuffle(self._dataset)

    @check_dataset
    def step(self, action_selection, action_confidence, stack_selection, stack_confidence, debug=False):
        if self._timestep >= len(self.current_sequence):
            raise RuntimeError("Sequence end reached, call reset()")

        self._stack.step(self.current_sequence[self._timestep], action_selection, stack_selection)

        if self._reward_function is None:
            reward = 0
        else:
            reward = self._reward_function(self._timestep, self.current_data_entry, self._stack, action_selection,
                                           action_confidence, stack_selection, stack_confidence)

        self._timestep += 1

        current_symbol = None

        # Check for end
        if self._timestep < len(self.current_sequence):
            current_symbol = self._current_symbol()

        if debug:
            print('SEQ: {}, T: {:2}, AS: {}, SS: {}, AR: {:.5f}, SR: {:.5f}, MEM: {}'.format(
                self.current_sequence, self._timestep, action_selection, stack_selection, reward[0], reward[1],
                self._stack.stacks))

        return current_symbol, self._stack.stacks, reward

    def dataset_size(self):
        return len(self._dataset)

    def result(self, length=1):
        return ''.join(list(self._stack.stacks[0])[-length:])

    def _current_symbol(self):
        return self.current_sequence[self._timestep]

    def encode_symbol(self, symbol):
        if symbol is None:
            return None

        symbol_size = len(self._alphabet.all_symbols)
        result = [0.] * symbol_size
        result[self._alphabet.all_symbols.index(symbol)] = 1.

        return np.array(result, dtype=self._dtype)

    def decode_symbol(self, encoded_symbol):
        return self._alphabet.all_symbols[np.argmax(encoded_symbol)]

    def encoded_stacks(self):
        return [self.encode_stack(stack) for stack in self._stack.stacks]

    def encode_stack(self, stack):
        return [self.encode_symbol(symbol) for symbol, in stack]

    def encode_temp(self):
        return self.encode_symbol(self._stack.temp)

    def encode_history(self):
        return [self.encode_action(action) if action is not None
                else np.zeros(self._action_count) for action in self._stack.history]

    def encode_action(self, action):
        a = np.zeros(self._action_count)
        a[action] = 1
        return a


class StackMachine:
    """ The stack machine is a rudimentary multi stack model of computation
        The stack machine takes input in the form of (symbol, action_id, stack_id)
        and applied the action and symbol (if applicable) to the indicated stack.

        The stack machine also has a temporary register to which popped/peeked values are
        written, as well as the overflow for addition.

        Actions available to the stack machine include:
        * Push(sy, st) - push the symbol sy to stack st
        * Push_temp(st) - push the value in the temporary register to stack st
        * Pop(st) - remove the top symbol of stack st and write it to the temporary register
        * Peek(st) - copy the top symbol of stack st to the temporary register
        * Dec(st) - pop the symbol of stack st, decrement it, then push it back. If the symbol is 0
            or non numeric, push the error symbol (E) instead
        * Add(st) - pop the top two values of stack st, add them together, write the 'tens' digit to
            the temporary register, push the 'units' digit to st. If one of the symbols in not numeric,
            push the error symbol (E) to st instead.
        * noop - do nothing

        """

    def __init__(self, actions, stack_count, stack_size, alphabet, history_length=9):
        self.stack_count = stack_count
        self.stack_size = stack_size
        self.stacks = []
        self.temp = '0'
        self.alphabet = alphabet
        self.history_length = history_length

        self.reset()

        self._actions = []
        for action in actions:
            self._actions.append(action)

        self.history = [None] * history_length

    def step(self, symbol, action_idx, stack_idx):
        self._actions[action_idx](self, symbol, stack_idx)
        self.history = self.history[1:]
        self.history.append(action_idx)

    def reset(self):
        self.stacks = [['0'] * self.stack_size for _ in range(self.stack_count)]
        self.temp = '0'
        self.history = [None] * self.history_length

    def get_actions(self):
        return self._actions

    def push(self, symbol, stack_idx):
        stack = self.stacks[stack_idx]
        stack.append(symbol)
        trim = len(stack) - self.stack_size
        self.stacks[stack_idx] = self.stacks[stack_idx][trim:]

    def pop(self, stack_idx):
        stack = self.stacks[stack_idx]
        self.temp = stack.pop()
        padding = ['0'] * (self.stack_size - len(stack))
        self.stacks[stack_idx] = padding + stack
        return self.temp

    def peek(self, stack_idx):
        self.temp = self.stacks[stack_idx][-1]
        return self.temp

    class Ops:
        class Classes:
            class Op:
                def __call__(self, stack_machine, symbol, stack_idx):
                    raise NotImplementedError

            class BinaryArithmeticOp(Op):
                def __init__(self, pop_args=False):
                    self._pop_args = pop_args

                def __call__(self, stack_machine, symbol, stack_idx):
                    stack = stack_machine.stacks[stack_idx]
                    a = stack[-2]
                    b = stack[-1]
                    is_a_digit = a in stack_machine.alphabet.digit_symbols
                    is_b_digit = b in stack_machine.alphabet.digit_symbols
                    if not is_a_digit or not is_b_digit:
                        stack_machine.push(stack_machine.alphabet.error_symbol, stack_idx)
                    else:
                        if self._pop_args:
                            stack_machine.pop(stack_idx)
                            stack_machine.pop(stack_idx)

                        carry, ans = self.op(int(a), int(b))
                        stack_machine.push(str(ans), stack_idx)
                        stack_machine.temp = str(carry)

                def op(self, a, b):
                    raise NotImplementedError

            class AddOp(BinaryArithmeticOp):
                def __init__(self, pop_args=True):
                    super().__init__(pop_args)

                def op(self, a, b):
                    ans = a + b
                    if ans < 10:
                        return 0, ans
                    else:
                        return 1, ans - 10

            class SubOp(BinaryArithmeticOp):
                def __init__(self):
                    super().__init__(pop_args=False)

                def op(self, a, b):
                    return 0, max(a - b, 0)

            class LearnedOp(Op):
                def __init__(self, controller, environment, timesteps, session):
                    self._controller = controller
                    self._environment = environment
                    self._session = session
                    self._timesteps = timesteps
                    self._zero_state = self._session.run(self._controller.get_zero_state())

                def __call__(self, stack_machine, symbol, stack_idx):
                    """ Perform the operation.
                        
                    :param stack_machine: a StackMachine instance containing the stacks of the top-level controller
                    :param symbol: the current symbol (hint: for correctly called addition it's likely to be '=')
                    :param stack_idx: index of the chosen stack
                    """
                    # TODO: Prepare input for the learned model.
                    # stack_machine.stacks is a cool place, check it out
                    # The sequence should have length of self._timesteps
                    # '=' is the padding symbol

                    # sequence should contain the converted problem
                    result_stacks = self.process(sequence)

                    # TODO: Put the result back to the stack machine.
                    # Remember that the first stack contains the output.

                def process(self, sequence):
                    dataset = [DataEntry(sequence)]
                    self._environment.use_dataset(dataset)

                    # Reset the environment and get initial symbol and stacks (not encoded).
                    symbol, stacks = self._environment.reset()

                    # Initial state of the LSTM network.
                    state = self._zero_state
                    state = np.array([state[0], state[1]])

                    # Encode symbol -> one-hot.
                    symbol = self._environment.encode_symbol(symbol)

                    # Encode all the stacks, flatten and reshape them for the controller.
                    flattened_stacks = [symbol for stack in self._environment.encoded_stacks() for symbol in stack]
                    shaped_stacks = np.reshape(np.array(flattened_stacks), -1)

                    # Go through the whole sequence.
                    while symbol is not None:
                        # Adjust dimensions for the tensorflow model.
                        symbol_in = np.expand_dims([symbol], 1)
                        stacks_in = np.expand_dims([shaped_stacks], 1)

                        # Prepare the feed_dict - input symbol and stacks, state, softmax divisors.
                        feed_dict = {self._controller.env_state_input: symbol_in,
                                     self._controller.env_stack_input: stacks_in,
                                     self._controller.rnn_state_ph: state,
                                     self._controller.action_softmax_div: [[1]],
                                     self._controller.stack_softmax_div: [[1]]}

                        # Run the forward pass.
                        a_select, s_select, state = self._session.run([self._controller.chosen_action,
                                                                       self._controller.chosen_stack,
                                                                       self._controller.hidden_state],
                                                                      feed_dict=feed_dict)

                        # Confidence is only relevant for reward, 100% can be used here.
                        a_conf = s_conf = 1.

                        # Step the environment and retrieve next symbol and stacks.
                        symbol, stacks, _ = self._environment.step(np.squeeze(a_select), a_conf, np.squeeze(s_select),
                                                                   s_conf)

                        # Encode symbol and stacks into one-hot.
                        symbol = self._environment.encode_symbol(symbol)
                        flattened_stacks = [symbol for stack in self._environment.encoded_stacks() for symbol in stack]
                        shaped_stacks = np.reshape(np.array(flattened_stacks), -1)

                    # Return all the stacks of the environment.
                    return stacks

        @staticmethod
        def noop(stack_machine, symbol, stack_idx):
            pass

        @staticmethod
        def push(stack_machine, symbol, stack_idx):
            stack_machine.push(symbol, stack_idx)

        @staticmethod
        def push_temp(stack_machine, symbol, stack_idx):
            stack_machine.push(stack_machine.temp, stack_idx)

        @staticmethod
        def pop(stack_machine, symbol, stack_idx):
            stack_machine.pop(stack_idx)

        @staticmethod
        def peek(stack_machine, symbol, stack_idx):
            stack_machine.peek(stack_idx)

        add = Classes.AddOp()
        sub = Classes.SubOp()

        @staticmethod
        def dec(stack_machine, symbol, stack_idx):
            symbol = stack_machine.pop(stack_idx)
            new_symbol = stack_machine.alphabet.error_symbol
            if symbol in stack_machine.alphabet.digit_symbols:
                new_digit = int(symbol) - 1
                if new_digit >= 0:
                    new_symbol = str(new_digit)

            stack_machine.push(new_symbol, stack_idx)

    class Rewards:

        @staticmethod
        def strong_supervision(timestep, data_entry, stack_machine, action_idx, action_confidence,
                               stack_idx, stack_confidence):
            """ Gives reward based on whether the correct action and stack were selected.
            
            For a wrong selection, -1.0 is given. For a correct selection, the reward is linearly scaled based on
            confidence: 1-confidence, with 0.01 as lower bound.
            """

            if data_entry.selections is None:
                return 0

            desired_action, desired_stack = data_entry.selections[timestep]
            positive = 1
            negative = -1
            action_reward = max(positive - action_confidence, 0.01) if action_idx == desired_action else negative
            stack_reward = max(positive - stack_confidence, 0.01) if stack_idx == desired_stack else negative

            if timestep == 6 and action_reward > 0:
                pass

            return action_reward, stack_reward
