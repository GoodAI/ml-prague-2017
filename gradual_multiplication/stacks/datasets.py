from collections import defaultdict

import numpy as np
import random

from stacks.stacks import StackMachine, DataEntry


class Data:
    """ Base class for a dataset.
    
    It provides the training and testing data and parameters for models based on the learned algorithm.
    """
    def __init__(self, timesteps, stack_count, actions, train_size):
        self.timesteps = timesteps
        self.stack_count = stack_count
        self.actions = actions
        self.train_size = train_size

        self.training_data = None
        self.all_data = None
        self.testing_data = None

    def generate_all(self):
        all_data = self.generate_data(self.timesteps)
        all_selections = self.generate_selections(all_data, self.timesteps, self.actions)

        data_entries = []
        for (sequence, answer), selections in zip(all_data, all_selections):
            data_entries.append(DataEntry(sequence, answer, selections))

        random.shuffle(data_entries)

        self.training_data = data_entries[:self.train_size]
        # Tests are done on both training and testing data.
        self.testing_data = data_entries[self.train_size:]
        self.all_data = data_entries

    def generate_data(self, timesteps, normalize=False):
        """ Generate data (symbol sequences) for the learning algorithm.
        
        :param timesteps: Total number of timesteps. The sequences should be padded to have length equal to this.
        :param normalize: If true, the dataset is to be normalized - examples that are too frequent should be filtered.
        :return: Pairs of strings [[input_sequence, result], ...].
        """
        raise NotImplementedError

    def generate_selections(self, pairs, timesteps, actions):
        """ Generate action and stack selections for the given sequences using actions.
        
        :param pairs: Pairs of strings [[input_sequence, result], ...] retrieved from self._generate_data.
        :param timesteps: Total number of timesteps per sequence. The algorithm should be of the same length.
        :param actions: Actions to generate the algorithm from.
        :return: A list of pairs of indices for each pair: [[[action_select, stack_select], ...], ...]
        """
        raise NotImplementedError

    @staticmethod
    def _action_selection(desired_action, actions):
        """ A convenience method for generating of action and stack selection.
        
        See MultidigitAddData._generate_selections() for an example.
        """
        def selection(stack_select):
            return actions.index(desired_action), stack_select

        return selection


class MultidigitAddData(Data):
    """ Double-digit addition data and algorithm for results up to 99 (inclusive).
    
    For AB+CD=EF, the returned pair would have the following format: ['ABCD================', 'EF'].
    """
    def __init__(self, actions, train_size):
        timesteps = 18
        stack_count = 3
        super().__init__(timesteps, stack_count, actions, train_size)

    def generate_data(self, timesteps, normalize=False):
        inputs = []
        for i in range(10):
            for j in range(10):
                for x in range(10):
                    for y in range(10):
                        ans = i*10+j + x*10+y
                        if ans < 100:
                            if ans < 10:
                                ans = '0' + str(ans)
                            else:
                                ans = str(ans)

                            p = [str(i)+str(j)+str(x)+str(y) + '=' * (timesteps-4), ans]
                            inputs.append(p)

        return inputs

    def generate_selections(self, pairs, timestep_count, actions):
        push = self._action_selection(StackMachine.Ops.push, actions)
        pop = self._action_selection(StackMachine.Ops.pop, actions)
        push_temp = self._action_selection(StackMachine.Ops.push_temp, actions)
        peek = self._action_selection(StackMachine.Ops.peek, actions)
        add = self._action_selection(StackMachine.Ops.add, actions)
        noop = self._action_selection(StackMachine.Ops.noop, actions)

        all_selections = []

        selections = [push(0), push(0), push(1), push(2),  # Push the whole number 1 into stack 0, distribute number 2.
                      pop(0), push_temp(2),  # Move the lower-order digit of number 1 to stack 2.
                      peek(0), push_temp(1),  # Copy the higher-order digit of number 1 to stack 1.
                      peek(2), push_temp(0),  # Copy the lower-order digit of number 1 back to stack 1.
                      add(1), add(2),  # Add the higher order digits and then the lower order digits.
                      push_temp(1), add(1),  # Copy the carry to stack 1 and add it.
                      pop(1), push_temp(0),  # Copy the higher order digit to stack 0.
                      pop(2), push_temp(0)]  # Copy the lower order digit to stack 0.

        while len(selections) < timestep_count:
            selections.append(noop(0))

        for _ in pairs:
            all_selections.append(np.array(selections))

        return all_selections


class MultidigitMultiplyDataBase(Data):
    """ A base class inherited by gradual and non-gradual multiplication.
    
    The data is shared, but the algorithms are different.
    """

    def __init__(self, timesteps, actions, train_size):
        stack_count = 3
        super().__init__(timesteps, stack_count, actions, train_size)

    def generate_data(self, timesteps, normalize=False):
        candidates = []
        for x in range(100):
            for y in range(10):
                answer = x * y
                if answer < 100:
                    candidates.append((x, y, answer))

        if normalize:
            buckets = defaultdict(list)
            for candidate in candidates:
                buckets[candidate[1]].append(candidate)

            selection_size = len(buckets[9]) + len(buckets[2])

            candidates = []
            for key, bucket in buckets.items():
                if key == 0 or key == 1:
                    for i in range(selection_size):
                        candidates.append(bucket.pop(np.random.randint(0, len(bucket))))
                else:
                    candidates.extend(bucket)

        inputs = []
        for x, y, answer in candidates:
            sequence = str(x).zfill(2) + str(y) + '=' * (timesteps-3)
            inputs.append([sequence, str(answer).zfill(2)])

        return inputs

    def generate_selections(self, pairs, timesteps, actions):
        raise NotImplementedError


class MultidigitMultiplyData(MultidigitMultiplyDataBase):
    """ Data and algorithm for double-digit gradual multiplication using a learned addition operation."""
    def __init__(self, actions, train_size, addition_op):
        self.addition_op = addition_op
        timesteps = 30
        super().__init__(timesteps, actions, train_size)

    def generate_selections(self, pairs, timesteps, actions):
        push = self._action_selection(StackMachine.Ops.push, actions)
        pop = self._action_selection(StackMachine.Ops.pop, actions)
        push_temp = self._action_selection(StackMachine.Ops.push_temp, actions)
        peek = self._action_selection(StackMachine.Ops.peek, actions)
        dec = self._action_selection(StackMachine.Ops.dec, actions)
        noop = self._action_selection(StackMachine.Ops.noop, actions)
        add = self._action_selection(self.addition_op, actions)

        all_selections = []
        for sequence in pairs:
            b = int(sequence[0][2:3])

            selections = [push(0), push(0), push(1)]

            selections.append(pop(0))
            selections.append(push_temp(1))
            selections.append(push_temp(2))
            selections.append(peek(0))
            selections.append(push_temp(1))
            selections.append(pop(2))
            selections.append(push_temp(0))

            selections.append(pop(1))
            selections.append(push_temp(0))
            selections.append(pop(1))
            selections.append(push_temp(0))

            while b > 1:
                selections.append(add(0))
                selections.append(dec(1))
                b -= 1

            if b == 0:
                selections.append(pop(0))
                selections.append(pop(0))
                selections.append(pop(0))
                selections.append(pop(0))

            if len(selections) > timesteps:
                print(len(selections))
                print(sequence)
                print(timesteps)
                raise Exception("There are fewer allocated timesteps than the maximum length algorithm.")

            while len(selections) < timesteps:
                selections.append(noop(0))

            all_selections.append(np.array(selections))

        return np.array(all_selections)


class MultidigitMultiplyNonGradualData(MultidigitMultiplyDataBase):
    """ Data and algorithm for double-digit non-gradual multiplication"""
    def __init__(self, actions, train_size):
        timesteps = 178
        super().__init__(timesteps, actions, train_size)

    def generate_selections(self, pairs, timesteps, actions):
        push = self._action_selection(StackMachine.Ops.push, actions)
        pop = self._action_selection(StackMachine.Ops.pop, actions)
        push_temp = self._action_selection(StackMachine.Ops.push_temp, actions)
        peek = self._action_selection(StackMachine.Ops.peek, actions)
        dec = self._action_selection(StackMachine.Ops.dec, actions)
        noop = self._action_selection(StackMachine.Ops.noop, actions)
        add = self._action_selection(StackMachine.Ops.add, actions)

        # 20 instructions
        add_sel = [pop(2), push_temp(1), peek(2), push_temp(0), peek(1), push_temp(2), pop(0), push_temp(2),
                   pop(0), push_temp(2), pop(0), push_temp(1), add(1), add(2), push_temp(1), add(1), pop(1),
                   push_temp(0), pop(2), push_temp(0)]

        all_selections = []

        for sequence in pairs:
            b = int(sequence[0][2:3])

            # Push the 4 digits.
            selections = [push(0), push(0), push(1)]

            selections.append(pop(0))
            selections.append(push_temp(1))
            selections.append(push_temp(2))
            selections.append(peek(0))
            selections.append(push_temp(2))
            selections.append(pop(1))
            selections.append(push_temp(0))

            while b > 1:
                selections = selections + add_sel
                selections.append(dec(1))
                b -= 1

            if b == 0:
                # If the second operand is zero, just output it and finish.
                selections.append(pop(0))
                selections.append(pop(0))

            if len(selections) > timesteps:
                print(len(selections))
                print(sequence)
                raise Exception("There are fewer allocated timesteps than the maximum length algorithm.")

            while len(selections) < timesteps:
                selections.append(noop(0))

            all_selections.append(np.array(selections))

        return np.array(all_selections)
