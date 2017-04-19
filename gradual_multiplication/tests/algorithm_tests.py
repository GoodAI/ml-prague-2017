import unittest
from stacks.datasets import MultidigitAddData, MultidigitMultiplyData, MultidigitMultiplyNonGradualData
from stacks.stacks import StackMachine, StackEnvironment, DataEntry


class StackMachineMock:
    def __init__(self):
        self.stacks = [[]]


class AlgorithmTests(unittest.TestCase):
    def test_addition(self):
        sequence = '3547'
        answer = '82'

        learned_addition = LearnedAdditionMock()

        stack_machine = StackMachineMock()
        stack_machine.stacks[0] = sequence.split()
        stack_idx = 0
        symbol = '='

        learned_addition(stack_machine, symbol, stack_idx)

        self.assertEqual(answer, learned_addition.environment.result(length=len(answer)))

    def test_multiplication(self):
        stack_size = 4
        learned_addition = LearnedAdditionMock()
        test_cases = [['089', '72'],
                      ['140', '00'],
                      ['610', '00'],
                      ['341', '34']]

        actions = [StackMachine.Ops.push,
                   StackMachine.Ops.pop,
                   StackMachine.Ops.push_temp,
                   StackMachine.Ops.peek,
                   StackMachine.Ops.dec,
                   StackMachine.Ops.noop,
                   learned_addition]

        dataset = MultidigitMultiplyData(actions, 1, learned_addition)

        pairs = [(sequence + '=' * (dataset.timesteps - 3), answer) for sequence, answer in test_cases]

        selections = dataset.generate_selections(pairs, dataset.timesteps, actions)

        environment = StackEnvironment(actions, dataset.stack_count, stack_size, reshuffle=False)
        environment.use_dataset([DataEntry(sequence, answer) for sequence, answer in pairs])

        for (_, result), sel in zip(pairs, selections):
            environment.reset()
            for action, stack in sel:
                environment.step(action, 1., stack, 1.)

            self.assertEqual(result, environment.result(length=2))

    def test_non_gradual_multiplication(self):
        timesteps = 178
        stack_count = 3
        stack_size = 5

        test_cases = [['089', '72'],
                      ['140', '00'],
                      ['610', '00'],
                      ['341', '34']]

        pairs = [(sequence + '=' * timesteps, answer) for sequence, answer in test_cases]

        actions = [StackMachine.Ops.push,
                   StackMachine.Ops.pop,
                   StackMachine.Ops.push_temp,
                   StackMachine.Ops.peek,
                   StackMachine.Ops.dec,
                   StackMachine.Ops.noop,
                   StackMachine.Ops.add]

        dataset = MultidigitMultiplyNonGradualData(actions, 1)

        selections = dataset.generate_selections(pairs, timesteps, actions)
        environment = StackEnvironment(actions, stack_count, stack_size, reshuffle=False)
        environment.use_dataset([DataEntry(sequence, answer) for sequence, answer in pairs])

        for (_, result), sel in zip(pairs, selections):
            environment.reset()
            for action, stack in sel:
                environment.step(action, 1., stack, 1.)

            self.assertEqual(result, environment.result(length=2))


class LearnedOperationMock(StackMachine.Ops.Classes.Op):
    def __init__(self, stack_size):
        self.stack_size = stack_size

        self.actions = self.setup_actions()
        self.dataset = self.setup_dataset()

        self.environment = StackEnvironment(self.actions, self.dataset.stack_count, self.stack_size)

    def __call__(self, stack_machine, symbol, stack_idx):
        sequence = ''.join(stack_machine.stacks[stack_idx])
        sequence += '=' * (self.dataset.timesteps - len(sequence))
        pair = [sequence, None]

        selections = self.dataset.generate_selections([pair], self.dataset.timesteps, self.actions)

        self.environment.use_dataset([DataEntry(pair[0], pair[1])])

        self.environment.reset()
        for action, stack in selections[0]:
            self.environment.step(action, 1., stack, 1.)

        stack_machine.stacks[stack_idx] = self.environment._stack.stacks[0]

    def setup_actions(self):
        raise NotImplementedError

    def setup_dataset(self):
        raise NotImplementedError


class LearnedAdditionMock(LearnedOperationMock):
    def __init__(self):
        super().__init__(stack_size=4)

    def setup_dataset(self):
        return MultidigitAddData(self.actions, 1)

    def setup_actions(self):
        return [StackMachine.Ops.push_temp,
                StackMachine.Ops.push,
                StackMachine.Ops.peek,
                StackMachine.Ops.add,
                StackMachine.Ops.pop,
                StackMachine.Ops.noop]



