from models.model import Model
from stacks.stacks import StackMachine
from stacks.datasets import MultidigitMultiplyData, MultidigitMultiplyNonGradualData


class MultiplicationModel(Model):
    """The gradual multiplication example for the ML Prague gradual arithmetic example."""
    def __init__(self, name, num_episodes, batch_size, input_size, stack_size, num_cells,
                 train_size, learned_addition, debug_interval=10, min_divisor=1, max_divisor=5):
        self.learned_addition = learned_addition

        super().__init__(name, num_episodes, batch_size, input_size, stack_size, num_cells, train_size,
                         debug_interval=debug_interval, min_divisor=min_divisor, max_divisor=max_divisor)

    def _setup_actions(self):
        return [StackMachine.Ops.push,
                StackMachine.Ops.pop,
                StackMachine.Ops.push_temp,
                StackMachine.Ops.peek,
                StackMachine.Ops.dec,
                StackMachine.Ops.noop,
                self.learned_addition]

    def _setup_dataset(self, actions, train_size):
        return MultidigitMultiplyData(actions, train_size, self.learned_addition)


class NonGradualModel(Model):
    """The non-gradual multiplication example for the ML Prague gradual arithmetic example."""
    def _setup_actions(self):
        return [StackMachine.Ops.push_temp,
                StackMachine.Ops.pop,
                StackMachine.Ops.push,
                StackMachine.Ops.dec,
                StackMachine.Ops.peek,
                StackMachine.Ops.add,
                StackMachine.Ops.noop]

    def _setup_dataset(self, actions, train_size):
        return MultidigitMultiplyNonGradualData(actions, train_size)
