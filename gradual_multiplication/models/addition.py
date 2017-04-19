from models.model import Model
from stacks.stacks import StackMachine
from stacks.datasets import MultidigitAddData


class AdditionModel(Model):
    """ The addition model for the ML Prague gradual arithmetic example"""
    def _setup_actions(self):
        return [StackMachine.Ops.push_temp,
                StackMachine.Ops.push,
                StackMachine.Ops.peek,
                StackMachine.Ops.add,
                StackMachine.Ops.pop,
                StackMachine.Ops.noop]

    def _setup_dataset(self, actions, train_size):
        return MultidigitAddData(actions, train_size)
