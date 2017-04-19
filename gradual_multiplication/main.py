import argparse
import tensorflow as tf
from stacks.stacks import StackEnvironment, StackMachine
from models.plotting import plot_stats

from models.addition import AdditionModel
from models.multiplication import MultiplicationModel, NonGradualModel
import random
import numpy as np


def run_experiment(session, model, run, retrain):
    stats = None
    if run:
        # Check if addition is trained, if not, train.
        if retrain or not model.try_load(session):
            print("Training {}".format(model.name))
            stats = model.train(session)

        print("Testing {}".format(model.name))
        model.test(session)

    return stats


def main(args):
    run_addition = not args.skip_add
    run_multiplication = not args.skip_mult
    run_non_gradual = args.run_non_gradual

    if run_multiplication:
        run_addition = True

    config = tf.ConfigProto(device_count={'GPU': 0}) if not args.use_gpu else None

    with tf.Session(config=config) as session:
        # Initialize the addition model.
        addition_model = AdditionModel('addition', args.add_episodes, args.batch_size, args.input_size,
                                       args.stack_size, args.add_lstm_size, args.add_train_size)

        # Get the inference-only model for reuse in multiplication.
        addition_controller = addition_model.inference_agent

        # Create the environment for addition inference.
        addition_environment = StackEnvironment(addition_model.actions, addition_model.dataset.stack_count,
                                                args.stack_size)

        # Create a new operation which wraps the addition inference model.
        learned_addition = StackMachine.Ops.Classes.LearnedOp(addition_controller, addition_environment,
                                                              addition_model.dataset.timesteps, session)

        # Initialize the multiplication model.
        multiplication_model = MultiplicationModel('multiplication', args.mult_episodes,
                                                   args.batch_size, args.input_size,
                                                   args.stack_size, args.mult_lstm_size,
                                                   args.mult_train_size, learned_addition)

        # Initialize the non-gradual model.
        non_gradual_model = NonGradualModel('non_gradual_multiplication', args.non_gradual_episodes,
                                            args.batch_size,
                                            args.input_size, args.stack_size, args.non_gradual_lstm_size,
                                            args.mult_train_size)

        # Initialize all variables, might be overwritten by loaded ones.
        session.run(tf.global_variables_initializer())

        # Run addition if required and plot results.
        addition_stats = run_experiment(session, addition_model, run_addition, args.retrain_add)
        if addition_stats is not None:
            plot_stats(addition_model.name, [addition_stats])

        # Run multiplication if required.
        multiplication_stats = run_experiment(session, multiplication_model, run_multiplication, args.retrain_mult)

        # Run non-gradual multiplication if required.
        non_gradual_stats = run_experiment(session, non_gradual_model, run_non_gradual, args.retrain_non_gradual)

        if multiplication_stats is not None or non_gradual_stats is not None:
            plot_stats(multiplication_model.name, [multiplication_stats, non_gradual_stats])


if __name__ == '__main__':
    random.seed(12345)
    np.random.seed(12345)
    tf.set_random_seed(12345)
    parser = argparse.ArgumentParser(description="A program that gradually learns addition and then multiplication.")

    # Stack size is shared across all models.
    parser.add_argument('--stack-size', default=4, type=int)

    # Number of LSTM cells in the controllers.
    parser.add_argument('--add_lstm_size', default=100, type=int)
    parser.add_argument('--mult_lstm_size', default=100, type=int)
    parser.add_argument('--non_gradual_lstm_size', default=100, type=int)

    # Size of one symbol in one-hot format.
    parser.add_argument('--input_size', default=14, type=int)

    # Training and testing dataset sizes for addition.
    parser.add_argument('--add_train_size', default=2000, type=int)

    # Training and testing dataset sizes for multiplication.
    parser.add_argument('--mult_train_size', default=150, type=int)

    # Number of training episodes for both parts. One episode = one training example.
    parser.add_argument('--add_episodes', default=3000, type=int)
    parser.add_argument('--mult_episodes', default=7000, type=int)
    parser.add_argument('--non_gradual_episodes', default=10000, type=int)

    # Batch size is shared across all models.
    parser.add_argument('--batch_size', default=5, type=int)

    # Force a retrain of an algorithm (even if there's a saved model).
    parser.add_argument('--retrain_add', action='store_const', const=True)
    parser.add_argument('--retrain_mult', action='store_const', const=True)
    parser.add_argument('--retrain_non_gradual', action='store_const', const=True)

    # Use GPU - not recommended.
    parser.add_argument('--use_gpu', action='store_const', const=False)

    # Which parts of the model to run. Note that addition and multiplication are run by default and that addition will
    # always run if multiplication is running.
    parser.add_argument('--skip_add', action='store_const', const=True)
    parser.add_argument('--skip_mult', action='store_const', const=True)
    parser.add_argument('--run_non_gradual', action='store_const', const=True)

    parsed_args = parser.parse_args()

    main(parsed_args)
