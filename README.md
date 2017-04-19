# Gentle introduction to Gradual Learning - ML Prague 2017

This repository contains the code for Machine Learning Prague 2017 workshop on Gradual Learning.

## Requirements

You don't need a GPU and an laptop CPU should be able to run our models.

The code is written in python 3.5 and requires the following packages:

- TensorFlow (version 1.0+)
- matplotlib (version 1.5+)

### Environment setup - Windows

The full WinPython package already contains everything you need. After installation, you can run the experiments using **WinPython Command Prompt**.

### Environment setup - Ubuntu

Install the python packages first:

`sudo apt install python3.5 python3-pip python3-tk`

Then, install pip packages the repository folder:

`pip install -r requirements.txt`

## Contents

There are two examples that we will look at. A full description of the models is included in the workshop presentation.

### Gradual multiplication

The **gradual_multiplication** folder contains an experiment that learns double-digit addition using intrinsic operations and then uses this as an additional operation to learn double-by-single-digit multiplication.

The runnable script is main.py.

The addition experiment is ready, but for the multiplication, there is a missing part in the `LearnedOp.__call__` method in **stacks.py:277**. Data from the calling model's stack needs to be converted and passed into the environment for the called model, and the results need to be passed back into the calling stack machine.
