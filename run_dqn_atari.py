import argparse
import gym
from gym import wrappers
import os.path as osp
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dqn
from dqn_utils import *
from atari_wrappers import *


def _cnn_to_linear(seq, input_shape=None):
    # From https://github.com/rarilurelo/pytorch_a3c
    if isinstance(input_shape, tuple):
        input_shape = list(input_shape)
    if input_shape is None:
        assert False, 'input_shape must be determined'
    for cnn in seq:
        if not isinstance(cnn, nn.Conv2d):
            continue
        kernel_size = cnn.kernel_size
        stride = cnn.stride
        for i, l in enumerate(input_shape):
            input_shape[i] = (l - kernel_size[i] + stride[i])//stride[i]
        channel_size = cnn.out_channels
    return input_shape[0] * input_shape[1] * channel_size

class atari_model(nn.Module):
    def __init__(self, in_channels=4, num_actions=18):
        super(atari_model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc5 = nn.Linear(512, num_actions)

    def forward(self, x):
        # print x

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.fc5(x)


def atari_learn(env,                
                num_timesteps):
    # This is just a rough estimate
    num_iterations = float(num_timesteps) / 4.0
    LEARNING_RATE = 5e-5
    lr_multiplier = 3.0
    lr_schedule = PiecewiseSchedule([
                                         (0,                   1e-4 * lr_multiplier),
                                         (num_iterations / 10, 1e-4 * lr_multiplier),
                                         (num_iterations / 2,  5e-5 * lr_multiplier),
                                    ],
                                    outside_value=5e-5 * lr_multiplier)
    optimizer = dqn.OptimizerSpec(
        constructor=optim.Adam,
        kwargs=dict(lr=LEARNING_RATE, eps=1e-4)
    )


    def stopping_criterion(env, t):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return get_wrapper_by_name(env, "Monitor").get_total_steps() >= num_timesteps

    exploration_schedule = PiecewiseSchedule(
        [
            (0, 1.0),
            (1e6, 0.1),
            (num_iterations / 2, 0.01),
        ], outside_value=0.01
    )

    dqn.learn(
        env,
        q_func=atari_model,
        optimizer_spec=optimizer,
        exploration=exploration_schedule,
        stopping_criterion=stopping_criterion,
        replay_buffer_size=1000000,
        batch_size=32,
        gamma=0.99,
        learning_starts=50000,
        learning_freq=4,
        frame_history_len=4,
        target_update_freq=10000,
        grad_norm_clipping=10
    )
    env.close()



def set_global_seeds(i):
    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(i)
    np.random.seed(i)
    random.seed(i)


def get_env(task, seed):
    env_id = task.env_id

    env = gym.make(env_id)

    set_global_seeds(seed)
    env.seed(seed)

    expt_dir = '/tmp/hw3_vid_dir2/'
    env = wrappers.Monitor(env, osp.join(expt_dir, "gym"), force=True)
    env = wrap_deepmind(env)

    return env

def main():
    # Get Atari games.
    benchmark = gym.benchmark_spec('Atari40M')

    # Change the index to select a different game.
    task = benchmark.tasks[3]

    # Run training
    seed = 0 # Use a seed of zero (you may want to randomize the seed!)
    env = get_env(task, seed)
    atari_learn(env, num_timesteps=task.max_timesteps)

if __name__ == "__main__":
    main()
