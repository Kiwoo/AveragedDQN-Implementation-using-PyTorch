import  sys
import  gym.spaces
import  itertools
import  numpy                       as np
import  random
from    collections                 import namedtuple
from    dqn_utils                   import *
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd


OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])


USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

class Variable(autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        if USE_CUDA:
            data = data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)

def learn(env,
          q_func,
          optimizer_spec,
          exploration=LinearSchedule(1000000, 0.1),
          stopping_criterion=None,
          replay_buffer_size=1000000,
          batch_size=32,
          gamma=0.99,
          learning_starts=50000,
          learning_freq=4,
          frame_history_len=4,
          target_update_freq=10000,
          grad_norm_clipping=10):
    """Run Deep Q-learning algorithm.

    You can specify your own convnet using q_func.

    All schedules are w.r.t. total number of steps taken in the environment.

    Parameters
    ----------
    env: gym.Env
        gym environment to train on.
    q_func: function
        Model to use for computing the q function. It should accept the
        following named arguments:
            img_in: tf.Tensor
                tensorflow tensor representing the input image
            num_actions: int
                number of actions
            scope: str
                scope in which all the model related variables
                should be created
            reuse: bool
                whether previously created variables should be reused.
    optimizer_spec: OptimizerSpec
        Specifying the constructor and kwargs, as well as learning rate schedule
        for the optimizer
    session: tf.Session
        tensorflow session to use.
    exploration: rl_algs.deepq.utils.schedules.Schedule
        schedule for probability of chosing random action.
    stopping_criterion: (env, t) -> bool
        should return true when it's ok for the RL algorithm to stop.
        takes in env and the number of steps executed so far.
    replay_buffer_size: int
        How many memories to store in the replay buffer.
    batch_size: int
        How many transitions to sample each time experience is replayed.
    gamma: float
        Discount Factor
    learning_starts: int
        After how many environment steps to start replaying experiences
    learning_freq: int
        How many steps of environment to take between every experience replay
    frame_history_len: int
        How many past frames to include as input to the model.
    target_update_freq: int
        How many experience replay rounds (not steps!) to perform between
        each update to the target Q network
    grad_norm_clipping: float or None
        If not None gradients' norms are clipped to this value.
    """
    assert type(env.observation_space) == gym.spaces.Box
    assert type(env.action_space)      == gym.spaces.Discrete

    ###############
    # BUILD MODEL #
    ###############

    if len(env.observation_space.shape) == 1:
        # This means we are running on low-dimensional observations (e.g. RAM)
        input_arg = env.observation_space.shape[0]
    else:
        img_h, img_w, img_c = env.observation_space.shape
        input_arg = frame_history_len * img_c
    num_actions = env.action_space.n


    q_values = q_func(input_arg, num_actions).type(dtype)
    target_q_values = q_func(input_arg, num_actions).type(dtype)

    optimizer = optimizer_spec.constructor(q_values.parameters(), **optimizer_spec.kwargs)
    replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)

    def select_epilson_greedy_action(model, obs, t):
        sample = random.random()
        eps_threshold = exploration.value(t)
        if sample > eps_threshold:
            obs = torch.from_numpy(obs).type(dtype).unsqueeze(0) / 255.0
            return model(Variable(obs, volatile=True)).data.max(1)[1].cpu()
        else:
            return torch.IntTensor([[random.randrange(num_actions)]])

    ######

    # Plotting
    
    time_plot                       = []
    mean_episode_reward_plot        = []
    best_mean_episode_reward_plot   = []
    episode_plot                    = []
    exploration_t_plot              = []
    learning_rate_plot              = []

 

    ###############
    # RUN ENV     #
    ###############
    num_param_updates = 0
    mean_episode_reward      = -float('nan')
    best_mean_episode_reward = -float('inf')
    last_obs                 = env.reset()
    LOG_EVERY_N_STEPS        = 1000
    SAVE_EVERY_N_STEPS       = 100000 

    for t in itertools.count():
        if stopping_criterion is not None and stopping_criterion(env, t):
            break

        ret = replay_buffer.store_frame( last_obs )
        obs = replay_buffer.encode_recent_observation()
        action = select_epilson_greedy_action(q_values, obs, t)[0, 0]

        last_obs, reward, done, _ = env.step(action)

        if done:
            last_obs = env.reset()
        
        replay_buffer.store_effect(ret, action, reward, done)

        #####
        if (t > learning_starts and
                t % learning_freq == 0 and
                replay_buffer.can_sample(batch_size)):

            obs_t_batch, act_batch, rew_batch, obs_tp1_batch, done_mask = replay_buffer.sample(batch_size)

            obs_t_batch     = Variable(torch.from_numpy(obs_t_batch).type(dtype) / 255.0)
            act_batch       = Variable(torch.from_numpy(act_batch).long())
            rew_batch       = Variable(torch.from_numpy(rew_batch))
            obs_tp1_batch   = Variable(torch.from_numpy(obs_tp1_batch).type(dtype) / 255.0)
            done_mask       = Variable(torch.from_numpy(done_mask)).type(dtype)

            if USE_CUDA:
                act_batch = act_batch.cuda()
                rew_batch = rew_batch.cuda()

            # train the model

            q_a_values = q_values(obs_t_batch).gather(1, act_batch.unsqueeze(1))
            q_a_vales_tp1 = target_q_values(obs_tp1_batch).detach().max(1)[0]
            # q_a_vales_tp1 = not_done_mask * q_a_vales_tp1
            target_values = rew_batch + (gamma * (1-done_mask) * q_a_vales_tp1)
            loss = (target_values - q_a_values).pow(2).sum()
            if t % LOG_EVERY_N_STEPS == 0:
                print "loss at {} : {}".format(t, loss.data[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            num_param_updates += 1

            # update the target network
            if t%target_update_freq==0:
                target_q_values.load_state_dict(q_values.state_dict())

            #####

        ### 4. Log progress
        episode_rewards = get_wrapper_by_name(env, "Monitor").get_episode_rewards()
        if len(episode_rewards) > 0:
            mean_episode_reward = np.mean(episode_rewards[-100:])
        if len(episode_rewards) > 100:
            best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)
        if t % LOG_EVERY_N_STEPS == 0:#        and model_initialized:
            print("Timestep %d" % (t,))
            print("mean reward (100 episodes) %f" % mean_episode_reward)
            print("best mean reward %f" % best_mean_episode_reward)
            print("episodes %d" % len(episode_rewards))
            print("exploration %f" % exploration.value(t))
            # print("learning_rate %f" % optimizer_spec.lr_schedule.value(t))

            time_plot.append(t)
            mean_episode_reward_plot.append(mean_episode_reward)
            best_mean_episode_reward_plot.append(best_mean_episode_reward)
            episode_plot.append(len(episode_rewards))
            exploration_t_plot.append(exploration.value(t)) 
            # learning_rate_plot.append(optimizer_spec.lr_schedule.value(t))

            sys.stdout.flush()
            # Plotting

            q1_data_all = {'time_plot': np.array(time_plot),
                           'mean_episode_reward_plot': np.array(mean_episode_reward_plot),
                           'best_mean_episode_reward_plot': np.array(best_mean_episode_reward_plot),
                           'episode_plot': np.array(episode_plot),
                           'exploration_t_plot': np.array(exploration_t_plot)}
                           # ,
                           # 'learning_rate_plot': np.array(learning_rate_plot)
            f = open('q2_1_lr3_data.p', 'w')

            pickle.dump(q1_data_all, f)
            f.close()
