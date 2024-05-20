import argparse
import numpy as np
import tensorflow as tf
import time
import pickle

import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
import tf_slim.layers as layers
import os

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=100, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=5000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default=None, help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="/tmp/policy/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=200, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False, help="parameter to add to load a previous model, this will NOT deactivate training")
    parser.add_argument("--display", action="store_true", default=False, help="parameter to add to see the agent playing, this will deactivate training")
    return parser.parse_args()

def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out

def make_env(scenario_name, arglist, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env

def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTrainer
    for i in range(num_adversaries):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.adv_policy=='ddpg')))
    for i in range(num_adversaries, env.n):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.good_policy=='ddpg')))
    return trainers


def train(arglist):
    with tf.compat.v1.Session() as session:
        os.makedirs(arglist.save_dir, exist_ok=True)

        env = make_env(arglist.scenario, arglist)

        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))
        with tf.compat.v1.variable_scope("steps"):
            steps_tf = tf.Variable(0, trainable=False, name="steps")
            eps_tf = tf.Variable(0, trainable=False, name="episodes")
            learning_steps_tf = tf.Variable(0, trainable=False, name="learning_steps")

        # Creating tensorboard output
        steps_tb = tf.compat.v1.placeholder(dtype=tf.int32)
        tf.compat.v1.summary.scalar("steps", steps_tb)
        episodes_tb =  tf.compat.v1.placeholder(dtype=tf.int32)
        tf.compat.v1.summary.scalar("episodes", episodes_tb)
        q_loss_tb = tf.compat.v1.placeholder(dtype=tf.float32)
        tf.compat.v1.summary.scalar("q_loss", q_loss_tb)
        p_loss_tb = tf.compat.v1.placeholder(dtype=tf.float32)
        tf.compat.v1.summary.scalar("p_loss", p_loss_tb)
        reward_tb = tf.compat.v1.placeholder(dtype=tf.float32)
        tf.compat.v1.summary.scalar("reward", reward_tb)
        merged = tf.compat.v1.summary.merge_all()
        train_writer = tf.compat.v1.summary.FileWriter(os.path.join(arglist.save_dir, 'tensorboard'), session.graph)

        # Initialize
        U.initialize()

        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        if arglist.display or arglist.restore or False:
            print('Loading previous state...')
            U.load_state(arglist.load_dir)
        learning_steps = 0
        eps_count = 0
        steps_count = 0
        episode_rewards = [0.0]  # sum of rewards for all agents
        q_losses = []
        p_losses = []
        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info
        saver = tf.compat.v1.train.Saver()
        obs_n = env.reset()
        episode_step = 0
        train_step = 0
        t_start = time.time()
        max_rew = -1000000000
        print('Starting iterations...')
        while True:
            # get action
            action_n = [agent.action(obs) for agent, obs in zip(trainers,obs_n)]
            # environment step
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)
            episode_step += 1
            steps_count += 1
            done = all(done_n)
            terminal = (episode_step >= arglist.max_episode_len)
            # collect experience
            for i, agent in enumerate(trainers):
                agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
            obs_n = new_obs_n

            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew

            if done or terminal:
                obs_n = env.reset()

                episode_step = 0
                episode_rewards.append(0)
                for a in agent_rewards:
                    a.append(0)
                agent_info.append([[]])
                eps_count += 1

            # increment global step counter
            train_step += 1

            # for displaying learned policies
            if arglist.display:
                time.sleep(0.1)
                #print(rew_n)
                rgb_array = env.render()
                continue

            # update all trainers, if not in display
            loss = None
            for agent in trainers:
                agent.preupdate()
            for agent in trainers:
                loss = agent.update(trainers, train_step)
            if loss is not None:
                learning_steps += 1
                q_losses.append(loss[0])
                p_losses.append(loss[1])

            # save model, display training output
            if terminal and ((len(episode_rewards) - 1)% arglist.save_rate == 0):
                tf.compat.v1.get_default_session().run(tf.compat.v1.assign_add(steps_tf, steps_count))
                tf.compat.v1.get_default_session().run(tf.compat.v1.assign_add(eps_tf, eps_count))
                tf.compat.v1.get_default_session().run(tf.compat.v1.assign_add(learning_steps_tf, learning_steps))
                learning_steps = 0
                eps_count = 0
                steps_count = 0
                mean_rew = np.mean(episode_rewards[-arglist.save_rate:])

                if mean_rew >= max_rew:
                    max_rew = mean_rew
                    U.save_state(os.path.join(arglist.save_dir, "best_model/"), saver=saver)

                U.save_state(arglist.save_dir, saver=saver)
                # print statement depends on whether or not there are adversaries
                if num_adversaries == 0:
                    print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                        tf.compat.v1.get_default_session().run(steps_tf), tf.compat.v1.get_default_session().run(eps_tf), np.mean(episode_rewards[-arglist.save_rate:]), round(time.time()-t_start, 3)))
                else:
                    print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                        [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards], round(time.time()-t_start, 3)))

                # writing tensorboard output
                summ = session.run(merged, feed_dict={
                    steps_tb: train_step,
                    episodes_tb: len(episode_rewards),
                    q_loss_tb:  np.mean(q_losses[-arglist.save_rate:]),
                    p_loss_tb: np.mean(p_losses[-arglist.save_rate:]),
                    reward_tb: np.mean(episode_rewards[-arglist.save_rate:])
                })
                train_writer.add_summary(summ, train_step)

                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))

            if len(episode_rewards) > arglist.num_episodes:
                print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                break

if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)
