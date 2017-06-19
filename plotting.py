import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, draw, show
from matplotlib.legend_handler import HandlerLine2D

def main():
	# with open('q1_data.p', 'rb') as f:
	# 	data = pickle.loads(f.read())
	# 	f.close()
	# timesteps = data['time_plot']
	# mean_episode_reward_plot = data['mean_episode_reward_plot']
	# best_mean_episode_reward_plot = data['best_mean_episode_reward_plot']
	# learning_rate_plot = data['learning_rate_plot']
	# exploration_t_plot = data['exploration_t_plot']
	# plt.figure()
	# line1 = plt.plot(timesteps/1000, mean_episode_reward_plot, label="mean 100-episode reward")
	# line2 =plt.plot(timesteps/1000, best_mean_episode_reward_plot, label="best mean reward")
	# plt.legend(loc=5)

	# plt.xlabel('time steps (every 1000 steps)')
	# plt.ylabel('rewards')
	# plt.savefig("q1")
	# plt.show()

	# print(mean_episode_reward_plot)
	# print(best_mean_episode_reward_plot)
	# print(learning_rate_plot)
	# print(exploration_t_plot)





	with open('q2_1_lr1_data.p', 'rb') as f:
		data = pickle.loads(f.read())
		f.close()
	timesteps = data['time_plot']
	mean_episode_reward_plot = data['mean_episode_reward_plot']
	best_mean_episode_reward_plot = data['best_mean_episode_reward_plot']
	learning_rate_plot = data['learning_rate_plot']
	exploration_t_plot = data['exploration_t_plot']
	plt.figure()
	line1 = plt.plot(timesteps/1000, mean_episode_reward_plot, label="mean 100-episode reward lr =1")
	


	with open('q2_1_lr10_data.p', 'rb') as f:
		data = pickle.loads(f.read())
		f.close()
	timesteps = data['time_plot']
	mean_episode_reward_plot = data['mean_episode_reward_plot']
	best_mean_episode_reward_plot = data['best_mean_episode_reward_plot']

	line3 = plt.plot(timesteps/1000, mean_episode_reward_plot, label="mean 100-episode reward lr =10")
	
	with open('q2_1_lr7_data.p', 'rb') as f:
		data = pickle.loads(f.read())
		f.close()
	timesteps = data['time_plot']
	mean_episode_reward_plot = data['mean_episode_reward_plot']
	best_mean_episode_reward_plot = data['best_mean_episode_reward_plot']

	line5 = plt.plot(timesteps/1000, mean_episode_reward_plot, label="mean 100-episode reward lr =7")

	with open('q2_1_lr3_data.p', 'rb') as f:
		data = pickle.loads(f.read())
		f.close()
	timesteps = data['time_plot']
	mean_episode_reward_plot = data['mean_episode_reward_plot']
	best_mean_episode_reward_plot = data['best_mean_episode_reward_plot']

	line7 = plt.plot(timesteps/1000, mean_episode_reward_plot, label="mean 100-episode reward lr =3")


	plt.legend(loc=5)

	plt.xlabel('time steps (every 1000 steps)')
	plt.ylabel('rewards')
	plt.savefig("q2_3")
	plt.show()


	# with open('q1_data.p', 'rb') as f:
	# 	data = pickle.loads(f.read())
	# 	f.close()
	# timesteps = data['time_plot']
	# mean_episode_reward_plot = data['mean_episode_reward_plot']
	# best_mean_episode_reward_plot = data['best_mean_episode_reward_plot']
	# learning_rate_plot = data['learning_rate_plot']
	# exploration_t_plot = data['exploration_t_plot']
	# plt.figure()
	# line1 = plt.plot(timesteps/1000, mean_episode_reward_plot, label="mean 100-episode reward lr =1")	


	# with open('q2_1_lr5_data.p', 'rb') as f:
	# 	data = pickle.loads(f.read())
	# 	f.close()
	# timesteps = data['time_plot']
	# mean_episode_reward_plot = data['mean_episode_reward_plot']
	# best_mean_episode_reward_plot = data['best_mean_episode_reward_plot']

	# line3 = plt.plot(timesteps/1000, mean_episode_reward_plot, label="mean 100-episode reward lr =5")
	
	# with open('q2_2_lr10_data.p', 'rb') as f:
	# 	data = pickle.loads(f.read())
	# 	f.close()
	# timesteps = data['time_plot']
	# mean_episode_reward_plot = data['mean_episode_reward_plot']
	# best_mean_episode_reward_plot = data['best_mean_episode_reward_plot']

	# line5 = plt.plot(timesteps/1000, mean_episode_reward_plot, label="mean 100-episode reward lr =10")

	# with open('q2_3_lr0_1_data.p', 'rb') as f:
	# 	data = pickle.loads(f.read())
	# 	f.close()
	# timesteps = data['time_plot']
	# mean_episode_reward_plot = data['mean_episode_reward_plot']
	# best_mean_episode_reward_plot = data['best_mean_episode_reward_plot']

	# line7 = plt.plot(timesteps/1000, mean_episode_reward_plot, label="mean 100-episode reward lr =0.5")


	# plt.legend(loc=5)

	# plt.xlabel('time steps (every 1000 steps)')
	# plt.ylabel('rewards')
	# plt.savefig("q2_3_mean")
	# plt.show()


if __name__ == '__main__':

    main()