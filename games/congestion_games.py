"""
This file implements congestion games and is a slightly modified version of
https://openreview.net/forum?id=gfwON7rAm4.
"""
import itertools as it
import numpy as np
from math import comb

class CongGame:
	#inputs: num players, list of linear multiplier on utility for num of players
	def __init__(self, n, weights):
		self.n = n
		self.weights = weights
		self.a = len(weights) #number of facilities
		self.facilities = [i for i in range(self.a)]

	def get_facility_rewards(self, actions):
		density = np.bincount(actions,minlength=self.a)
		facility_rewards = self.a * [0]
		for j in range(self.a):
			facility_rewards[j] = density[j] * self.weights[j][0] + self.weights[j][1]
		return facility_rewards

	def get_agent_reward(self, actions, agent_action):
		facility_rewards = self.get_facility_rewards(actions)
		return facility_rewards[agent_action]

	def get_reward(self, actions):
		rewards = self.n * [0]
		for i in range(self.n):
			rewards[i] = self.get_agent_reward(actions, actions[i])
		return rewards