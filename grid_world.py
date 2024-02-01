from enum import Enum
import numpy as np
import sys
from utils.cmdp import Cmdp
from utils.grid import Grid
from utils.mdp import Mdp
import csv
from scipy.optimize import minimize

output_dir = '/tmp'

# Set up the game.
N = 2 # Number of agents.
grid = Grid(4,4)
S = grid.states
A = 4 # Number of actions: up=0, right=1, down=2, left=3.
H = 6 # Horizon.
initial_state = 0
target_state = 11
mu = np.zeros(S) # Initial state distribution.
mu[initial_state] = 1.0
R = np.zeros((H,S,A)) # Reward matrix.
R[:,1,:] = 2.0
R[:,4,:] = 1.0
R[:,target_state,:] = 10.0
C = np.zeros((H,S*S,A)) # Constraint matrix.
for s in range(S):
	if s in [initial_state, target_state]:
		continue
	js = grid.individual_to_joint_states(s,s)
	C[:,js,:] = 1
alpha = 0.1 # Constraint threshold.
tight_alpha = 0.1
threshold = 1e-10 # Threshold for termination of CMPG-CA.


successful_runs = 0
while True:
	if successful_runs >= 20:
		break
	# Initialize the algorithm with a a safe, sub-optimal policy: agent 1 plays a uniform random
	# policy, and agent 2 plays a deterministic policy that always goes up. This ensures that there are
	# no constraint violations in the initial policy.
	policies = np.zeros((N,H,S,A))
	for n in range(2):
		for h in range(H):
			for s in range(S):
				policies[n,h,s,:] = np.random.rand(A)
				policies[n,h,s,:] = np.divide(policies[n,h,s,:], np.sum(policies[n,h,s,:]))
	policies[1,:,0,:] = 0.0
	policies[1,:,0,0] = 1.0
	# Create the transition matrix for each agent.
	P = np.zeros((H,S,A,S))
	for s in range(S):
		for a in range(A):
			ns = grid.next_state(s, a)
			P[:,s,a,ns] = 1.0
	# Run the CMPG-CA algorithm.
	rewards = []
	constraints = []
	diffs = np.empty((0,N))
	post_convergence_rounds = 0
	while True:
		acmdp = Cmdp(S,A,P,H,R,mu) # Auxiliary CMDP to compute occupancy measures.
		occ_measures = np.zeros((N,H,S,A))
		for n in range(N):
			occ_measures[n,:,:,:] = acmdp._policy_to_occ(policies[n,:,:,:])
		# Set up the per-agent CMDPs
		C1 = np.zeros((H,S,A))
		C2 = np.zeros((H,S,A))
		for h in range(H):
			for s1 in range(S):
				for s2 in range(S):
					for a1 in range(A):
						for a2 in range(A):
							C1[h,s1,a1] += occ_measures[1,h,s2,a2]*C[h,grid.individual_to_joint_states(s1,s2),0]
							C2[h,s2,a2] += occ_measures[0,h,s1,a1]*C[h,grid.individual_to_joint_states(s1,s2),0]
		cmdps = [Cmdp(S,A,P,H,R,mu,C1,tight_alpha), Cmdp(S,A,P,H,R,mu,C2,tight_alpha)]
		curr_reward = np.sum([cmdps[n].eval_policy(policies[n,:,:,:]) for n in range(N)])
		curr_constraint = cmdps[0].eval_constraint(policies[0,:,:,:]) # Both agents have the same constraint.
		if curr_constraint>alpha:
			print('Constraint violated: {}, skipping this game'.format(curr_constraint))
			break
		rewards.append(curr_reward)
		constraints.append(curr_constraint)
		# Both agents compute their best response w.r.t. the other agent's policy.
		print('Current reward: {} / Current constraint: {}'.format(curr_reward, curr_constraint))
		new_policies = np.zeros((N,H,S,A))
		new_rewards = np.zeros(N)
		new_constraints = np.zeros(N)
		for n in range(N):
			new_policies[n,:,:,:] = cmdps[n].solve(method = 'primal-dual', iters=10000)
			new_rewards[n] = cmdps[n].eval_policy(new_policies[n,:,:,:]) + cmdps[1-n].eval_policy(policies[1-n,:,:,:])
			new_constraints[n] = cmdps[n].eval_constraint(new_policies[n,:,:,:])
			print('Agent {} reward diffs: {} / constraint: {}'.format(n, new_rewards[n]-curr_reward, new_constraints[n]))
		print('***************************************')
		new_diffs = np.array([new_rewards[n]-curr_reward for n in range(N)])
		diffs = np.append(diffs, [new_diffs], axis=0)
		if np.max(new_diffs)<=threshold:
			print('Converged with reward {} and constraint {}'.format(curr_reward, curr_constraint))
			post_convergence_rounds += 1
		if post_convergence_rounds >= 3:
			for n in range(N):
				with open('{}/best_response_{}.csv'.format(output_dir, n), 'a', newline='') as file:
					writer = csv.writer(file)
					writer.writerow(diffs[:,n])
					file.close()
			with open('{}/constraints.csv'.format(output_dir), 'a', newline='') as file:
				writer = csv.writer(file)
				writer.writerow(constraints)
				file.close()
			successful_runs += 1
			break
		j = np.argmax(new_diffs)
		policies[j,:,:,:] = new_policies[j,:,:,:] # Update the policy of the agent that improved the most.
	print('Successful runs so far: {}'.format(successful_runs))


# After running CA-CMPG, we also solve the dual problem to compare the values.
def joint_actions_to_individual_actions(a):
	a1 = int(a/4)
	a2 = a-a1*4
	return a1, a2

def individual_actions_to_joint_actions(a1, a2):
	return a1*4+a2

# Set up the CMDP for the dual problem.
Sd = grid.joint_states
Ad = A*A
Pd = np.zeros((H,Sd,Ad,Sd))
Rd = np.zeros((H,Sd,Ad))
Cd = np.zeros((H,Sd,Ad))
for s in range(Sd):
	s1,s2 = grid.joint_state_to_individual_states(s)
	for a in range(Ad):
		a1,a2 = joint_actions_to_individual_actions(a)
		ns1 = grid.next_state(s1, a1)
		ns2 = grid.next_state(s2, a2)
		ns = grid.individual_to_joint_states(ns1, ns2)
		Pd[:,s,a,ns] = 1.0
		Rd[:,s,a] = R[:,s1, a1] + R[:,s2, a2]
		Cd[:,s,a] = C[:,s,0]
mud = np.zeros(Sd)
mud[initial_state] = 1

def dual_fun(lam):
	print('Dual function invoked with lambda={}'.format(lam))
	L = Rd - lam*Cd
	mdp = Mdp(Sd, Ad, Pd, H, L, mud)
	policy = mdp.solve()
	val = mdp.eval_policy(policy)+alpha*lam
	return val


print('Successfully created the dual CMDP')
constraints = [{'type': 'ineq', 'fun': lambda x: x}]
dual = minimize(dual_fun, 1, constraints=constraints)
print('Dual value: {}'.format(dual.fun))