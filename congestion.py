"""
This file implements the congestion game experiment from the paper.
"""
import games.congestion_games as cg
import numpy as np
import itertools as it
import utils.mdp as mdp
import utils.cmdp as cmdp
import time
import csv

# The results will be saved in this directory.
output_dir = '/tmp/'

# Set up the distancing game described in Leonardos et al. (2021).
N = 8 #number of agents
safe_state = cg.CongGame(N,[[1,0],[2,0],[4,0],[6,0]])
bad_state = cg.CongGame(N,[[1,-100],[2,-100],[4,-100],[6,-100]])
state_dic = {0: safe_state, 1: bad_state}
A = safe_state.a 
actions = list(it.product(range(A), repeat=8)) # List of all possible actions.
mu = [0.5, 0.5] # Initial state distribution.
S = len(state_dic)
H = 2 # Horizon.
# Build the transition matrix.
P = np.zeros((H, S, len(actions), S))
for a in range(len(actions)):
    density = np.bincount(actions[a], minlength=A)
    # Transitions from the safe state.
    if max(density) > N/2:
        P[:,0,a,1] = 1
    else:
        P[:,0,a,0] = 1
    # Transitions from the bad state.
    if max(density) > N/4:
        P[:,1,a,1] = 1
    else:
        P[:,1,a,0] = 1
# Build the constraint matrix.
C = np.zeros((H, S, len(actions)))
for a in range(len(actions)):
    action = actions[a]
    density = np.bincount(action, minlength=A)
    if max(density)>N/2:
        C[:,1,a] = 1
alpha = 0.5 # Constraint threshold.

for run in range(50):
    print('Run: {}'.format(run))
    # Initialize policies randomly.
    policies = np.zeros((N, H, S, A))
    for n in range(N):
        for h in range(H):
            for s in range(S):
                policies[n,h,s,:] = np.random.random(A)
                policies[n,h,s,:] = policies[n,h,s,:] / np.sum(policies[n,h,s,:])
    # Start the CA-CMPG algorithm.
    threshold = 1e-6
    i = 0
    diffs = np.empty((0,N))
    constraints = np.array([])
    post_converged = 0
    while True:
        print('Iteration {}'.format(i))
        # Build the CMDP for each agent.
        Pn = np.zeros((N, H, S, A, S))
        Rn = np.zeros((N, H, S, A))
        Cn = np.zeros((N, H, S, A))
        start = time.time()
        for h in range(H):
            for s in range(S):
                for a in range(len(actions)):
                    action = actions[a]
                    rewards = state_dic[s].get_reward(action)
                    assert(len(rewards) == N)
                    for n in range(N):
                        factor = np.prod([policies[agent,h,s,action[agent]] for agent in range(N) if agent!=n])
                        Pn[n,h,s,action[n],:] += factor*P[h,s,a,:]
                        Rn[n,h,s,action[n]] += factor*rewards[n]
                        Cn[n,h,s,action[n]] += C[h,s,a]*factor
        end = time.time()
        print("Time to build P and R for all agents: {}".format(end-start))

        # Solve the CMDP for each agent.
        new_policies = np.zeros((N, H, S, A))
        curr_rewards = np.zeros(N)
        new_rewards = np.zeros(N)
        for n in range(N):
            print('Agent {}'.format(n))
            m = cmdp.Cmdp(S, A, Pn[n], H, Rn[n], mu, Cn[n], alpha)
            curr_rewards[n] = m.eval_policy(policies[n,:,:,:])
            curr_constraint = m.eval_constraint(policies[n,:,:,:])
            if n == 0:
                constraints = np.append(constraints, curr_constraint)
            print('Current constraint: {}'.format(curr_constraint))
            start = time.time()
            new_policies[n,:,:,:] = m.solve(method='lp')
            end = time.time()
            print("Time to solve the CMDP: {}".format(end-start))
            new_rewards[n] = m.eval_policy(new_policies[n,:,:,:])
            print('Current reward: {} / New reward: {}'.format(curr_rewards[n], new_rewards[n]))
            new_constraint = m.eval_constraint(new_policies[n,:,:,:])
            print('Current constraint: {} / New constraint: {}'.format(curr_constraint, new_constraint))

        diffs = np.vstack((diffs, new_rewards - curr_rewards))
        if np.max(np.abs(new_rewards - curr_rewards)) < threshold:
            post_converged += 1
            break
        if post_converged >= 3:
            print('Converged after {} iterations.'.format(i))
            break
        j = np.argmax(new_rewards - curr_rewards)
        policies[j,:,:,:] = new_policies[j,:,:,:]
        i+=1

    print('Final policies:')
    for n in range(N):
        print('Agent {}'.format(n))
        for h in range(H):
            print('Time {}'.format(h))
            print(policies[n,h,:,:])
    
    for n in range(N):
        with open('{}/best_response_{}.csv'.format(output_dir, n), 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(diffs[:,n])
            file.close()
    with open('{}/constraints.csv'.format(output_dir), 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(constraints)
        file.close()
