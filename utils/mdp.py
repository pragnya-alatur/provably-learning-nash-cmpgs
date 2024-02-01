"""
This file implements the MDP class for the finite-horizon setting.
"""
import numpy as np

class Mdp:
    def __init__(self, S, A, P, H, R, mu):
        self.S = S
        self.A = A
        self.P = P
        self.H = H
        self.R = R
        self.mu = mu

    
    def solve(self):
        # This method solves the MDP using a backward induction approach.
        policy = np.zeros((self.H, self.S,self.A))
        V = np.zeros((self.H, self.S))
        for h in np.flip(range(self.H)):
            next_V = None
            if h == self.H-1:
                next_V = np.zeros(self.S)
            else:
                next_V = V[h+1,:]
            # Compute the V table for the current step.
            for s in range(self.S):
                values = np.zeros(self.A)
                for a in range(self.A):
                    values[a] = self.R[h,s,a] + np.dot(self.P[h,s,a,:], next_V)
                a = np.argmax(values)
                policy[h,s,a] = 1
                V[h,s] = values[a]
        return policy

    
    def eval_policy(self, policy):
        V = np.zeros((self.H, self.S))
        for h in np.flip(range(self.H)):
            next_V = None
            if h == self.H-1:
                next_V = np.zeros(self.S)
            else:
                next_V = V[h+1,:]
            # Compute the V table for the current step.
            for s in range(self.S):
                values = np.zeros(self.A)
                for a in range(self.A):
                    values[a] = self.R[h,s,a]
                    for ns in range(self.S):
                        values[a] += self.P[h,s,a,ns] * next_V[ns]
                V[h,s] = np.dot(policy[h,s,:], values)
        return np.dot(V[0,:], self.mu)