""" This file implements a CMDP solver for the finite-horizon setting."""
import numpy as np
from scipy.optimize import linprog
import sys
from utils.mdp import Mdp


class Cmdp:
    def __init__(self, S, A, P, H, R, mu, C=None, alpha=None):
        self.S = S
        self.A = A
        self.P = P
        self.H = H
        self.R = R
        self.C = C
        self.alpha = alpha
        self.mu = mu

    
    def solve(self, method='lp', iters=None):
        if method == 'lp':
            return self._solve_lp()
        if method == 'primal-dual':
            return self._solve_primal_dual(iters)
        sys.exit('Error: Unknown method: {}'.format(method))


    def _solve_lp(self):
        # This method solves the CMDP using the following linear program:
        # min_{rho} sum_{(s,a)} sum_{h} rho_h(s,a) * R(s,a)
        #      sum_{a} rho_1(s,a) = mu(s), for all s in [1,S].
        #      sum_{a} rho_h(s,a) = sum_{(s',a')} rho_{h-1}(s',a') * P(s|s',a'), for all s in [1,S], h in [2,H].
        num_vars = self.S*self.A*self.H
        eq_constraints = self.S + self.S*self.H
        A_eq = np.zeros((eq_constraints,num_vars))
        b_eq = np.zeros(eq_constraints)
        next_constr_index = 0
        # The following equality constraints ensure that the occupation measures at step h=1 are
        # consistent with the initial distribution.
        for s in range(self.S):
            a_eq = np.zeros((self.H, self.S, self.A))
            a_eq[0,s,:] = 1
            A_eq[next_constr_index,:] = a_eq.flatten()
            b_eq[next_constr_index] = self.mu[s]
            next_constr_index += 1
        # The following equality constraints ensure that the occupation measures at step h=2,...,H
        # are consistent with the transition probabilities.
        for h in range(1,self.H):
            for s in range(self.S):
                a_eq = np.zeros((self.H, self.S, self.A))
                a_eq[h,s,:] = 1
                for ps in range(self.S):
                    for a in range(self.A):
                        a_eq[h-1,ps,a] = -self.P[h-1,ps,a,s]
                A_eq[next_constr_index,:] = a_eq.flatten()
                next_constr_index += 1
        A_ub, b_ub = None, None
        if self.C is not None:
            A_ub = np.array([self.C.flatten()])
            b_ub = np.array([self.alpha])
        res = linprog(
            -self.R.flatten(), A_ub=A_ub, b_ub=b_ub,
            A_eq=A_eq, b_eq=b_eq, bounds=[(0,1)]*num_vars, method='highs')
        if not res.success:
            sys.exit('Error: Failed to solve CMDP: {}'.format(res.message))
        occ_measures = res.x.reshape((self.H, self.S, self.A))
        return self._occ_to_policy(occ_measures)


    def _occ_to_policy(self, occ_measures):
        policy = np.zeros((self.H, self.S,self.A))
        for h in range(self.H):
            for s in range(self.S):
                if occ_measures[h,s,:].sum() == 0.0:
                    policy[h,s,:] = np.divide(1.0,self.A)
                    continue
                policy[h,s,:] = np.divide(occ_measures[h,s,:], occ_measures[h,s,:].sum())
        return policy


    def _solve_primal_dual(self, max_iters=None):
        lambdas = np.zeros(max_iters)
        eps = 1e-4 # Terminating condition for the primal-dual algorithm.
        occ_measures = np.zeros((self.H, self.S, self.A))
        mdp = Mdp(self.S, self.A, self.P, self.H, self.R, self.mu)
        print('Starting the primal-dual algorithm')
        for i in range(1,max_iters):
            L = self.R - lambdas[i-1]*self.C
            mdp.R = L
            policy = mdp.solve()
            occ = self._policy_to_occ(policy)
            occ_measures += occ
            eta = 1.0/np.sqrt(i+1)
            constr_value = np.dot(occ.flatten(), self.C.flatten())
            lambdas[i] = max(0.0, lambdas[i-1]-eta*(self.alpha-constr_value))
            if abs(lambdas[:i].mean() - lambdas[:i+1].mean()) < eps:
                return self._occ_to_policy(occ_measures/(i+1))
        return self._occ_to_policy(occ_measures/(i+1))


    def _policy_to_occ(self, policy):
        occ = np.zeros((self.H, self.S, self.A))
        for a in range(self.A):
            occ[0,:,a] = np.multiply(self.mu, policy[0,:,a])
        for h in range(1,self.H):
            for s in range(self.S):
                mult = np.dot(occ[h-1,:,:].flatten(), self.P[h-1,:,:,s].flatten())
                occ[h,s,:] += mult*policy[h,s,:]
        return occ


    def eval_policy(self, policy):
        occ = self._policy_to_occ(policy)
        return np.dot(occ.flatten(), self.R.flatten())

    
    def eval_constraint(self, policy):
        occ = self._policy_to_occ(policy)
        return np.dot(occ.flatten(), self.C.flatten())
