"""
This file plots the dual function for the matrix games that are used as counter-examples for strong
duality in the paper.
"""

import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import csv


"""
The two agents aim to solve the following matrix game:

max_{x,y} x'Ay s.t. x'By <= c,
where:
* A is the shared reward matrix,
* B is the constraint matrix,
* c is the constraint threshold,
* x,y denote the agents' strategies.
"""
A = np.array([[3,2], [2,4]])
B = np.array([[0,0], [0,1]])
c = 0.5


"""
This function evaluates the dual function for the matrix game at the given dual variable 'lam'.
For any l, the dual function is defined as follows:
L(l) = max_{x,y} {x'Ay + l*(c-x'By)}
"""
def dual(lam):
	obj = A.flatten()-lam*B.flatten()
	# Constraints to ensure that we only consider valid probability distributions as strategies.
	constrA = np.array([np.ones((A.shape)).flatten()])
	constrb = np.array([1])
	bounds = [(0,1),(0,1),(0,1),(0,1)]
	sol = linprog(-obj, A_eq=constrA, b_eq=constrb, bounds=bounds)
	return -sol.fun + c*lam


# Plots the dual function in the range [0,2].
_, ax = plt.subplots()
x = np.linspace(0,2,300)
y = [dual(lamb) for lamb in x]
ax.plot(x,y)
ax.set_xlabel(r'$\lambda$')
ax.set_ylabel(r'$d(\lambda)$')
plt.show()
