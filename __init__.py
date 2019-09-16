import numpy as np
from watermatrices import Amat, Bmat, yvec
import scipy
import scipy.linalg
import copy
from prettytable import PrettyTable
import matplotlib.pyplot as plt
from plotify import Plotify

# -------------- Exercise A --------------- #

A = Amat
B = Bmat

def infinite_norm(A):
  maxes = []
  for j in A:
    abs_row = []
    for i in j:
      abs_row.append(abs(i))

    maxes.append(sum(abs_row))

  return max(maxes)

def cond_inf(A):
  norm_A = infinite_norm(A)
  norm_A_inverse = infinite_norm(np.linalg.inv(A))

  cond = norm_A * norm_A_inverse
  return cond

def combine_submatrices(A,B):
  A_B = np.concatenate((A,B), axis=1)
  B_A = np.concatenate((B,A), axis=1)
  combined_matrix = np.concatenate((A_B,B_A), axis=0)

  return combined_matrix

def create_S_matrix():
  I = np.identity(7)
  I_neg = (np.identity(7, int) * -1).astype(float) # int is to get rid of -0. (not a problem, but looks annoying)
  zero_matrix = np.zeros((7,7))
  
  first_row = np.concatenate((I,zero_matrix), axis=1)
  second_row = np.concatenate((zero_matrix,I_neg), axis=1)
  combined_matrix = np.concatenate((first_row,second_row), axis=0)

  return combined_matrix

def create_z_vector():
  z = np.concatenate((yvec, yvec * -1), axis=0)

  return z

E = combine_submatrices(A,B)
S = create_S_matrix()
z = create_z_vector()

# cond1 = ω =  1.3
# cond2 = ω =  1.607
# cond3 = ω =  3.0

frequencies = [1.300, 1.607, 3.000]
def create_E_omega_S(omega):
  return np.subtract(E, omega * S)

for index, omega in enumerate(frequencies):
  E_omega_S = create_E_omega_S(omega)
  cond = cond_inf(E_omega_S)
  print('cond∞(E - ωS) = ', cond, ', where ω = ', omega)

delta_omega = 0.5e-3

print('')

# -------------- Exercise B --------------- #

for index, omega in enumerate(frequencies):
  bound = cond_inf(np.subtract(E, omega * S)) * (infinite_norm(delta_omega * S) /
                                                 infinite_norm(np.subtract(E, omega * S)))
  print('the bound on the relative forward error in the max-norm for ω = ', omega, 'is: ', bound)


# -------------- Exercise C --------------- #

def lu_factorize(A):
  U = copy.deepcopy(A)
  n = U.shape[0]
  L = np.identity(n)

  for k in range(1, n):
    for j in range(k):
      L[k,j] = U[k,j] / U[j,j]
      U[k] = np.subtract(U[k], U[j] * L[k,j])

  return L, U

def forward_substitute(L,b):
  n = L.shape[0]
  x = np.zeros(n)
  for j in range(n):
    x[j] = b[j]/L[j,j]
    for i in range(j+1, n):
      b[i] = b[i] - L[i,j] * x[j]
  
  return x

def back_substitute(U,y):
  n = U.shape[0]
  x = np.zeros(n)
  for j in range(n-1,-1,-1):
    x[j] = y[j] / U[j,j]
    for i in range(j):
      y[i] = y[i] - U[i,j] * x[j]
  
  return x

def solve_test():
  A = np.array([[2,1,1],
                [4,1,4],
                [-6,-5,3]])
  b = np.array([4,11,4])

  L, U = lu_factorize(A)
  print('L\n', L)
  print('U\n', U)
  y = forward_substitute(L, b)
  x = back_substitute(U, y)

  return x, y

x, y = solve_test()
print('y', y)
print('x', x)

# -------------- Exercise D --------------- #


def solve_alpha(omega):
  A = create_E_omega_S(omega)
  L, U = lu_factorize(A)

  y = forward_substitute(L, z)
  x = back_substitute(U,y)

  alpha = np.matmul(z.T, x)

  return alpha

table = PrettyTable()
table.field_names = ['', 'α(ω)', 'α(ω + δω)', '(ω - δω)']

for omega in frequencies:
  delta_omega = 0.5e-3
  alpha = solve_alpha(omega)
  alpha_plus = solve_alpha(omega + delta_omega)
  alpha_minus = solve_alpha(omega - delta_omega)
  
  table.add_row([('ω = ' + str(omega)), alpha, alpha_plus, alpha_minus])

print(table)

# -------------- Exercise E --------------- #

frequencies= np.linspace(1.2, 4.0, num=1000)
alphas = []

for omega in frequencies:
  alpha = solve_alpha(omega)
  alphas.append(alpha)

plotify = Plotify() # a small skin over matplotlib developed by me.

plotify.plot(x=frequencies, y=alphas, xlabel='Frequency (ω)', ylabel='Alpha Value', title='Alpha Values Over Different Frequencies')

# -------------- Exercise F --------------- #
 

