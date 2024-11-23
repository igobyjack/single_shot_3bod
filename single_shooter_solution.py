import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.linalg import norm, inv
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from load_gmat import *
import gmatpy as gmat
import re
from tqdm import tqdm
from joblib import Parallel, delayed
import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint
from jax import jit
import os
os.environ["JAX_ENABLE_X64"] = "True"
os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
def Single_Shooting_3_body(X0, Xf, t0, tf, mu, simSun, simWind, tol, iter_max):
    r_miss = X0[0:3]
    r_target = Xf[0:3]
    stm0 = np.eye(6).reshape(-1)
    r_miss_saved = []
    k = 1
    while norm(r_miss) / norm(r_target) > tol and k < iter_max:
        

        #this updates the time in the GMAT script by writing directly on the file, it's in seconds and I currently have it set to the 't' variable. Should be able to take a float. 
        #this has to come before the gmat.LoadScript() function, otherwise it will not update the time in the script since we're writing on the file itself. 

        with open('CislunarTEMPLATE.script', 'r') as file:
            content = file.read()

        t_for_gmat = tf * TU
        updated_content = re.sub(r'\{Sat\.ElapsedSecs\s*=\s*\d+(\.\d+)?\}', f'{{Sat.ElapsedSecs = {t_for_gmat}}}', content)
        with open('CislunarTEMPLATE.script', 'w') as file:
            file.write(updated_content)

        theScript = 'CislunarTEMPLATE'
        gmat.LoadScript(theScript + '.script')
        
        Sat = gmat.GetObject("Sat")

        #updated so it converts the units
        Sat.SetField("X", (X0[0] * LU))
        Sat.SetField("Y", (X0[1] * LU))
        Sat.SetField("Z", (X0[2] * LU))
        Sat.SetField("VX", (X0[3] * Vconv))
        Sat.SetField("VY", (X0[4] * Vconv))
        Sat.SetField("VZ", (X0[5] * Vconv))
        Sat.SetField("StateType", "Cartesian")
        elfm = gmat.GetObject("Earth_Luna_ForceModel")
        elsfm = gmat.GetObject("Sun_Earth_Luna_ForceModel")
        elswfm = gmat.GetObject("Sun_Earth_Luna_SRP_ForceModel")
        elwfm = gmat.GetObject("Earth_Luna_SRP_ForceModel")
        prop = gmat.GetObject("spacecraft_prop")
        
        prop.AddPropObject(Sat)
        
        if simSun and simWind:
            prop.SetReference(elswfm)
        if not simSun and simWind:
            prop.SetReference(elwfm)
        if simSun and not simWind:
            prop.SetReference(elsfm)
        if not simSun and not simWind:
            prop.SetReference(elfm)
        
        #at this instant, the STM and final states are generated 
        gmat.RunScript()
        
        #read STM
        with open('C:/Desktop/GMAToutput/STM.txt', 'r') as file:
            lines = file.readlines()
            last_line = lines[-6:]
            stmMatrix = []
            for line in last_line:
                values = [float(value) for value in line.split()]
                stmMatrix.append(values)
        #convert STM to standard units
        S = np.diag([1/LU, 1/LU, 1/LU, TU/LU, TU/LU, TU/LU])
        stm_f = S.dot(np.array(stmMatrix)).dot(np.linalg.inv(S))


        #read final state
        #the tweak ive made here is I found I could get more accurate final values when reporting to the outputs using the script
        #this just reads the final line of the output file
        #I know it probably adds some more to the run time, but it fixes a lot of stuff with accuracy 
        with open('C:/Desktop/GMAToutput/Stats.txt', 'r') as file:
            lines = file.readlines()
            
            # Extract the last line
            last_line = lines[-1]
            
            # Split the last line into individual numbers
            values = last_line.split()
            
            # Convert these numbers from strings to floats
            float_values = [float(value) for value in values]
        
            sol = np.array(float_values)

            for i in range(3):
                sol[i] *= LU

            for i in range(3, 6):
                sol[i] *= Vconv
        
        gmat.SaveScript(r"C:/Desktop/OrbitsPython/bogodaTest.script")

        r_miss = sol[:3]/LU - r_target
        r_miss_saved.append(r_miss)
        B = stm_f[:3, 3:] # gmat stm propagator
        if np.linalg.cond(B) < 1/np.finfo(B.dtype).eps : # np.linalg.det(B) !=0:
            dv0 = -inv(B).dot(r_miss)
        else:
            break
        X0[3:6] += dv0
        k += 1
    if norm(r_miss) / norm(r_target) > tol:
         X0[3:6] = 1
    return np.array(r_miss_saved), X0[3:6]
@jit
def x_stm_propagator_3_body(t, Y, mu):
    dY = jnp.zeros(42)
    # State derivatives
    dY = jnp.array(cr3bp_dynamics(t, Y[:6], mu))
    # STM derivatives
    stm = Y[6:].reshape((6, 6))
    A = System_Matrix_3_body(Y[:3], mu)
    dstm = A.dot(stm)
    dY = jnp.concatenate([dY, dstm.reshape(-1)])
    return dY
@jit
def cr3bp_dynamics(t, state, mu):
    x, y, z, vx, vy, vz = state
    r1 = jnp.sqrt((x + mu)**2 + y**2 + z**2)
    r2 = jnp.sqrt((x - 1 + mu)**2 + y**2 + z**2)
    ax = x + 2 * vy - (1 - mu) * (x + mu) / r1**3 - mu * (x - 1 + mu) / r2**3
    ay = y - 2 * vx - (1 - mu) * y / r1**3 - mu * y / r2**3
    az = -(1 - mu) * z / r1**3 - mu * z / r2**3
    return jnp.array([vx, vy, vz, ax, ay, az])
@jit
def System_Matrix_3_body(Y, mu):
    x, y, z = Y[0], Y[1], Y[2]
    # Calculate distances r1 and r2
    r1 = jnp.sqrt((x + mu)**2 + y**2 + z**2)
    r2 = jnp.sqrt((x - 1 + mu)**2 + y**2 + z**2)
    # Calculate second partial derivatives of Omega
    Uxx = 1 - (1 - mu) * (1 - 3 * (x + mu)**2 / r1**2) / r1**3 - mu * (1 - 3 * (x - 1 + mu)**2 / r2**2) / r2**3
    Uyy = 1 - (1 - mu) * (1 - 3 * y**2 / r1**2) / r1**3 - mu * (1 - 3 * y**2 / r2**2) / r2**3
    Uzz = -(1 - mu) * (1 - 3 * z**2 / r1**2) / r1**3 - mu * (1 - 3 * z**2 / r2**2) / r2**3
    Uxy = 3 * y * ((1 - mu) * (x + mu) / r1**5 + mu * (x - 1 + mu) / r2**5)
    Uxz = 3 * z * ((1 - mu) * (x + mu) / r1**5 + mu * (x - 1 + mu) / r2**5)
    Uyz = 3 * y * z * ((1 - mu) / r1**5 + mu / r2**5)
    # Construct the matrix A
    A = jnp.zeros((6, 6), dtype=jnp.float64)
    A = A.at[0, 3].set(1)
    A = A.at[1, 4].set(1)
    A = A.at[2, 5].set(1)
    A = A.at[3, 0].set(Uxx)
    A = A.at[3, 1].set(Uxy)
    A = A.at[3, 2].set(Uxz)
    A = A.at[3, 4].set(2)
    A = A.at[4, 0].set(Uxy)
    A = A.at[4, 1].set(Uyy)
    A = A.at[4, 2].set(Uyz)
    A = A.at[4, 3].set(-2)
    A = A.at[5, 0].set(Uxz)
    A = A.at[5, 1].set(Uyz)
    A = A.at[5, 2].set(Uzz)
    return A
def Transfer_Cost(t, X0, Xf, t0, f_area, mu, simSun, simWind, tol, iter_max):
    # Create a deep copy of X0 to avoid modifying the original array
    X0_copy = np.copy(X0)
    # X0_copy[3:6] = X0_copy[3:6] + dv0
    # Solve the Lambert problem with the copied X0
    r_miss_saved, v0_optimal = Single_Shooting_3_body(X0_copy, Xf, t0, t[0], mu, simSun, simWind, tol, iter_max)
    # Modify the copy of X0 for propagation
    X0_copy[3:] = v0_optimal
    # Propagate the trajectory with the modified X0
    sol_optimal = solve_ivp(cr3bp_dynamics, [t0, t[0]], X0_copy, args=(mu,), rtol=1e-6, atol=1e-9, method='LSODA')
    # Calculate the delta-v required at the start and end of the transfer
    dv0 = X0[3:6] - v0_optimal
    dvf = Xf[3:6] - sol_optimal.y[3:6, -1]
    # Return the sum of the squared norms of the delta-v vectors
    return norm(dv0) + norm(dvf) # + 2*norm(dv0-dvf)
def optimize_time_of_transfer(X0, Xf, mu_E, mu_m, a_H, simSun, simWind):
    t_initial_guess = np.linalg.norm(X0[:3]-Xf[:3])/np.linalg.norm(X0[3:])
    if np.linalg.norm(X0[:3]-np.array([1, 0, 0])) < a_H and np.linalg.norm(Xf[:3]-np.array([1, 0, 0])):
        mu_ig = mu_m
    else:
        mu_ig = mu_E
    t_ig = (X0[-1]+Xf[-1])/2
    result = minimize(Transfer_Cost, t_ig, args=(X0[:6], Xf[:6], t0, f_area, mu, simSun, simWind, tol, iter_max), method='Powell', bounds=bnds, options={'xtol': 1e-1, 'ftol': 1e-2}, tol=1e-1)
    return np.concatenate((X0, Xf, result.x, np.array([result.fun])), axis=0)
def optimal_dv(opt_result, v0, simSun, simWind):
    r_miss_saved, v0_optimal = Single_Shooting_3_body(opt_result[:6], opt_result[7:13], t0, opt_result[14], mu, simSun, simWind, tol, iter_max)
    X0_modified = np.copy(opt_result[:6])
    X0_modified[3:6] = v0_optimal
    trans_traj = solve_ivp(cr3bp_dynamics, [t0, opt_result[14]], X0_modified, args=(mu,), rtol=1e-6, atol=1e-9, method='LSODA')
    dv0 = v0_optimal - v0
    dvf = opt_result[10:13] - trans_traj.y[3:6, -1]
    return np.concatenate((dv0, dvf))
# Constants
mu = 1.215058560962404E-2  # Mass ratio for Earth-Moon system (DU3/TU2)
mu_E = 1 - mu
mu_m = mu
LU = 389703 # Length unit (km)
TU = 382981 # Time unit (s)
Vconv = 389703/382981 # Velocity unit (km/s) (I am lazy)
t0 = 0.0  # Initial time
# Model parameters
simSun = False
simWind = False
# Optimizations parameters
tol = 1e-6
iter_max = 10
# Optimization
bnds = [(0, None)]
X0_opt = np.zeros((6))
Xf_opt = np.zeros((6))
opt_cost = 1e8
f_area = (3475 + 200) / LU
a_H = 0.157874
X0 = np.array([0.8019851743190237,1.8534361957379693e-23,4.535145602527196e-25,-7.943509473098871e-13,0.5239032339083756,-2.808529682310476e-25]) # Initial condition for departure
v0 = X0[3:6]
P0 = 3.2784950944894966
X0 = np.concatenate((X0, np.array([P0])))
Xf = np.array([1.1301300724055106,4.420007953305004e-28,5.434357403625468e-33,7.034749743192063e-17,0.12946153858401233,-3.638625213535309e-33]) # Final coniditon for arrival
Pf = 3.3959935804560506
Xf = np.concatenate((Xf, np.array([Pf])))
opt_result = optimize_time_of_transfer(X0, Xf, mu_E, mu_m, a_H, simSun, simWind)
dv = optimal_dv(opt_result, v0, simSun, simWind)
# Display total cost
print(norm(dv[:3]) + norm(dv[3:]))