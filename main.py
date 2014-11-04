# -*- coding: utf-8 -*-
"""
Simulation program
"""
import numpy as np
import pylab as pl
import control as ctrl
from math import *
import utility

gravity_const = 9.8

class InvertedPendulum:

    def __init__(self,
                 Jb,
                 JB,
                 Db,
                 DB,
                 mb,
                 mB,
                 r,
                 l,
                 sampling_time
                ):
        # plant parameters and simulation conditions
        self.Jb = Jb
        self.JB = JB
        self.Db = Db
        self.DB = DB
        self.mb = mb
        self.mB = mB
        self.r = r
        self.l = l

        self.alpha = self.Jb + (self.mb + self.mB) * self.r**2 
        self.beta = self.mB * self.l * self.r
        self.gamma = self.JB + self.mB * self.l**2
        self.delta = self.mB * gravity_const * self.l**2        
        
        self.torque = 0.0
        self.sampling_time = sampling_time
  
        self.xvec = [0.0, 0.0, 5.0 * np.pi / 180.0, 0.0] # [theta theta' phi phi'] 
        self.dxvec = [0.0, 0.0, 0.0, 0.0]

    # calculate derivative of state value
    def calc_derivative(self, torque_reac):

        M11 = self.alpha
        M12 = self.alpha + self.beta * np.cos(self.xvec[2])
        M21 = self.alpha + self.beta * np.cos(self.xvec[2])
        M22 = self.alpha + 2 * self.beta * np.cos(self.xvec[2]) + self.gamma
        D1 = self.Db * self.xvec[1]
        D2 = self.DB * self.xvec[3]        
        C1 = - self.beta * np.sin(self.xvec[2]) * self.xvec[3]**2
        C2 = - self.beta * np.sin(self.xvec[2]) * self.xvec[3]**2
        G1 = 0.0
        G2 = -mB * gravity_const * self.l * np.sin(self.xvec[2])

        Mmat = np.array([[M11, M12], [M21, M22]])
        Dmat = np.array([[D1], [D2]])
        Cmat = np.array([[C1], [C2]])
        Gmat = np.array([[G1], [G2]])
        Umat = np.array([[self.torque-torque_reac], [0.0]])
        dq_vec = np.linalg.solve(Mmat, Umat- Dmat - Cmat - Gmat)
        #print dq_vec[0][0]

        return (self.xvec[1], dq_vec[0][0], self.xvec[3], dq_vec[1][0])

    # update 
    def update(self):
        self.xvec[0] = self.xvec[0] + self.dxvec[0] * self.sampling_time
        self.xvec[1] = self.xvec[1] + self.dxvec[1] * self.sampling_time
        self.xvec[2] = self.xvec[2] + self.dxvec[2] * self.sampling_time
        self.xvec[3] = self.xvec[3] + self.dxvec[3] * self.sampling_time

class Controller:
    def __init__(self, dt, inverted_pendulum):
        self.dt = dt
        #for pid
        self.error_integral = 0
        self.error_diff = 0
        self.error_diff_ = 0
        self.error_ = 0
        #for DOB
        self.tmp1_dob = 0.0
        self.tmp2_dob = 0.0            

        #system matrices
        self.Ac = np.array([[0, 1, 0, 0], [0, -inverted_pendulum.Db * (1/inverted_pendulum.gamma+1/inverted_pendulum.alpha), -inverted_pendulum.delta / inverted_pendulum.gamma, inverted_pendulum.DB / inverted_pendulum.gamma], [0, 0, 0, 1], [0, inverted_pendulum.Db / inverted_pendulum.gamma, inverted_pendulum.delta / inverted_pendulum.gamma, inverted_pendulum.DB / inverted_pendulum.gamma]])
        self.Bc = np.array([[0], [1/inverted_pendulum.gamma + 1/inverted_pendulum.alpha], [0], [-1 / inverted_pendulum.gamma]])
        self.Cc = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
        self.Dc = 0.0        
        self.Uc = ctrl.ctrb(self.Ac, self.Bc)
                
        #discretized system matrices
        self.Ad, self.Bd = utility.c2d(self.Ac, self.Bc, self.dt)
        self.Cd = self.Cc
        self.Dd = self.Dc
        
        self.Qlqr = 200.0 * np.eye(4)
        #self.Qlqr = 100000.0 * np.eye(3)
        
        #self.Qlqr = np.array([[1.0, 0.0, 0.0, 0.0],[0.0, 1.0, 0.0, 0.0],[0.0, 0.0, 1000.0, 0.0],[0.0, 0.0, 0.0, 1000.0]])
        self.Rlqr = np.array([1.0])
        self.Klqr, Sricatti, Eigen = ctrl.lqr(self.Ac, self.Bc, self.Qlqr, self.Rlqr)
        #self.Kdlqr, Sdricatti, Edigen = ctrl.dlqr(self.Ad, self.Bd, self.Qlqr, self.Rlqr)

    #classical PID controller            
    def pid_controller(self, error, Kp, Ki, Kd, wc_diff):
        self.error_diff = (self.error_diff_ + wc_diff * (error - self.error_)) / (1.0 + wc_diff * self.dt) 
        self.error_integral = self.error_integral + error * self.dt
        
        input = Kp * error + Kd * self.error_diff + Ki * self.error_integral;

        self.error_diff_ = self.error_diff
        self.error_ = error        

        return input        
 
    def full_state_feedback(self, Kgain):
        input = 0
        for i in range(len(inverted_pendulum.xvec)):
            input = input - Kgain[0, i] * inverted_pendulum.xvec[i] 
        return input
 
    #disturbance observer
    def simple_dob(self, tau, dq, wc_dob):
        self.tmp1_dob = tau + inverted_pendulum.inertia * wc_dob * dq;
        self.tmp2_dob = (self.tmp2_dob + self.dt * wc_dob * self.tmp1_dob) / (1 + self.dt * wc_dob);
        dist = self.tmp2_dob - inverted_pendulum.inertia * wc_dob * dq; 

        return dist

# simulation parameters
Jb=0.05
JB=0.1
Db=0.0
DB=0.0
mb = 1.0
mB = 3.0
rw = 0.1
lb = 0.5
torque = 0.0

sampling_time = 0.001
control_sampling_time=0.001
simulation_time = 10

# state parameters and data for plot
xvec0_data=[]
xvec1_data=[]
xvec2_data=[]
xvec3_data=[]

theta_cmd_data=[]
phi_cmd_data=[]

# inverted_pendulum simulation object
inverted_pendulum = InvertedPendulum(Jb, JB, Db, DB, mb, mB, rw, lb, sampling_time)
phi_controller = Controller(control_sampling_time, inverted_pendulum)
theta_controller = Controller(control_sampling_time, inverted_pendulum)
lqr_controller = Controller(control_sampling_time, inverted_pendulum)

# main loop 10[sec]
for i in range(simulation_time*(int)(1/inverted_pendulum.sampling_time)):
    time = i * inverted_pendulum.sampling_time

    control_delay = (int)(lqr_controller.dt / inverted_pendulum.sampling_time) #[sample]
    if i % control_delay == 0:
        """ controller """
        # definition for control parameters
        if i == 0 :
            dist = 0.0
            torque_reac = 0.0
        else:
            torque_reac = 0.0
            
        """
        # PID control
        kp_phi = 100.0
        kd_phi = 10.0
        phi_cmd = 0.0
        phi_res = inverted_pendulum.xvec[2]
        phi_error = phi_cmd - phi_res
        phi_torque = -phi_controller.pid_controller(phi_error, kp_phi, 0, kd_phi, 500.0)        
        
        kp_theta = 50.0
        kd_theta = 10.0
        theta_cmd = 0.0
        theta_res = inverted_pendulum.xvec[0]
        theta_error = theta_cmd - theta_res        
        theta_torque = theta_controller.pid_controller(theta_error, kp_theta, 0, kd_theta, 500.0)        
        theta_torque = 0.0 #        
        inverted_pendulum.torque = phi_torque + theta_torque
        """        
        
        #LQR
        inverted_pendulum.torque = lqr_controller.full_state_feedback(lqr_controller.Klqr)
        #inverted_pendulum.torque = lqr_controller.Klqr[0, 0] * inverted_pendulum.xvec[0] + lqr_controller.Klqr[0, 1] * inverted_pendulum.xvec[2] + lqr_controller.Klqr[0, 2] * inverted_pendulum.xvec[3]

        #data update
        xvec0_data.append(inverted_pendulum.xvec[0])
        xvec1_data.append(inverted_pendulum.xvec[1])
        xvec2_data.append(inverted_pendulum.xvec[2])
        xvec3_data.append(inverted_pendulum.xvec[3])              
        theta_cmd_data.append(theta_cmd)        
        phi_cmd_data.append(phi_cmd)

        """ controller end """

    """ plant """
    #reaction torque
    if time == 8.0:
        torque_reac = 100.0
    else:
        torque_reac = 0.0

    # derivative calculation
    inverted_pendulum.dxvec = inverted_pendulum.calc_derivative(torque_reac)
    # euler-integration
    inverted_pendulum.update()

    """ plant end """

# data plot
time_data = np.arange(0, simulation_time, lqr_controller.dt)
pl.figure()
pl.plot(time_data, theta_cmd_data[:], label="theta cmd")
pl.plot(time_data, xvec0_data[:], label="theta res")
pl.legend()
pl.grid()
pl.xlabel('time [s]')
pl.ylabel('angle [rad]')

time_data = np.arange(0, simulation_time, lqr_controller.dt)
pl.figure()
pl.plot(time_data, phi_cmd_data[:], label="phi cmd")
pl.plot(time_data, xvec2_data[:], label="phi res")
pl.legend()
pl.grid()
pl.xlabel('time [s]')
pl.ylabel('angle [rad]')

pl.show()
