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
                 mb,
                 mB,
                 r,
                 l,
                 sampling_time
                ):
        # plant parameters and simulation conditions
        self.Jb = Jb
        self.JB = JB
        self.mb = mb
        self.mB = mB
        self.r = r
        self.l = l
        
        self.damping = damping
        self.torque = 0.0
        self.sampling_time = sampling_time
  
        self.xvec = [0.0, 0.0, 0.0, 0.0] # [theta theta' phi phi'] 
        self.dxvec = [0.0, 0.0, 0.0, 0.0]

    # calculate derivative of state value
    def calc_derivative(self, torque_reac):
        alpha = self.Jb + (self.mb + self.mB) * self.r**2 
        beta = self.mB * self.l * self.r
        gamma = self.JB + self.mB * self.l**2
        M11 = alpha
        M12 = alpha + beta * np.cos(self.xvec[2])
        M21 = alpha + beta * np.cos(self.xvec[2])
        M22 = alpha + 2 * beta * np.cos(self.xvec[2]) + gamma
        C1 = - beta * np.sin(self.xvec[2]) * self.xvec[3]**2
        C2 = - beta * np.sin(self.xvec[2]) * self.xvec[3]**2
        G1 = 0.0
        G2 = -mB * gravity_const * self.r * np.sin(self.xvec[2])

        Mmat = np.array([[M11, M12], [M21, M22]])
        Cmat = np.array([[C1], [C2]])
        Gmat = np.array([[G1], [G2]])
        Umat = np.array([[self.torque-torque_reac], [0.0]])
        dq_vec = np.linalg.solve(Mmat, Umat - Cmat - Gmat)
        print dq_vec[0][0]

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

    #classical PID controller            
    def pid_controller(self, error, Kp, Ki, Kd, wc_diff):
        self.error_diff = (self.error_diff_ + wc_diff * (error - self.error_)) / (1.0 + wc_diff * self.dt) 
        self.error_integral = self.error_integral + error * self.dt
        
        input = Kp * error + Kd * self.error_diff + Ki * self.error_integral;

        self.error_diff_ = self.error_diff
        self.error_ = error        

        return input        
 
    #disturbance observer
    def simple_dob(self, tau, dq, wc_dob):
        self.tmp1_dob = tau + inverted_pendulum.inertia * wc_dob * dq;
        self.tmp2_dob = (self.tmp2_dob + self.dt * wc_dob * self.tmp1_dob) / (1 + self.dt * wc_dob);
        dist = self.tmp2_dob - inverted_pendulum.inertia * wc_dob * dq; 

        return dist

# simulation parameters
Jb=0.1
JB=0.0
mb = 1.0
mB = 3.0
rw = 0.1
lb = 0.5
torque = 0.0

sampling_time = 0.001
control_sampling_time=0.001
simulation_time = 3

# state parameters and data for plot
xvec0_data=[]
xvec1_data=[]
xvec2_data=[]
xvec3_data=[]

theta_cmd_data=[]
phi_cmd_data=[]

# inverted_pendulum simulation object
inverted_pendulum = InvertedPendulum(Jb, JB, mb, mB, rw, lb, sampling_time)
phi_controller = Controller(control_sampling_time, inverted_pendulum)
theta_controller = Controller(control_sampling_time, inverted_pendulum)

# main loop 10[sec]
for i in range(simulation_time*(int)(1/inverted_pendulum.sampling_time)):
    time = i * inverted_pendulum.sampling_time

    control_delay = (int)(controller.dt / inverted_pendulum.sampling_time) #[sample]
    if i % control_delay == 0:
        """ controller """
        # definition for control parameters
        if i == 0 :
            dist = 0.0
            torque_reac = 0.0
        else:
            torque_reac = 0.0
            
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
    if time == 1.0:
        torque_reac = 10.0
    else:
        torque_reac = 0.0

    # derivative calculation
    inverted_pendulum.dxvec = inverted_pendulum.calc_derivative(torque_reac)
    # euler-integration
    inverted_pendulum.update()

    """ plant end """

# data plot
time_data = np.arange(0, 3, controller.dt)
pl.figure()
pl.plot(time_data, theta_cmd_data[:], label="theta cmd")
pl.plot(time_data, xvec0_data[:], label="theta res")
pl.legend()
pl.grid()
pl.xlabel('time [s]')
pl.ylabel('angle [rad]')

pl.figure()
pl.plot(time_data, phi_cmd_data[:], label="phi cmd")
pl.plot(time_data, xvec2_data[:], label="phi res")
pl.legend()
pl.grid()
pl.xlabel('time [s]')
pl.ylabel('angle [rad]')

pl.show()
