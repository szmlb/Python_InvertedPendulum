# -*- coding: utf-8 -*-

import numpy as np

#factorial
def factorial (n):
    if n <= 1:
        return 1
    else :
        return n * factorial(n-1)

#Discretize A matrix
def discretizeAmat(Amat, dim_of_system, sampling_time, order_of_taylor):
    tmp2 = np.zeros((dim_of_system, dim_of_system))
    for i in range(order_of_taylor):
       tmp1 = np.eye(np.size(Amat)/2)
       for j in range(i):
           tmp1 = tmp1.dot(Amat)
       tmp2 = tmp2 + tmp1 * sampling_time**i / factorial(i)
    return tmp2

#Discretize B matrix
def discretizeBmat(Amat, Bmat, dim_of_system, sampling_time, order_of_taylor):
    division = 100.0 # sampling_time : step = 1000 : 1
    step = sampling_time / division
    tmp = np.zeros((dim_of_system, dim_of_system))
    for i in range(int(division)):
        t = i * step
        tmp = tmp + discretizeAmat(Amat, dim_of_system, t, order_of_taylor) * step
    return tmp.dot(Bmat)

#Discretize A and B matrix simultaneously
def c2d(Amat, Bmat, sampling_time):
    order_of_taylor = 10 #maclaurin approximation till 10th order
    dim_of_system = (int)(np.size(Amat)**(0.5))
    Amat_discretized = discretizeAmat(Amat, dim_of_system, sampling_time, order_of_taylor)
    Bmat_discretized = discretizeBmat(Amat, Bmat, dim_of_system, sampling_time, order_of_taylor)
    return Amat_discretized, Bmat_discretized

#Function for quantization
def quantization(num, resolution, quantized_max_num):
    quantized_num = ((float)(resolution - 1) / quantized_max_num ) * num
    if quantized_num - (float)((int)(quantized_num)) < 0.5 :
        round_value = (float)((int)(quantized_num))        #Omitting if value is under 0.5
    else:
        round_value = (float)((int)(quantized_num + 0.9))  #Omitting if value is over 0.5
    quantized_num = (float)((quantized_max_num * round_value)) / (float)(resolution - 1);
    return quantized_num
