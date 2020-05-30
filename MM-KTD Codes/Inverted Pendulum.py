import decimal
from decimal import Decimal

import pandas as pd
from mpmath import *
import seaborn as sns
import matplotlib.pyplot as plt
mp.dps=300
# print('ans',exp(111997.82249403))
import gym
import gym_foo
env = gym.make('foo-v0')
import numpy as np
from numpy.linalg import inv
# import bigfloat
#### policy logic


def  policy_logic(observation,Sigma,Mu,Theta):
     zeros=np.zeros((10,1)) ###phi dimension
     S=observation
     L=9 ### number of RBF
     def PHI_function(S):
         phi_list=[1]
         for i in range(L):
             fg=np.reshape(Mu[i, :],(2,1)) ##2*1
             A=np.matmul((S - fg).T, np.linalg.pinv(Sigma[:,:,i]))
             phi_list.append(exp(-.5*np.matmul(A,(S- fg))[0,0]))
         y = np.array(phi_list)
         phi = np.reshape(y, (10, 1))

         return phi,phi_list



    ###########   finding H_K
     S_next=np.zeros((2,3))
     A=[]
     A_test=[]
     H_k=np.zeros((3,30))

     fetures_k_list_0 = []
     features_next_k_list_0 = []
     fetures_k_list_1 = []
     features_next_k_list_1 = []
     fetures_k_list_2 = []
     features_next_k_list_2 = []


     for i in range(3):

         action=i
         if i == 0:
             # fetures_k[:,i] =[p for p in np.concatenate((np.concatenate((PHI_function(S), zeros), axis=0), zeros), axis=0)]
             phi,phi_list=PHI_function(S)
             fetures_k_list_0=phi_list+[0]*20
             observation, reward, done, info = env.step(action)
             S_next[:, i] = observation
             pf=np.reshape(S_next[:, i],(2,1))
             # features_next_k[:, i] = [p for p in np.concatenate((np.concatenate((PHI_function(pf), zeros), axis=0), zeros), axis=0)]
             phi_next, phi_list_next = PHI_function(pf)
             features_next_k_list_0 =  phi_list_next+[0]*20
             fetures_k_list = fetures_k_list_0
             features_next_k_list = features_next_k_list_0
         elif i == 1:
             phi, phi_list = PHI_function(S)
             fetures_k_list_1 = [0] * 10 + phi_list+[0] * 10
             observation, reward, done, info = env.step(action)
             S_next[:, i] = observation
             pf = np.reshape(S_next[:, i], (2, 1))
             # features_next_k[:, i] = [p for p in np.concatenate((np.concatenate((PHI_function(pf), zeros), axis=0), zeros), axis=0)]
             phi_next, phi_list_next = PHI_function(pf)
             features_next_k_list_1 = [0] * 10 + phi_list_next+[0]*10
             fetures_k_list = fetures_k_list_1
             features_next_k_list = features_next_k_list_1
         else:
             phi, phi_list = PHI_function(S)
             fetures_k_list_2 = [0] * 20 + phi_list
             observation, reward, done, info = env.step(action)
             S_next[:, i] = observation
             pf = np.reshape(S_next[:, i], (2, 1))
             # features_next_k[:, i] = [p for p in np.concatenate((np.concatenate((PHI_function(pf), zeros), axis=0), zeros), axis=0)]
             phi_next, phi_list_next = PHI_function(pf)
             features_next_k_list_2 = [0] * 20 + phi_list_next
             fetures_k_list=fetures_k_list_2
             features_next_k_list=features_next_k_list_2   ##1*30

         A.append(np.matmul(features_next_k_list, Theta))
         A_test.append(np.matmul(fetures_k_list, Theta))
         ####

     a_max=A.index(max(A))
     # print('a_max',a_max)
     a_max_test=A_test.index(max(A_test))
     # print('a_max_test',a_max_test)
     Q_function_test=max(A_test)
     # print('Q',Q_function_test)
     if a_max==0:
        phi_max=features_next_k_list_0
     elif a_max==1:
        phi_max = features_next_k_list_1
     else:
        phi_max = features_next_k_list_2


     phi_max_1=np.array(phi_max)
     phi_max_f=np.reshape(phi_max_1,(30,1))

     B=[]
     H_total=np.zeros((3,1))
     for i in range(3):
         if i==0:
            H_k[i,:]=[a_i-.95*b_i for a_i,b_i in zip(fetures_k_list_0,phi_max)]
         elif i==1:
            H_k[i, :] = [a_i-.95*b_i for a_i,b_i in zip(fetures_k_list_1,phi_max)]
         else:
             H_k[i, :] = [a_i-.95*b_i for a_i,b_i in zip(fetures_k_list_2,phi_max)]

         C=np.reshape(H_k[i,:],(1,30)) ###1*30
         # print(np.matmul(C,C.T))
         H_total[i]= [k for k in np.matmul(C,C.T)]
         B.append(np.matmul(C,C.T))

     a_final_max=B.index(max(B))
     # print('a_final_max',a_final_max)
     # print('B',B)
     if a_final_max==0:
        H_return=[a_i-.95*b_i for a_i,b_i in zip(fetures_k_list_0,phi_max)]
        phi_current=fetures_k_list_0
     elif a_final_max==1:
        H_return = [a_i-.95*b_i for a_i,b_i in zip(fetures_k_list_1,phi_max)]
        phi_current = fetures_k_list_1
     else:
        H_return = [a_i-.95*b_i for a_i,b_i in zip(fetures_k_list_2,phi_max)]
        phi_current = fetures_k_list_2



     H_return_f=np.reshape(H_return,(1,30))
     phi_current_f=np.reshape(phi_current, (30, 1))

     return a_final_max,a_max_test,H_return_f,phi_current_f,phi_max_f,Q_function_test



############Training
Theta_initial=np.zeros((30,1))  ##Theat_0
######reset observation
P_initial = 10 * np.identity(30, float)
F = np.identity(30)
Q = (10 ^ (-7)) * np.identity(30)
Sigma = np.zeros((2, 2, 9))
for i in range(9):
    Sigma[:, :, i] = 2 * np.identity(2, float)

Mu = np.array([[-np.math.pi / 4, -.5], [-np.math.pi / 4, 0], [-np.math.pi / 4, 0.5], [0, -.5], [0, 0], [0, .5],
               [np.math.pi / 4, -.5], [np.math.pi / 4, 0], [np.math.pi / 4, .5]])
lambda_Mu = 200
lambda_Sigma = 100
R = [.1, .2, .5, 1, 2, 5, 10, 20, 50, 100]
I = np.identity(30)
episode_index = []
sample_index = []
Theta_total_train=[]
angle_total_train=[]
Q_total_train=[]
number_episode=1
# #
for episode in range(number_episode):  ###############################start of episode

    step=0
    S_k = env.reset()
    done = False
    # K=np.zeros((30,len(R)))
    theta_m=np.zeros((30,len(R)))

    # theta_m=[]
    Pi=np.zeros((30,30,len(R)))
    w=np.zeros(len(R))
    S_k = np.reshape(S_k, (2, 1))  ##2*1
    # number_sample=10
    while not done:
    # for t in range(number_sample):
        action,action_test, H,phi_current,phi_max,Q_function_test = policy_logic(S_k, Sigma, Mu,Theta_initial)
        S_next, reward, done, info = env.step(action)
        S_next=np.reshape(S_next,(2,1))   ##2*1


        ##Kalman Filter
        Theta_previous_k=np.matmul(F,Theta_initial)
        P_previou_k=np.matmul(np.matmul(F,P_initial),F.T)+Q

        Theta_update=np.zeros((30,1))
        W = 0
        for m in range(len(R)):
            # el=np.array(1,float)
            temp=1/(np.matmul(np.matmul(H,P_previou_k),H.T)+R[m]) ###scalar
            K=np.matmul(P_previou_k, H.T)*temp ###30*1
            theta_m[:,m]=[i for i in Theta_previous_k+K*(reward-np.matmul(H,Theta_previous_k))]
            g=np.reshape(theta_m[:,m],(30,1))
            temp2=np.matmul((I-np.matmul(K,H)),P_previou_k)
            Pi[:,:,m]=np.matmul(temp2,(I-np.dot(K,H)).T)+np.matmul(R[m]*K,K.T) ##30*30
            temp3=(reward-np.matmul(H,Theta_previous_k)).T*temp #scalar
            decimal.getcontext().prec = 100
            el=-.5*(temp3*(reward-np.matmul(H,Theta_previous_k)))
            w[m] = exp(el[0][0])
            W=W+w[m]
            # print('el',el)
            Theta_update=Theta_update+w[m]*g  ##30*1
        C=1/(W)
        w=w*C

        P_update=np.zeros((30,30))
        for m in range(len(R)):
            g = np.reshape(theta_m[:, m], (30, 1))
            temp4=Pi[:,:,m]+np.matmul(g-Theta_update,(g-Theta_update).T)  ##30*30
            P_update=P_update+w[m]*temp4

        ####Update RBFs

        S=np.matmul(phi_current.T,Theta_update)-reward-.95*np.matmul(phi_max.T,Theta_update)
        S=S**2
        # print('S',S)
        for j in range(len(Mu)):
            fg = np.reshape(Mu[j, :], (2, 1))  ##2*1
            temp5=np.matmul(np.linalg.pinv(Sigma[:,:,j]),S_k-fg) **2*1
            temp6=np.matmul(np.matmul(temp5,(S_k-fg).T),np.linalg.pinv(Sigma[:,:,j]))  #2*2

            Q=np.matmul(phi_current.T, Theta_update) ##scalar
            df=2*lambda_Sigma*np.sqrt(S)*Q*temp6 #2*2
            if np.matmul(phi_current.T, Theta_update) * S > 0:
                Sigma[:,:,j]=Sigma[:,:,j]-df
            else:
                Mu[j,:]=[we for we in fg-2*lambda_Sigma*(np.sqrt(S))*np.matmul(phi_current.T,Theta_update)*temp5]

        Theta_total_train.append(S_next[0])
        angle_total_train.append(S_next[1])
        Q_total_train.append(Q_function_test)
        angle=S_next[0]
        Theta_initial = Theta_update
        # print(Theta_update)
        # print(Theta_initial)
        P_initial = P_update
        S_k=S_next
        step=step+1

    print("Episode:", episode, ",finished at:", step, ',angle', angle, ',reward', reward)
    episode_index.append(episode)
    sample_index.append(step)



print('episode_index',episode_index)
print('sample_index',sample_index)


############test
episode_index_test=[]
sample_index_test=[]
number_episode_test=50
number_sample_test = 500
Theta_total_test=[]
velocity_total_test=[]
Q_total_test=[]
m=0
for i in range(number_episode_test):
    S_k = env.reset()
    S_k = np.reshape(S_k, (2, 1))  ##2*1
    for t in range(number_sample_test):
        action,action_test, H,phi_current,phi_max,Q_function_test = policy_logic(S_k, Sigma, Mu,Theta_initial)
        S_next, reward, done, info = env.step(action_test)
        S_next=np.reshape(S_next,(2,1))   ##2*1
        S_k=S_next
        # print('action_test', action_test)
        # print(S_next[0])
        Theta_total_test.append(S_next[0])
        velocity_total_test.append(S_next[1])
        Q_total_test.append(Q_function_test)
        m=m+1
        if done:
            print("Episode_test:", i, ",finished at:", t, ',angle_test', S_next[0], ',reward_test', reward)
            episode_index_test.append(i)
            sample_index_test.append(t)
            break



print(Q_total_test)


####################   Q-values:
Theta=np.arange(-pi/2,pi/2,pi/20)
velocity=np.arange(-2,2,.4)
V_value=[]

for i in range(len(velocity)):
    for j in range(len(Theta)):
        S_k=[Theta[j],velocity[i]]
        action,action_test, H,phi_current,phi_max,Q_function_test =policy_logic(S_k, Sigma, Mu, Theta_initial)
        V_value.append(Q_function_test)
