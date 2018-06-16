from qutip import *
import numpy as np 
import matplotlib.pyplot as plt
import math
from skimage.feature import peak_local_max
from pathos import pools
import pickle
import os
import glob
import sys

saveDir = 'peaks'

EList = np.linspace(0.00125,0.00300,3)
wdList = np.linspace(10.6070,10.6109,3)

WORKERS = 9

def routine(runParam):
    
    E_i = runParam[0]
    wd_i = runParam[1]
    saveIdx = runParam[2]

    print('running param set {} {}'.format(E_i, wd_i))
    
    # Define all the variables 
    kappa = 0.0012 # 0.0012 
    gJC = 0.3347 # 0.3347
    wc = 10.5665 # Cavity frequency/detuning 10.5665
    w0 = 8.1831 # Qubit frequency 8.1831
    gamma=0.0001 # 0.0001
    Emax =0.01 # 0.01
    EN=1
    #E = 0.01 #0.01(original) # Epsilon
    E = E_i
    N = 50 #50
    nloop = 1

    #wd = 10.6005 # Driving frequency (10.6005 original)
    wd = wd_i
    wlist = np.linspace(wd, wd,nloop)

    # Identity operators are defined for the space of the light field and the space of the atom

    ida = identity(N)
    idatom = identity(2)

    # Define cavity field and atomic operators

    a  = tensor(destroy(N),idatom)
    sm = tensor(ida,sigmam())

    # Hamiltonian # Reminder check hamiltonian from caltech paper

    H1= 1*gJC*(a.dag()*sm + sm.dag()*a) + 1*E*(a+a.dag())

    # Collapse Operators

    C1    = math.sqrt(kappa)*a
    C2    = math.sqrt(gamma)*sm

    C1dC1 = C1.dag()*C1
    C2dC2 = C2.dag()*C2

    # Calculate the Liouvillian

    L1 = spre(C1)*spost(C1.dag())-0.5*spre(C1dC1)-0.5*spost(C1dC1)
    L2 = spre(C2)*spost(C2.dag())-0.5*spre(C2dC2)-0.5*spost(C2dC2)
    L12  = L1+L2

    gQ=math.sqrt(4)
    xvec=  np.arange(-10,10.01,0.025) # 0.025
    yvec = np.arange(-10,10.01,0.025) 

    #print("epsilon", E)
    #print("and drive frequency", wd)
    #print("kappa", kappa)
    k=0
    while k < nloop :
        wl = wlist[k]    
        H = (w0-wl)*(sm.dag()*sm) + (wc-wl)*(a.dag()*a) + H1    
        LH = -complex(0,1) * (spre(H) - spost(H))
        L_final = LH + L12

        # Steady States

        rhoss = steadystate(L_final)
        rhosscav=ptrace(rhoss,0)
        rhocavsq=rhosscav*rhosscav

        #subplot(ceil(sqrt(nloop)), 
        #ceil(sqrt(nl oop)), k)
#         fig, ax = plt.subplots()
        Q3 = qfunc(rhosscav,xvec,yvec,gQ)
#         c = ax.contourf(xvec, yvec, np.real(Q3), 500, cmap=plt.cm.get_cmap('winter'))

#         ax.set_xlim([-3.5, 6]) # -3.5 to 6
#         ax.set_ylim([-4, 3])
#         plt.colorbar(c, ax=ax)
#         plt.xlabel('x')
#         plt.ylabel('y')

        k += 1
        #contour(xvec,yvec,real(Q3), 500)
        #plt.plot(xvec,yvec)
        #plt.show()
        #print(rhosscav)
        #print(rhoss)
    
    coordinates = peak_local_max(Q3)
    #print(coordinates)
    #print (Q3[coordinates])
    
    infoPacket = {}
    infoPacket['E_i'] = E_i
    infoPacket['wd_i'] = wd_i
    infoPacket['coors'] = coordinates
    peaks = []
    for coor in coordinates:
        peaks.append(Q3[coor[0], coor[1]])
    infoPacket['peaks'] = peaks
    
    
    #infoPacket['peaks'] = Q3[coordinates]
    #infoPacket['Q3'] = Q3
    #infoPacket['xvec'] = xvec
    #infoPacket['yvec'] = yvec
    #return infoPacket

    pickle.dump(infoPacket, open(os.path.join(saveDir,'{}.qpeak'.format(saveIdx)),'wb'))
            

saveIdx = 0

if os.path.exists(saveDir):
    print('SAVE FOLDER ALREADY EXISTS. MOVE/DELETE IT AND TRY AGAIN!')
    sys.exit(1)

totalL = len(EList)*len(wdList)
runParams = []
for E_i in EList:
    for wd_i in wdList:
	runParams.append([E_i, wd_i, saveIdx])
        #infoPacket = routine(E_i,wd_i)
        
	#print('{}/{}'.format(saveIdx,totalL))

        #if not os.path.exists(saveDir):
        #    os.makedirs(saveDir)        
        #pickle.dump(infoPacket, open(os.path.join(saveDir,'{}.qpeak'.format(saveIdx)),'wb'))
        saveIdx += 1

if not os.path.exists(saveDir):
    os.makedirs(saveDir)

mapping = pools.ProcessPool(WORKERS).map
mapping(routine, runParams)
