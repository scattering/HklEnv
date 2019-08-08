import numpy as np
import matplotlib.pyplot as plt

#This filename is where the results have been stored (storspot)
filename = "/wrk/kmm11/howdy/"

#Choose a folder to store results in
imagestore = "/home/kmm11/resultimgs/"

#initial file load
numTrials = 128
numEnvs = 1
start_trial = 0
converg = .359

#Though some of this graphing has been created to allow multiple environments
#a significant portion will only graph one environment
endzs = [[] for i in range(numEnvs)]
data = [[[] for i in range(numTrials)] for j in range(numEnvs)]
hs = [[[] for i in range(numTrials)] for j in range(numEnvs)]
ks = [[[] for i in range(numTrials)] for j in range(numEnvs)]
ls = [[[] for i in range(numTrials)] for j in range(numEnvs)]
chis = [[[] for i in range(numTrials)] for j in range(numEnvs)]
lzeroes = []
convergs = []
epLength = [] #the repeats will only be visualized for the first environment currently
for i in range(1,numTrials+1):
    for j in range (0, numEnvs):
        data[j][i-1] = np.loadtxt(filename +  "zLog-" + str(i) + "_" + str(j) + ".txt")
        #hs[j][i-1] = np.loadtxt("hLog-" + str(i) + "_" + str(j) + ".txt")
        #ks[j][i-1] = np.loadtxt("kLog-" + str(i) + "_" + str(j) + ".txt")
        ls[j][i-1] = np.loadtxt(filename + "lLog-" + str(i) + "_" + str(j) + ".txt")
        chis[j][i-1] = np.loadtxt(filename + "chiLog-" + str(i) + "_" + str(j) + ".txt")

#Parsing array information
for j in range (0, len(data)):
    for i in range (start_trial, len(data[j])): 
        plt.plot(data[j][i], label = str(i))
        endzs[j].append(data[j][i][-1])
        k = 0
        while data[j][i][k] < converg and k < len(data[j][i]) - 1:
            k += 1
        convergs.append(k)
        l = list(ls[j][i])
        lzeroes.append(l.count(0))
    plt.xlabel("measurement steps")
    plt.ylabel("z value")
    plt.savefig(imagestore + "zs.png")
    plt.close()
    
    
plt.plot(convergs, 'ro')
plt.xlabel("episodes")
plt.ylabel("step at which convergence is reached")
plt.savefig(imagestore + "convergs.png")    
plt.close()

plt.plot(lzeroes, 'ko')
plt.xlabel("episodes")
plt.ylabel("num l zeroes")
plt.savefig(imagestore + "lzeroes.png")
plt.close()
    
for i in range (start_trial, len(endzs)):
    plt.plot(endzs[i], 'bo')
    plt.xlabel("episodes")
    plt.ylabel("z value")

plt.savefig(imagestore + "endzs.png")
plt.close()
    
'''
for j in range (0, len(data)):
    for i in range (start_trial, len(hs[j])):
        plt.plot(hs[j][i], 'bo')
    plt.xlabel("measurement steps")
    plt.ylabel("hs")
    plt.show()
    plt.close()

for j in range (0, len(ks)):
    for i in range (start_trial, len(ks[j])):
        plt.plot(ks[j][i], 'co')
    plt.xlabel("measurement steps")
    plt.ylabel("ks")
    plt.show()
    plt.close()
'''
for j in range (0, len(data)):
    for i in range (start_trial, len(ls[j])):
        plt.plot(ls[j][i], 'mo')
    plt.xlabel("measurement steps")
    plt.ylabel("ls")
    plt.savefig(imagestore + "ls.png")
    plt.close()
    
for j in range (0, len(data)):
    for i in range (start_trial, len(chis[j])):
        plt.plot(chis[j][i], 'co')
    plt.xlabel("measurement steps")
    plt.ylabel("chis")
    plt.savefig(imagestore + "chis" + str(j) + ".png")
    plt.close()
