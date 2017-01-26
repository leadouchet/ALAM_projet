
import numpy as np 
import random
import csv
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D



def propre(x, eps): #x est une matrice type numpy
	nb_ind=len(x)

	#matrice de similarite
	L=np.zeros((nb_ind,nb_ind))
	for i in range(nb_ind) :
		for j in range(i,nb_ind) :
			L[i][j]=math.exp(-np.linalg.norm(x[i]-x[j])**2/eps)
			L[j][i]=L[i][j]


	
	#normalisation de la matrice L

	# On commence par creer la matrice D, diagonale et positive
	D=np.zeros(np.shape(L))
	Dp=np.zeros(np.shape(L))
	for i in range(nb_ind) :
		D[i,i]=sum(L[i]) 
		Dp[i][i]=D[i,i]**(-0.5)


	#creation de la matrice M normalisee
	M=np.dot(np.linalg.inv(D),L)

	#matrice symetrique Ms
	Ms=np.dot(Dp,np.dot(L,Dp))


	#decomposition de la matrice Ms en valeur propres
	#calcule des valeurs propres
	valpropre,vecpropre=np.linalg.eig(Ms)

	#trie les vecteurs propres dans l ordre decroissants de la valeur absolue de leur valeur propres
	tri=np.concatenate(([valpropre],vecpropre),axis=0)
	valpropre_ordre=sorted(valpropre,reverse=True)

	triage=np.zeros(np.shape(tri))
	for i in range(len(valpropre_ordre)) :
		triage[:,i]=tri[:,tri[0,:]==valpropre_ordre[i]][:,0]
	valpropre=triage[0,:]
	vecpropre=np.delete(triage,0,axis=0)


	#decomposition de la matrice M=phi*V*rho
	psi=np.dot(np.linalg.inv(np.sqrt(D)),vecpropre)
	phi=np.dot(np.sqrt(D),np.transpose(vecpropre)) #np.linalg.inv(vecpropre) ne fonctionne pas det=0
	return (valpropre,psi,phi)



def diffusion_map(vecproprepsi,valpropre,k,t):
	psit=np.zeros((len(vecproprepsi),k))
	for i in range(k) :
		psit[:,i]=(valpropre[i+1]**t)*vecproprepsi[:,i+1]
	return psit


#df = pd.read_csv("dCt_values.tab",sep='\t',decimal=",",header=0,index_col=0 )

#normaliser les donnees


fileName = "dCt_values.tab"
head = []
data = []
cell = []
with open(fileName, 'rb') as f:
    reader = csv.DictReader(f, delimiter = '\t')
            # create a dict out of reader, converting all values to
            # integers
    for i, row in enumerate(list(reader)):
        data.append([])
        for key, value in row.iteritems():
            if key == "Cell":
                cell.append(value)
            else:                                                                   
                data[i].append(float(value))
        for key, value in row.iteritems():
            if key!="Cell":
                head.append(key)

x=np.array(data)

res=propre(x,0.1)
diff=diffusion_map(res[1],res[0],4, 10)



# graphes 3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(diff[:,0], diff[:,1],diff[:,3])
plt.show()

