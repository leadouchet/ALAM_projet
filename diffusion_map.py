
import numpy as np 
import random
import matplotlib.pyplot as plt
import pandas as pd 
import math



def propre(x, eps): #x est une matrice type numpy
	nb_ind=len(x)
	#matrice de similarite
	L=np.zeros((nb_ind,nb_ind))
	for i in range(nb_ind) :
		for j in range(i,nb_ind) :
			L[i][j]=math.exp(-sum((x[i]-x[j])**2)/eps)
			L[j][i]=L[i][j]
	#normalisation de la matrice L

	# On commence par creer la matrice D, diagonale et positive
	D=np.zeros(np.shape(L))
	for i in range(nb_ind) :
		D[i,i]=sum(L[i]) 

	#creation de la matrice M normalisee
	M=np.dot(np.linalg.inv(D),L)
	#matrice symetrique Ms
	Ms=np.dot(np.dot(np.sqrt(D),M),np.linalg.inv(np.sqrt(D)))

	#decomposition de la matrice Ms en valeur propres
	decomposition=np.linalg.eig(Ms)
	valpropre=decomposition[0]
	vecpropre=decomposition[1]

	#trie les vecteurs propres dans l ordre decroissants de la valeur absolue de leur valeur propres
	tri=np.concatenate(([valpropre],vecpropre),axis=0)
	valpropre_ordre=np.fliplr(np.sort(np.abs([valpropre])))

	triage=np.zeros(np.shape(tri))
	for i in range(np.shape(valpropre_ordre)[1]) :
		triage[:,i]=tri[:,tri[0,:]==valpropre_ordre[0,i]][:,0]
	valpropre=triage[0,:]
	vecpropre=np.delete(triage,0,axis=0)


	#decomposition de la matrice M=phi*V*rho
	psi=np.dot(np.linalg.inv(np.sqrt(D)),vecpropre)
	phi=np.dot(np.transpose(vecpropre),np.sqrt(D)) #np.linalg.inv(vecpropre) ne fonctionne pas det=0
	return (valpropre,psi,phi)

def diffusion_map(vecproprepsi,valpropre,k,t):
	psit=np.zeros((len(vecproprepsi),k))
	for i in range(k) :
		psit[:,i]=(valpropre[i+1]**t)*vecproprepsi[:,i+1]
	return psit

#def puissancemat(matrice,puissance):
#	if puissance==1:
#		return matrice
#	else :
#		return np.dot(matrice,puissancemat(matrice,puissance-1))


df = pd.read_csv("dCt_values.csv",sep='\t',decimal=",",header=0,index_col=0 )
#normaliser les donnees
x=np.array(df.values)

res=propre(x,1)

diff=diffusion_map(res[1],res[0],1000, 2)

#heatmap
plt.imshow(diff, cmap='hot', interpolation='nearest')
plt.show()
