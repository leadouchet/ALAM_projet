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
	Ms=np.dot(np.dot(np.linalg.inv(np.sqrt(D)),L),np.linalg.inv(np.sqrt(D)))
	#decomposition de la matrice Ms
	valpropre=[np.linalg.eig(Ms)[0]]
	vecpropre=np.linalg.eig(Ms)[1]
	#trie les vecteurs propres dans lordre decroissants de leur valeur propres
	tri=np.concatenate((valpropre,vecpropre),axis=0)
	valpropre=np.fliplr(np.sort(valpropre))
	triage=np.zeros(np.shape(tri))
	for i in range(np.shape(valpropre)[1]) :
		triage[:,i]=tri[:,tri[0,:]==valpropre[0,i]][:,0]
	vecpropre=np.delete(triage,0,axis=0)
	print vecpropre

	#decomposition de la matrice M=phi*V*rho
	V=np.diag(valpropre[0,:])
	phi=np.dot(np.linalg.inv(np.sqrt(D)),vecpropre)
	rho=np.dot(np.transpose(vecpropre),np.sqrt(D)) #np.linalg.inv(vecpropre) ne fonctionne pas det=0
	return (valpropre,phi,rho)

def diffusion_map(vecproprephi,valpropre,k,t):
	phit=np.zeros((len(vecproprephi),k))
	for i in range(k) :
		phit[:,i]=(valpropre[0,i]**t)*vecproprephi[:,i]
	return phit

#def puissancemat(matrice,puissance):
#	if puissance==1:
#		return matrice
#	else :
#		return np.dot(matrice,puissancemat(matrice,puissance-1))


df = pd.read_csv("dCt_values.csv",sep='\t',decimal=",",header=0,index_col=0 )
#normaliser les donnees
x=np.array(df.values)[0:100,0:100]
print np.shape(x)
res=propre(x,1)
res[1]
diff=diffusion_map(res[1],res[0],90,1
print diff
#heatmap
plt.imshow(diff, cmap='hot', interpolation='nearest')
plt.show()
