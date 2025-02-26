import numpy as np
B=np.eye(4)
N=np.array([[1,0,0,0],[4,1,0,0],[8,4,1,0],[16,8,4,1]])
cb=np.zeros(4)
cn=np.array([-8,-4,-2,-1])
b=np.array([5,25,125,625])
Bset,Nset=[4,5,6,7],[0,1,2,3]
iter=1
while(((cn-cb@(np.linalg.inv(B)@N))<0).any()):
	x=np.zeros(len(Bset)+len(Nset))
	x[Bset]=b@np.linalg.inv(B.T)
	print("Iteration: ",iter,"Corner point: ",x)
	q=np.argmin(cn-cb@(np.linalg.inv(B)@N))
	min,j=np.inf,-1
	z=False
	for i in range(len(Bset)):
		if N[i,q]>0:
			j=i if min>b[i]/N[i,q] else j
			min=b[i]/N[i,q] if min>b[i]/N[i,q] else min
			z=True
	if z==False:
		print("Unbounded")
		exit()
	Bset[j],Nset[q]=Nset[q],Bset[j]
	t=B[:,j].copy()
	B[:,j]=N[:,q]
	N[:,q]=t
	cb[j],cn[q]=cn[q],cb[j]
	N=np.linalg.inv(B)@N
	cn=cn-cb@(np.linalg.inv(B)@N)
	b=b@np.linalg.inv(B.T)
	B=np.eye(len(Bset))
	cb=np.zeros(len(Bset))
	iter+=1
x=np.zeros(len(Bset)+len(Nset))
x[Bset]=b@np.linalg.inv(B.T)
print("Iteration: ",iter,"Minimizer: ",x)