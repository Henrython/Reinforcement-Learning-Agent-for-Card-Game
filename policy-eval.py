# given a random policy, evaluate the state value function

import numpy as np
np.set_printoptions(precision=10)
#np.random.seed(1234)
# 20 state random walk, first and last state termination with reward 0 and 10
# policy: 0.5 prob transfer from one to adjcent state


def initial(size):
	discount=0.8
	v=np.zeros((size,1))
	p=np.zeros((size,size))
	p[0,0]=1
	p[-1,-1]=1
	for i in range(1,size-1):
		p[i,i-1],p[i,i+1]=0.5,0.5
	r=np.zeros((size,1))
	r[size-2,0]=0.5
	count=np.zeros((size,1))
	mdp={'v':v,'p':p,'r':r,'count':count,'discount':discount}
	return mdp

def value_iter(size,epsilon):
	mdp=initial(size)
	v=mdp['v']
	p=mdp['p']
	r=mdp['r']
	gamma=mdp['discount']
	converge=False
	while not converge:
		v_old=np.array(v)
		v=r+gamma*np.dot(p,v_old)
		mse=np.sum((v-v_old)**2)
		if mse<epsilon:
			converge=True
	return v

def walk():
	if np.random.binomial(1,0.5)==1:return 1
	else: return -1	
	
def monte_carlo(size,iter_times,method='first_visit',alpha=0.01):
	'''first visit mc simulation'''
	mdp=initial(size)
	v=mdp['v']
	gamma=mdp['discount']
	count=mdp['count']
	def replace(visited,gt):
		for i in range(visited.shape[0]):
			if visited[i]==0:
				gt[i]=v[i]
		return gt	
	
	def one_over_count(count):
		out=np.zeros(count.shape)
		for i in range(len(count)):
			if count[i]>0:
				out[i]=1/count[i]
		return out	
	
	def sample(start):
		visited=np.zeros((size))
		discount=np.zeros((size,1))
		discount[start]=1
		count[start]+=1
		visited[start]=1
		g=np.zeros((size,1))
		terminate=False
		while not terminate:
			reward=0
			start=start+walk()
			#print(start)
			if discount[start]==0:
				count[start]+=1
				visited[start]=1
			if start==size-1:
				reward=1
				terminate=True
			if start==0:
				terminate=True
			g=g+discount*reward
			discount=discount*gamma
			if discount[start]==0:
				discount[start]=1
		return g,visited
	
	def first_visit():

		nonlocal v
		for i in range(iter_times):
			start=np.random.random_integers(1,size-2)
			gt=sample(start)[0]	
			v=v+gt
		return v*one_over_count(count)
	
	def incremental():
		
		nonlocal v
		for i in range(iter_times):
			start=np.random.random_integers(1,size-2)
			gt,visited=sample(start)
			gt=replace(visited,gt)
			#print(gt)
			increment=one_over_count(count)
			#print(count)
			v=v+((gt-v)*increment)
		return v
	
	def incremental_alpha(alpha):
		nonlocal v
		for i in range(iter_times):
			start=np.random.random_integers(1,size-2)
			gt,visited=sample(start)
			gt=replace(visited,gt)
			v=v+(gt-v)*alpha
		return v
	if method=='first_visit':return first_visit()
	if method=='incremental':return incremental()
	if method=='incremental_alpha':return incremental_alpha(alpha)
		

def td_learn(size,iter_times,alpha):
	mdp=initial(size)
	v=mdp['v']
	gamma=mdp['discount']
	dim=v.shape[0]-1
	print(dim)
	for j in range(iter_times):
		for start in range(1,dim):
			reward=0
			action=walk()
			if start+action==dim:
				reward=1
			v[start]+=alpha*(reward+gamma*v[start+action]-v[start])
	return v		

if __name__=='__main__':
	print(value_iter(10,0.0000000000001))
	print(monte_carlo(10,10000))
	print(monte_carlo(10,10000,'incremental'))
	print(monte_carlo(10,10000,'incremental_alpha'))
	#print(td_learn(5,10000000,0.0001))X
		

