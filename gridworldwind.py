# gridworld wind with king move
import numpy as np

number_move=8

def state_space(size,start,end,wind=[]):
	'''define a gridworld state space with size*size. wind defines the coordinetes of the winds
	and the mean and prob of varing from the mean. wind=[(coordinete,mean,prob)]'''
	state={}
	wind_dic={}
	for i in wind:
		wind_dic[i[0]]=np.array((i[1],i[2]))
	for i in range(size):
		if i in wind_dic:
			state[i]=wind_dic[i]
		else:
			state[i]=np.zeros((1,2)).squeeze()
	state['start']=start
	state['end']=end
	return state

def initialize(size):
	action={}
	q={}
	e={}
	for i in range(size):
		for j in range(size):
			#action[(i,j)]=None#{'up':1/8,'down':1/8,'left':1/8,'right':1/8,'upright':1/8,'upleft':1/8,'downright':1/8,'downleft':1/8}
			q[(i,j)]=np.zeros((1,number_move)).squeeze()#every state has 8 actions, init the q to be 0.
			e[(i,j)]=np.zeros((1,number_move)).squeeze()#every state has 8 actions, init the eligibility to 0.
	return q,e

def bound(x,size):
	if x>=size-1:
		return size-1
	else: return max(0,x)

def greedy_improve(q,epsilon=0.1):
	'''greedy improve pi to be epsilon_greedy'''
	key=np.random.rand()#choose a number between 0,1 uniformly
	uniformly=True
	for i in q:
		if i!=0:uniformly=False
	if key<epsilon or uniformly :
		action=np.random.randint(0,number_move)
	else:
		action=q.argmax()
	return action
	
def sarsa(size,namda,start,end,iter_time,alpha=0.1,gamma=0.8):
	walk=[(0,1),(0,-1),(1,0),(-1,0),(1,1),(-1,1),(1,-1),(-1,-1)]
	state=state_space(size,start,end,wind=[(4,1,0),(5,1,0),(6,1,0)])
	q=initialize(size)[0]
	
	for i in range(iter_time):
		e=initialize(size)[1]
		start=state['start']
		action=greedy_improve(q[start])#sample a action for the epsilon_greedy
		terminate=False
		while not terminate:
			go=walk[action]
			force=state[start[0]][0]
			if force !=0:
				go=(go[0],force)
			next_state=(bound(start[0]+go[0],size),bound(start[1]+go[1],size))
			if next_state==state['end']:
				reward=10
				terminate=True
			else:
				reward=-1
			next_action=greedy_improve(q[next_state])
			td_erro=reward+gamma*q[next_state][next_action]-q[start][action]
			e[start][action]+=1
			for s in q:
				q[s]=q[s]+alpha*td_erro*e[s]
				e[s]=gamma*namda*e[s]
			start=next_state
			action=next_action
	s=state['start']
	while s != state['end']:
		act=walk[q[s].argmax()]
		print('state:',s,'action:',act)
		if state[s[0]][0] !=0:
			act=(act[0],state[s[0]][0])
		s=(bound(s[0]+act[0],size),bound(s[1]+act[1],size))
	return q	
			
if __name__=='__main__':
	size=10
	q_val=sarsa(10,0.01,(0,4),(6,5),100000)
