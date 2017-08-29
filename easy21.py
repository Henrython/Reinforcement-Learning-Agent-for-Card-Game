# Easy 21

# The game is played with an infinite deck of cards (i.e. cards are sampled
# with replacement)
# Each draw from the deck results in a value between 1 and 10 (uniformly
# distributed) with a colour of red (probability 1/3) or black (probability
# 2/3).
# There are no aces or picture (face) cards in this game
# At the start of the game both the player and the dealer draw one black
# card (fully observed)
# Each turn the player may either stick or hit
# If the player hits then she draws another card from the deck
# If the player sticks she receives no further cards
# The values of the player's cards are added (black cards) or subtracted (red
# cards)
# If the player's sum exceeds 21, or becomes less than 1, then she \goes
# bust" and loses the game (reward -1)
# If the player sticks then the dealer starts taking turns. The dealer always
# sticks on any sum of 17 or greater, and hits otherwise. If the dealer goes
# bust, then the player wins; otherwise, the outcome { win (reward +1),
# lose (reward -1), or draw (reward 0) { is the player with the largest sum.
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
black_prob=2/3
N0=100

def color(x):
	if x==0:return -1
	else: return x

def action_int(x):
	if x == 'hit': return 0
	elif x == 'stick': return 1
	elif x == 1 : return 'stick'
	elif x == 0 : return 'hit'

def true_alpha(n):
	if n != 0:
		return 1/n
	else:
		return n

class Agent_greedy(object):
	def __init__(self):
		Q_Nsa={}
		buffer={}
		eligibility={}
		for i in range(1,11):
			for j in range(1,22):
				for action in range(0,2):
					Q_Nsa[(i,j,action)]=np.zeros((1,2)).squeeze()# (q_val,times)
					buffer[(i,j,action)]=0
					eligibility[(i,j,action)]=0
		self.Q_N=Q_Nsa
		self.buffer=buffer
		self.score=0
		self.elig=eligibility
	
	def greedy(self,state,epsilon):
		bound=np.random.rand()
		if bound<epsilon or self.Ns(state)==0:
			action=np.random.binomial(1,0.5)
		else:
			if self.Q_N[(state)+tuple([1])][0]>=self.Q_N[(state)+tuple([0])][0]:
				action=1
			else:
				action=0
		return action
	
	def Ns(self,state):
		'''how many times has the agent been in state'''
		return np.sum([self.Q_N[state+tuple([0])][1],self.Q_N[state+tuple([1])][1]])
	
	def MC_Learn(self,state_action,alpha,error):
		'''update the q value towards the error'''
		self.Q_N[state_action][0]+=alpha*(error)
	
	def TD_lambda(self,state_action,alpha,error):
		'''TD learning update'''
		self.Q_N[state_action][0]+=alpha*error*self.elig[state_action]
	
	def reset(self):
		self.score=0
		for i in self.buffer:
			self.buffer[i]=0
		for i in self.elig:
			self.elig[i]=0
			
class Eviroment(object):
	def __init__(self):
		self.dealer_score=0
	
	def step(self,agent,action):
		end=False
		if action == 'hit':
			new_number=np.random.randint(1,11)
			new_color=color(np.random.binomial(1,black_prob))
			new_score=new_number*new_color
			agent.score+=new_score
			if agent.score>21 or agent.score<1: 
				reward,end=-1,True
			else:
				reward=0
			return agent.score,reward,end
		if action == 'stick':
			while self.dealer_score<17 and not end:
				new_number=np.random.randint(1,11)
				new_color=color(np.random.binomial(1,black_prob))
				new_score=new_number*new_color
				self.dealer_score += new_score
				if self.dealer_score<1 or self.dealer_score>21:
					end=True
					reward=1
			end=True
			if self.dealer_score>agent.score:
				reward=-1
			elif self.dealer_score==agent.score:
				reward=0
			else:
				reward=1
			return agent.score,reward,end
			
	def reset(self):
		self.dealer_score=0
		
	def MonteCarlo(self,iter_time):
		agent=Agent_greedy()
		for episode in range(iter_time):
			self.reset()
			agent.reset()
			self.dealer_score=self.dealer_show=np.random.randint(1,11)
			agent.score=np.random.randint(1,11)
			done=False
			path=[]
			while not done:
				state=(self.dealer_show,agent.score)
				epsilon=N0/(N0+agent.Ns(state))
				action=agent.greedy(state,epsilon)
				state_action=state+tuple([action])
				path.append(state_action)
				agent.Q_N[state_action][1]+=1
				_,reward,end=self.step(agent,action_int(action))
				if end:
					done=True
				agent.buffer[state_action]+=reward					
			for state_action in path:
				error=agent.buffer[state_action]-agent.Q_N[state_action][0]
				alpha=1/(agent.Q_N[state_action][1])
				agent.MC_Learn(state_action,alpha,error)
		return agent		
		
	def TD(self,lambd,iter_time):
		'''TD(lambda) control'''
		agent=Agent_greedy()
		for episode in range(iter_time):
			self.reset()
			agent.reset()
			self.dealer_score=self.dealer_show=np.random.randint(1,11)
			agent.score=np.random.randint(1,11)
			done=False
			state=(self.dealer_show,agent.score)
			epsilon=N0/(N0+agent.Ns(state))
			action=agent.greedy(state,epsilon)
			while not done:
				state_action=state+tuple([action])
				agent.Q_N[state_action][1]+=1
				_,reward,end=self.step(agent,action_int(action))
				if end:
					
					done=True
					error=reward-agent.Q_N[state_action][0]
				else:
					next_state=(self.dealer_show,agent.score)
					epsilon=N0/(N0+agent.Ns(next_state))
					next_action=agent.greedy(next_state,epsilon)
					next_state_action=next_state+tuple([next_action])
					error=reward+agent.Q_N[next_state_action][0]-agent.Q_N[state_action][0]
					state=next_state
					action=next_action
				agent.elig[state_action]+=1
				for s in agent.Q_N:
					alpha=true_alpha(agent.Q_N[s][1])
					agent.TD_lambda(s,alpha,error)
					agent.elig[s]*=lambd
		return agent		
		
	 
if __name__ == '__main__': 
	game=Eviroment()
	agent_mc=game.MonteCarlo(1000000)
	agent_td=game.TD(1,1000000)
	dealer=np.array([i for i in range(1,11)])
	player=np.array([i for i in range(1,22)])
	v_star_mc=[]
	v_star_td=[]
	x_axi=[]
	y_axi=[]
	for x in dealer:
		for y in player:
			val_mc=max([agent_mc.Q_N[(x,y,1)][0],agent_mc.Q_N[(x,y,0)][0]])
			val_td=max([agent_td.Q_N[(x,y,1)][0],agent_td.Q_N[(x,y,0)][0]])
			v_star_mc.append(val_mc)
			v_star_td.append(val_td)
			x_axi.append(x)
			y_axi.append(y)
	fig1 = plt.figure(1)
	fig2 = plt.figure(2)
	ax1 = fig1.add_subplot(111, projection='3d')
	ax1.set_xlabel('dealer')
	ax1.set_ylabel('agent score')
	ax1.set_zlabel=('return')
	ax1.set_title('MonteCarlo control')
	ax2 = fig2.add_subplot(111, projection='3d')
	ax2.set_xlabel('dealer')
	ax2.set_ylabel('agent score')
	ax2.set_zlabel=('return')
	ax2.set_title('TD control')
	ax1.scatter(x_axi,y_axi,v_star_mc)
	ax2.scatter(x_axi,y_axi,v_star_td)
	plt.show()