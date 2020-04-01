#!/usr/bin/env python
# coding: utf-8

# # Continuous Control
# 
# ---
# 
# You are welcome to use this coding environment to train your agent for the project.  Follow the instructions below to get started!
# 
# ### 1. Start the Environment
# 
# Run the next code cell to install a few packages.  This line will take a few minutes to run!

# In[1]:


get_ipython().system('pip -q install ./python')


# The environments corresponding to both versions of the environment are already saved in the Workspace and can be accessed at the file paths provided below.  
# 
# Please select one of the two options below for loading the environment.

# In[2]:


from unityagents import UnityEnvironment
import numpy as np

# select this option to load version 1 (with a single agent) of the environment
env = UnityEnvironment(file_name='/data/Reacher_One_Linux_NoVis/Reacher_One_Linux_NoVis.x86_64')

# select this option to load version 2 (with 20 agents) of the environment
# env = UnityEnvironment(file_name='/data/Reacher_Linux_NoVis/Reacher.x86_64')


# Environments contain **_brains_** which are responsible for deciding the actions of their associated agents. Here we check for the first brain available, and set it as the default brain we will be controlling from Python.

# In[3]:


# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]


# ### 2. Examine the State and Action Spaces
# 
# Run the code cell below to print some information about the environment.

# In[4]:


# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])


# ### 3. Take Random Actions in the Environment
# 
# In the next code cell, you will learn how to use the Python API to control the agent and receive feedback from the environment.
# 
# Note that **in this coding environment, you will not be able to watch the agents while they are training**, and you should set `train_mode=True` to restart the environment.

# In[5]:


env_info = env.reset(train_mode=True)[brain_name]      # reset the environment    
states = env_info.vector_observations                  # get the current state (for each agent)
scores = np.zeros(num_agents)                          # initialize the score (for each agent)
while True:
    actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
    actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
    env_info = env.step(actions)[brain_name]           # send all actions to tne environment
    next_states = env_info.vector_observations         # get next state (for each agent)
    rewards = env_info.rewards                         # get reward (for each agent)
    dones = env_info.local_done                        # see if episode finished
    scores += env_info.rewards                         # update the score (for each agent)
    states = next_states                               # roll over states to next time step
    if np.any(dones):                                  # exit loop if episode finished
        break
print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))


# When finished, you can close the environment.

# In[6]:


#env.close()


# ### 4. It's Your Turn!
# 
# Now it's your turn to train your own agent to solve the environment!  A few **important notes**:
# - When training the environment, set `train_mode=True`, so that the line for resetting the environment looks like the following:
# ```python
# env_info = env.reset(train_mode=True)[brain_name]
# ```
# - To structure your work, you're welcome to work directly in this Jupyter notebook, or you might like to start over with a new file!  You can see the list of files in the workspace by clicking on **_Jupyter_** in the top left corner of the notebook.
# - In this coding environment, you will not be able to watch the agents while they are training.  However, **_after training the agents_**, you can download the saved model weights to watch the agents on your own machine! 

# In[7]:


import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=400, fc2_units=300, bn_mode=0):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
            bn_mode (int): Use Batch Normalization - 0=disabled, 1=BN before Activation, 2=BN after Activation (3, 4 are alt. versions of 1, 2)
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        # Dense layers
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        
        # Normalization layers
        self.bn1 = nn.BatchNorm1d(fc1_units)
        if bn_mode!=2:
            self.bn2 = nn.BatchNorm1d(fc2_units)     
        if bn_mode==3:    
            self.bn3 = nn.BatchNorm1d(action_size)   
        self.bn_mode=bn_mode
        
        self.reset_parameters()
        
        #print("[INFO] Actor initialized with parameters : state_size={} action_size={} seed={} fc1_units={} fc2_units={} bn_mode={}".format(state_size, action_size, seed, fc1_units, fc2_units, bn_mode))
        
        

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
    
        # Reshape the state to comply with Batch Normalization
        if state.dim() == 1:
            state = torch.unsqueeze(state,0)
        
        if self.bn_mode==0:
            # Batch Normalization disabled
            x = F.relu(self.fc1(state))
            x = F.relu(self.fc2(x))
            return F.tanh(self.fc3(x))
        elif self.bn_mode==1:
            # Batch Normalization before Activation
            x = self.fc1(state)
            x = self.bn1(x)   
            x = F.relu(x)
            x = self.fc2(x)
            x = self.bn2(x)   
            x = F.relu(x)
            x = self.fc3(x)
            return F.tanh(x)
        elif self.bn_mode==2:
            # Batch Normalization after Activation  
            x = F.relu(self.fc1(state))
            x = self.bn1(x) 
            x = F.relu(self.fc2(x))
            return F.tanh(self.fc3(x))
        elif self.bn_mode==3:
            # Batch Normalization before Activation (alternate version)
            x = self.fc1(state)
            x = self.bn1(x)   
            x = F.relu(x)
            x = self.fc2(x)
            x = self.bn2(x)   
            x = F.relu(x)
            x = self.fc3(x)
            x = self.bn3(x)   
            return F.tanh(x)
        elif self.bn_mode==4:
            # Batch Normalization after Activation  (alternate version)
            x = F.relu(self.fc1(state))
            x = self.bn1(x) 
            x = F.relu(self.fc2(x))
            x = self.bn2(x)   
            return F.tanh(self.fc3(x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fcs1_units=400, fc2_units=300, bn_mode=0):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
            bn_mode (int): Use Batch Norm. - 0=disabled, 1=BN before Activation, 2=BN after Activation (3, 4 are alt. versions of 1, 2)
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        # Dense layers
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        
        # Normalization layers
        self.bn1 = nn.BatchNorm1d(fcs1_units)
        if bn_mode>2:
            self.bn2 = nn.BatchNorm1d(fc2_units)
        self.bn_mode=bn_mode

        self.reset_parameters()
        
        #print("[INFO] CRITIC initialized with parameters : state_size={} action_size={} seed={} fcs1_units={} fc2_units={} bn_mode={}".format(state_size, action_size, seed, fcs1_units, fc2_units, bn_mode))
        

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        
        # Reshape the state to comply with Batch Normalization
        if state.dim() == 1:
            state = torch.unsqueeze(state,0)
                  
        if self.bn_mode==0:
            # Batch Normalization disabled
            xs = F.relu(self.fcs1(state))
            x = torch.cat((xs, action), dim=1)
            x = F.relu(self.fc2(x))
            return self.fc3(x)
        elif self.bn_mode==1:
            # Batch Normalization before Activation
            xs = self.fcs1(state)
            xs = self.bn1(xs)   
            xs = F.relu(xs)
            x = torch.cat((xs, action), dim=1)
            x = self.fc2(x)
            x = F.relu(x)
            return self.fc3(x)
        elif self.bn_mode==2:
            # Batch Normalization after Activation  
            xs = F.relu(self.fcs1(state))
            xs = self.bn1(xs) 
            x = torch.cat((xs, action), dim=1)
            x = F.relu(self.fc2(x))
            return self.fc3(x)
        elif self.bn_mode==3:
            # Batch Normalization before Activation (alternate version)
            xs = self.fcs1(state)
            xs = self.bn1(xs)   
            xs = F.relu(xs)
            x = torch.cat((xs, action), dim=1)
            x = self.fc2(x)
            x = self.bn2(x) 
            x = F.relu(x)
            return self.fc3(x)
        elif self.bn_mode==4:
            # Batch Normalization after Activation (alternate version) 
            xs = F.relu(self.fcs1(state))
            xs = self.bn1(xs) 
            x = torch.cat((xs, action), dim=1)
            x = F.relu(self.fc2(x))
            x = self.bn2(x)   
            return self.fc3(x)
            
    


# In[8]:


import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    def __init__(self, state_size, action_size, random_seed=10,
                actor_fc1_units=400, actor_fc2_units=300,
                critic_fcs1_units=400, critic_fc2_units=300,
                buffer_size=int(1e5), batch_size=128,
                gamma=0.99, tau=1e-3 , bn_mode=0,
                lr_actor=1e-4, lr_critic=1e-4, weight_decay=0,
                add_ounoise=True, mu=0., theta=0.15, sigma=0.2):
        
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
            actor_fc1_units (int): number of units for the layer 1 in the actor model
            actor_fc2_units (int): number of units for the layer 2 in the actor model
            critic_fcs1_units (int): number of units for the layer 1 in the critic model
            critic_fc2_units (int): number of units for the layer 2 in the critic model
            buffer_size (int) : replay buffer size
            batch_size (int) : minibatch size
            gamma (int) : discount factor
            tau (int) : for soft update of target parameter
            bn_mode (int): Use Batch Norm - 0=disabled, 1=BN before Activation, 2=BN after Activation (3, 4 are alt. versions of 1, 2)
            lr_actor (int) : learning rate of the actor 
            lr_critic (int) : learning rate of the critic 
            weight_decay (int) : L2 weight decay  
            add_ounoise (bool) : Add Ornstein-Uhlenbeck noise
            mu (float) : Ornstein-Uhlenbeck noise parameter
            theta (float) : Ornstein-Uhlenbeck noise parameter
            sigma (float) : Ornstein-Uhlenbeck noise parameter            
        """
        
        print("\n[INFO]ddpg constructor called with parameters: state_size={} action_size={} random_seed={} actor_fc1_units={} actor_fc2_units={} critic_fcs1_units={} critic_fc2_units={} buffer_size={} batch_size={} gamma={} tau={} bn_mode={} lr_actor={} lr_critic={} weight_decay={} add_ounoise={} mu={} theta={} sigma={} \n".format(state_size, action_size, random_seed, 
                                                          actor_fc1_units, actor_fc2_units, critic_fcs1_units, critic_fc2_units,
                                                          buffer_size, batch_size, gamma, tau, 
                                                          bn_mode, lr_actor, lr_critic, weight_decay, 
                                                          add_ounoise, mu, theta,sigma))
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.buffer_size=buffer_size
        self.batch_size=batch_size
        self.gamma=gamma
        self.tau=tau
        self.lr_actor=lr_actor
        self.lr_critic=lr_critic
        self.weight_decay=weight_decay
        self.add_ounoise=add_ounoise

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed, actor_fc1_units, actor_fc2_units, bn_mode=bn_mode).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed, actor_fc1_units, actor_fc2_units, bn_mode=bn_mode).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor)
        
        # Make sure the Target Network has the same weight values as the Local Network
        for target, local in zip(self.actor_target.parameters(), self.actor_local.parameters()):
            target.data.copy_(local.data)
    
        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed, critic_fcs1_units, critic_fc2_units, bn_mode=bn_mode).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed, critic_fcs1_units, critic_fc2_units, bn_mode=bn_mode).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=lr_critic, weight_decay=self.weight_decay)
        
        # Make sure the Target Network has the same weight values as the Local Network
        for target, local in zip(self.critic_target.parameters(), self.critic_local.parameters()):
            target.data.copy_(local.data)

        # Noise process
        self.noise = OUNoise(size=action_size, seed=random_seed, 
                             mu=mu, theta=theta, sigma=sigma)

        # Replay memory
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, random_seed)
    
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences, self.gamma)

    def act(self, state):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if self.add_ounoise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss (using gradient clipping)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


# In[9]:


# Reset the environment    
env_info = env.reset(train_mode=True)[brain_name]     

# number of agents
num_agents = len(env_info.agents)

# size of each action
ENV_ACTION_SIZE = brain.vector_action_space_size

# size of the state space 
states = env_info.vector_observations
ENV_STATE_SIZE = states.shape[1]


# In[10]:


# Agent default hyperparameters
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
ACTOR_FC1_UNITS = 400   # Number of units for the layer 1 in the actor model
ACTOR_FC2_UNITS = 300   # Number of units for the layer 2 in the actor model
CRITIC_FCS1_UNITS = 400 # Number of units for the layer 1 in the critic model
CRITIC_FC2_UNITS = 300  # Number of units for the layer 2 in the critic model
BN_MODE = 0             # Use Batch Norm. - 0=disabled, 1=BN before Activation, 2=BN after Activation (3, 4 are alt. versions of 1, 2)
ADD_OU_NOISE = True     # Add Ornstein-Uhlenbeck noise
MU = 0.                 # Ornstein-Uhlenbeck noise parameter
THETA = 0.15            # Ornstein-Uhlenbeck noise parameter
SIGMA = 0.2             # Ornstein-Uhlenbeck noise parameter


def ddpg(n_episodes=5000, max_t=500,
         state_size=ENV_STATE_SIZE, action_size=ENV_ACTION_SIZE, random_seed=10, 
         actor_fc1_units=ACTOR_FC1_UNITS, actor_fc2_units=ACTOR_FC2_UNITS,
         critic_fcs1_units=CRITIC_FCS1_UNITS, critic_fc2_units=CRITIC_FC2_UNITS,
         buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, bn_mode=BN_MODE,
         gamma=GAMMA, tau=TAU, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, weight_decay=WEIGHT_DECAY,
         add_ounoise=ADD_OU_NOISE, mu=MU, theta=THETA, sigma=SIGMA):  
    
    # Instantiate the Agent
    agent = Agent(state_size=state_size,action_size=action_size, random_seed=random_seed,
                  actor_fc1_units=actor_fc1_units, actor_fc2_units=actor_fc2_units,
                  critic_fcs1_units=critic_fcs1_units, critic_fc2_units=critic_fc2_units,
                  buffer_size=buffer_size, batch_size=batch_size, bn_mode=bn_mode,
                  gamma=gamma, tau=tau,
                  lr_actor=lr_actor, lr_critic=lr_critic, weight_decay=weight_decay,
                  add_ounoise=add_ounoise, mu=mu, theta=theta, sigma=sigma)
                  
    scores_deque = deque(maxlen=100)
    scores = []

    print("\nStart training:")
    for i_episode in range(1, n_episodes+1):
        
        # Reset the env and get the state (Single Agent)
        env_info = env.reset(train_mode=True)[brain_name]     
        state = env_info.vector_observations[0]
        
        # Reset the DDPG Agent (Reset the internal state (= noise) to mean mu)
        agent.reset()
        
        # Reset the score 
        score = 0
        
        for t in range(max_t):
            action = agent.act(state)                   # select an action 

            env_info=env.step(action)[brain_name]        # send action to the environment
            next_state = env_info.vector_observations[0] # get next state (Single Agent)
            reward = env_info.rewards[0]                 # get reward (Single Agent)
            done = env_info.local_done[0]                # see if episode finished (Single Agent)
            
            #if i_episode<2:
            #    print("Debug: steps={} reward={} done={}".format(t,reward,done))
            
            # Save experience in replay memory, and use random sample from buffer to learn
            agent.step(state, action, reward, next_state, done)
            
            state = next_state
            score += reward
            if done:
                #print("Episode {} has terminated at step {}".format(i_episode, t))
                break 
        
        # Save scores and compute average score over last 100 episodes
        scores_deque.append(score)
        scores.append(score)
        avg_score = np.mean(scores_deque)
        
        print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(i_episode, avg_score, score), end="")
        if i_episode % 100 == 0:
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            # Early stop
            if avg_score > 30:
                print('\rEnvironment solved in {} episodes with an Average Score of {:.2f}'.format(i_episode, avg_score))
                return scores
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, avg_score))  
    return scores


# In[11]:


def plot_training(scores):
    # Plot the Score evolution during the training
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.tick_params(axis='x', colors='deepskyblue')
    ax.tick_params(axis='y', colors='deepskyblue')
    plt.plot(np.arange(1, len(scores)+1), scores, color='deepskyblue')
    plt.ylabel('Score', color='deepskyblue')
    plt.xlabel('Episode #', color='deepskyblue')
    plt.show()


# In[12]:


scores = ddpg(n_episodes=1500, max_t=1000, 
              actor_fc1_units=256, actor_fc2_units=128,
              critic_fcs1_units=256, critic_fc2_units=128, bn_mode=2,
              gamma=0.99, tau=1e-3, lr_actor=2e-4, lr_critic=2e-4, weight_decay=0.,
              add_ounoise=True, mu=0., theta=0.15, sigma=0.1 )

plot_training(scores)


# In[13]:


import matplotlib.pyplot as plt

# Set plotting options
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')
np.set_printoptions(precision=3, linewidth=120)


# In[14]:


plot_training(scores)

