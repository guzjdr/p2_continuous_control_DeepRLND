from common import *
from buffer import ReplayBuffer
from modelQ import DeepQNetwork, DuelingQNetwork


#Module Variables
BUFFER_SIZE = int(1e5) # memory replay buffer size
BATCH_SIZE = 64 # batch size
GAMMA = 0.99 # Q learning discount size
TAU = 1e-3 # for soft update from local network to taget network
LR_ACTOR = 5e-4 # learning rate
LR_CRITIC = 5e-4 # learning rate  
UPDATE_CRITIC_EVERY = 1 # number of frames used to update the local network
UPDATE_ACTOR_TARGET = 2 * UPDATE_CRITIC_EVERY

class Agent():
    """ Agent
    
    Params
    ******
    
    """
    def __init__(self, state_size, action_size, seed, duelingQNet=False):
        """ """
        #Environment Deifiniton Variables
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        
        #Deep-Q Networks
        self.qnet_local = DeepQNetwork(state_size, action_size, seed).to(device) if not duelingQNet else DuelingQNetwork(state_size, action_size, seed).to(device)
        self.qnet_target = DeepQNetwork(state_size, action_size, seed).to(device) if not duelingQNet else DuelingQNetwork(state_size, action_size, seed).to(device) 
        self.optimizer = optim.Adam(self.qnet_local.parameters(), lr=LR)
        
        #Buffer - Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        #Time step variable
        self.t_step = 0
        
    def step(self, state, action, reward, next_state, done):
        """ """
        #Update memory
        #pdb.set_trace()
        self.memory.add(state, action, reward, next_state, done)
        
        #Learn every number of time steps - UPDATE_EVERY
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            #
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)
    def act(self, state, eps=0.):
        """ """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnet_local.eval()
        with torch.no_grad():
            action_values = self.qnet_local(state)
        self.qnet_local.train()

         # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
        
    def learn(self, experiences, gamma):
        """ """
        states, actions, rewards, next_states, dones = experiences
        
        Q_targets_next = self.qnet_target(next_states).detach().max(1)[0].unsqueeze(1)
        #Update target net q values
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        
        #Expected Q values from local network
        Q_expected = self.qnet_local(states).gather(1, actions.long())
        
        #Compute the loss
        loss = F.mse_loss(Q_expected, Q_targets)
        #Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        #Slow annealing of target network w.r.t local network
        self.soft_update(self.qnet_local, self.qnet_target, TAU)
    
    def soft_update(self, local_model, target_model, tau):
        """ """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-TAU)*target_param.data)
        
        
