import numpy as np

class ReplayBuffer:
    def __init__(self, max_size=100000, state_dim=4, action_dim=2):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        
        self.states = np.zeros((max_size, state_dim))
        self.actions = np.zeros((max_size, action_dim))
        self.rewards = np.zeros((max_size, 1))
        self.next_states = np.zeros((max_size, state_dim))
        self.dones = np.zeros((max_size, 1))
    
    def add(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, batch_size)
        return (
            self.states[indices].astype('float32'),
            self.actions[indices].astype('float32'),
            self.rewards[indices].astype('float32'),
            self.next_states[indices].astype('float32'),
            self.dones[indices].astype('float32')
        )