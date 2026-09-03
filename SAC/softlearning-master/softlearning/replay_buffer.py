import numpy as np

class ReplayBuffer:
    def __init__(self, max_size=100000, state_dim=4, action_dim=2):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        
        # 신경망이 어차피 float32 로 계산하므로 처음부터 float32 로 저장한다.
        # float64 로 두면 메모리가 2배(94MB -> 47MB)이고, sample() 마다
        # astype 으로 복사본을 한 번 더 만들게 된다.
        self.states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.next_states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)
    
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
        # 저장이 이미 float32 라 astype 복사가 필요 없다 (복사 2회 -> 1회).
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )