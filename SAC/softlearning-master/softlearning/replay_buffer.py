import numpy as np

#환경과 상호작용
class ReplayBuffer:
    def __init__(self, max_size=100000, state_dim=4, action_dim=2):
        self.max_size = max_size #최대 저장 경험
        self.ptr = 0 #다음 덮어쓸 위치 (포인터) - 원형 큐?느낌
        self.size = 0 #현재 경험
        
        self.states = np.zeros((max_size, state_dim))
        self.actions = np.zeros((max_size, action_dim))
        self.rewards = np.zeros((max_size, 1))
        self.next_states = np.zeros((max_size, state_dim))
        self.dones = np.zeros((max_size, 1))
    
    #경험 저장 - 최근 10만개, 원형큐
    def add(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    #미니배치 추출
    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, batch_size)
        return (
            self.states[indices].astype('float32'),
            self.actions[indices].astype('float32'),
            self.rewards[indices].astype('float32'),
            self.next_states[indices].astype('float32'),
            self.dones[indices].astype('float32') # done == 1이면 미레 보상 차단.
                                                  # 아니면 Q함수에 가중치를 곱해 미래 보상 예측
        )