import tensorflow as tf
import numpy as np

class GaussianPolicy(tf.keras.Model):
    """연속 제어용 정책 네트워크"""
    
    def __init__(self, state_dim, action_dim, hidden_size=256, **kwargs):
        # 부모 클래스(tf.keras.Model) 생성자에 kwargs를 넘겨주어 trainable, dtype 등을 처리합니다.
        super().__init__(**kwargs)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size

        self.fc1 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.fc2 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.mu = tf.keras.layers.Dense(action_dim)
        self.log_std = tf.keras.layers.Dense(action_dim)
    
    def call(self, states):
        x = self.fc1(states)
        x = self.fc2(x)
        mu = self.mu(x)
        log_std = tf.clip_by_value(self.log_std(x), -20, 2)
        
        # Gaussian 샘플링
        std = tf.exp(log_std)
        epsilon = tf.random.normal(tf.shape(mu))
        actions = mu + std * epsilon
        
        # Log probability 계산
        log_probs = -0.5 * ((actions - mu) ** 2 / (std ** 2 + 1e-6) + 
                            2 * log_std + tf.math.log(2 * np.pi))
        log_probs = tf.reduce_sum(log_probs, axis=-1, keepdims=True)
        
        # Action squashing (tanh)
        actions = tf.tanh(actions)
        
        # Jacobian 보정 (squashing으로 인한 log_prob 수정)
        log_probs = log_probs - tf.reduce_sum(
            tf.math.log(1 - actions ** 2 + 1e-6), axis=-1, keepdims=True)
        
        return actions, log_probs

    # clone_model이 정상 작동하기 위해 설정을 저장하고 복원하는 메서드 추가
    def get_config(self):
        config = super().get_config()
        config.update({
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_size": self.hidden_size,
        })
        return config


class QNetwork(tf.keras.Model):
    """Q-function 네트워크"""
    
    def __init__(self, hidden_size=256, **kwargs):
        # 부모 클래스(tf.keras.Model) 생성자에 kwargs를 넘겨주어 trainable, dtype 등을 처리합니다.
        super().__init__(**kwargs)
        self.hidden_size = hidden_size

        self.fc1 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.fc2 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.out = tf.keras.layers.Dense(1)
    
    def call(self, state_action):
        x = self.fc1(state_action)
        x = self.fc2(x)
        q_value = self.out(x)
        return q_value

    # clone_model이 정상 작동하기 위해 설정을 저장하고 복원하는 메서드 추가
    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_size": self.hidden_size,
        })
        return config