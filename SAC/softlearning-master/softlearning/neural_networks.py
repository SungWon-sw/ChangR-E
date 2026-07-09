import tensorflow as tf
import numpy as np

class GaussianPolicy(tf.keras.Model):
    """연속 제어용 정책 네트워크"""
    
    def __init__(self, state_dim, action_dim, hidden_size=256, **kwargs):
        # tensor 제공 모델 tf.keras.Model 이용
        # state_dim : 환경에서 수용하는 차원, action_dim : 에이전트가 출력하는 차원
        super().__init__(**kwargs) #keras model 초기화
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size

        self.fc1 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.fc2 = tf.keras.layers.Dense(hidden_size, activation='relu')
        # 2개 층 쌓기 - state_dim을 추상적인 시그널벡터로 압축

        self.mu = tf.keras.layers.Dense(action_dim)
        # 1개 층 쌓기 - 정규분포 평균
        self.log_std = tf.keras.layers.Dense(action_dim)
        # 1개 층 쌓기 - 정규분포 log( 표준편차 )

    # 순전파
    def call(self, states):
        x = self.fc1(states)
        x = self.fc2(x)
        # states -> fc1 -> fc2 추출

        mu = self.mu(x)
        log_std = tf.clip_by_value(self.log_std(x), -20, 0)
        # 학습 안정화 전략
        std = tf.exp(log_std)

        epsilon = tf.random.normal(tf.shape(mu))
        u = mu + std * epsilon  # 평균 + 표준편차 * 노이즈
                                # epsilon을 난수로 설정해서 역전파가 가능한 랜덤을 만듦

        # log_prob 계산 - 마할라노비스 거리 계산 후 정규화
        log_probs = -0.5 * ((u - mu) ** 2 / (std ** 2 + 1e-6) +
                            2 * log_std + tf.math.log(2.0 * np.pi))
        log_probs = tf.reduce_sum(log_probs, axis=-1, keepdims=True)

        # Squash
        actions = tf.tanh(u)

        # Jacobian 보정: log(1 - tanh(u)^2) = log(1 - a^2)
        # Squash에서 바뀐 확률 밀도를 보정.
        log_probs -= tf.reduce_sum(
            tf.math.log(1 - actions ** 2 + 1e-6), axis=-1, keepdims=True)

        return actions, log_probs
        #log_probs는 0에 가까울수록(==크면) 선택 확률 높음. target_entrophy에 가까우면 좋음

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
        # Q값 . . . 기대 보상
    
    def call(self, state_action):
        x = self.fc1(state_action)
        x = self.fc2(x)
        q_value = self.out(x)
        return q_value
        # double-Q : 2번 call된다.

    # clone_model이 정상 작동하기 위해 설정을 저장하고 복원하는 메서드 추가
    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_size": self.hidden_size,
        })
        return config