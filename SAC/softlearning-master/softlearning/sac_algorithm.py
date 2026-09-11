import tensorflow as tf
import numpy as np
from neural_networks import QNetwork  

class SAC:
    """최소화된 Soft Actor-Critic 구현"""
    
    def __init__(self, 
                 policy, #현재 정책
                 q_network_1, # Q네트워크 1
                 q_network_2, # Q네트워크 2
                 state_dim,
                 action_dim,
                 policy_lr=1e-4, # 학습률
                 q_lr=1e-4, # Q학습률 1
                 alpha_lr=1e-4, # 엔트로피학습률
                 discount=0.99, # 보상의 지수가중뭐시기그거평균
                 tau=5e-3       # 타겟네트워크업데이트속도. 온라인 네트워크 반영비율
                 ):
        
        self.policy = policy
        self.Q1 = q_network_1
        self.Q2 = q_network_2

        # Target networks (천천히 업데이트)
        self.Q1_target = QNetwork(hidden_size=q_network_1.hidden_size)
        self.Q2_target = QNetwork(hidden_size=q_network_2.hidden_size)

        dummy = tf.zeros((1, state_dim + action_dim))

        q_network_1(dummy)
        q_network_2(dummy)
        self.Q1(dummy);        self.Q1_target(dummy)
        self.Q2(dummy);        self.Q2_target(dummy)
        self.Q1_target.set_weights(self.Q1.get_weights())
        self.Q2_target.set_weights(self.Q2.get_weights())

        # 정책도 여기서 미리 build 해 둔다.
        # @tf.function 은 tracing(첫 호출) 이후에 변수가 새로 생기면 에러를 낸다.
        self.policy(tf.zeros((1, state_dim)))
        
        # Optimizers
        self.policy_optimizer = tf.optimizers.Adam(policy_lr)
        self.q1_optimizer = tf.optimizers.Adam(q_lr)
        self.q2_optimizer = tf.optimizers.Adam(q_lr)
        self.alpha_optimizer = tf.optimizers.Adam(alpha_lr)
        
        # Entropy parameter
        self.log_alpha = tf.Variable(0.0, trainable=True) # 역전파로 학습 -> 엔트로피 온도 계수의 log값.
                                                          # alpha가 크면 많이 바뀐다.
        self.target_entropy = -float(action_dim)  # 2D action space
        
        self.discount = discount
        self.tau = tau
    
    @property
    def alpha(self):
        return tf.exp(self.log_alpha) #알파를 절댓값취하는 작용
    
    def update_critic(self, batch):
        """Q-function 업데이트"""
        observations, actions, rewards, next_observations, dones = batch
        
        # Q 타겟 계산
        next_actions, next_log_probs = self.policy(next_observations)
        q1_targets = self.Q1_target(tf.concat([next_observations, next_actions], -1))
        q2_targets = self.Q2_target(tf.concat([next_observations, next_actions], -1))
        # 다음 관측이 A면 B행동을 해라 -> [A,B]
        
        min_q_target = tf.minimum(q1_targets, q2_targets)
        q_target = tf.stop_gradient(
            rewards + (1 - dones) * self.discount * (
                min_q_target - self.alpha * next_log_probs
            )
        )
        # soft bellman 타겟 -> 최종 타겟 결정 ( == 정해 )
        
        # Q1 업데이트
        with tf.GradientTape() as tape:
            q1_values = self.Q1(tf.concat([observations, actions], -1))
            # - 
            q1_loss = tf.reduce_mean((q1_values - q_target) ** 2)
            # Q값이 타겟에 얼마나 가까운지

        q1_grad = tape.gradient(q1_loss, self.Q1.trainable_variables)
        self.q1_optimizer.apply_gradients(
            zip(q1_grad, self.Q1.trainable_variables))
        # 가중치 업데이트

        # Q2 업데이트
        with tf.GradientTape() as tape:
            q2_values = self.Q2(tf.concat([observations, actions], -1))
            q2_loss = tf.reduce_mean((q2_values - q_target) ** 2)
        
        q2_grad = tape.gradient(q2_loss, self.Q2.trainable_variables)
        self.q2_optimizer.apply_gradients(
            zip(q2_grad, self.Q2.trainable_variables))
        
        return q1_loss, q2_loss
    
    def update_actor(self, batch):
        """Policy 업데이트"""
        observations, _, _, _, _ = batch
        
        # 정책 손실 계산 -> 엔트로피 패널티 부여 -> 알파를 곱해서 최종
        with tf.GradientTape() as tape:

            obs_tensor = tf.cast(observations, tf.float32)
            actions, log_probs = self.policy(obs_tensor)  # ← 명시적 변환
            q1_values = self.Q1(tf.concat([obs_tensor, actions], -1))
            q2_values = self.Q2(tf.concat([obs_tensor, actions], -1))
            
            min_q = tf.minimum(q1_values, q2_values)
            policy_loss = tf.reduce_mean(
                self.alpha * log_probs - min_q
            )
        
        # 왜필요한거지 여긴
        # 수정
        policy_grad = tape.gradient(policy_loss, self.policy.trainable_variables)
        policy_grad = [g if g is not None else tf.zeros_like(v)
                    for g, v in zip(policy_grad, self.policy.trainable_variables)]
        self.policy_optimizer.apply_gradients(
            zip(policy_grad, self.policy.trainable_variables))

        # log_probs 를 같이 돌려준다.
        # update_alpha 가 정책을 다시 샘플링하지 않고 이걸 그대로 재사용한다.
        return policy_loss, log_probs

    def update_alpha(self, log_probs):
        """Entropy 계수 자동 조정

        log_probs : update_actor 가 방금 계산한 값을 그대로 받는다.
        어차피 stop_gradient 로 상수 취급하므로 정책을 다시 샘플링할 이유가 없다.
        (정책 forward pass 1회 절약 + actor/alpha 두 손실이 같은 샘플을 공유)
        """
        with tf.GradientTape() as tape:
            alpha_loss = -tf.reduce_mean(
                self.log_alpha * tf.stop_gradient(log_probs + self.target_entropy)
            )
        
        alpha_grad = tape.gradient(alpha_loss, [self.log_alpha])
        self.alpha_optimizer.apply_gradients(zip(alpha_grad, [self.log_alpha]))
        
        return alpha_loss
    
    def train_step(self, batch):
        """한 배치로 critic -> actor -> alpha -> target 을 한 번에 업데이트.

        train.py 에서 부르는 진입점. 아래 _train_step 이 @tf.function 이라
        Python 을 거치지 않고 그래프 하나로 실행된다.
        numpy 배열을 그대로 넘기면 TF 버전에 따라 매 호출 re-tracing 이
        일어날 수 있어서, 여기서 텐서로 바꿔 넘긴다.
        """
        return self._train_step(
            *(tf.convert_to_tensor(x, dtype=tf.float32) for x in batch))

    @tf.function
    def _train_step(self, observations, actions, rewards, next_observations, dones):
        # 주의: 이 안의 Python 코드는 첫 호출(tracing) 때 딱 한 번만 실행된다.
        #       print() 대신 tf.print() 를 써야 하고, .numpy() 는 쓸 수 없다.
        batch = (observations, actions, rewards, next_observations, dones)

        q1_loss, q2_loss = self.update_critic(batch)
        policy_loss, log_probs = self.update_actor(batch)
        alpha_loss = self.update_alpha(log_probs)
        self.update_target_networks()

        return q1_loss, q2_loss, policy_loss, alpha_loss

    def update_target_networks(self):
        """타겟 네트워크를 천천히 업데이트"""
        # 온라인 네트워크 기반

        for target_var, var in zip(
            self.Q1_target.trainable_variables, 
            self.Q1.trainable_variables
        ):
            target_var.assign(self.tau * var + (1 - self.tau) * target_var)
        
        for target_var, var in zip(
            self.Q2_target.trainable_variables, 
            self.Q2.trainable_variables
        ):
            target_var.assign(self.tau * var + (1 - self.tau) * target_var)