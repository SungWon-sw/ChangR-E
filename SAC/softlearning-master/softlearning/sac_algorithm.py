import tensorflow as tf
import numpy as np
from copy import deepcopy

class SAC:
    """최소화된 Soft Actor-Critic 구현"""
    
    def __init__(self, 
                 policy,
                 q_network_1, 
                 q_network_2,
                 policy_lr=3e-4,
                 q_lr=3e-4,
                 alpha_lr=3e-4,
                 discount=0.99,
                 tau=5e-3):
        
        self.policy = policy
        self.Q1 = q_network_1
        self.Q2 = q_network_2
        
        # Target networks (천천히 업데이트)
        self.Q1_target = tf.keras.models.clone_model(q_network_1)
        self.Q2_target = tf.keras.models.clone_model(q_network_2)
        
        # Optimizers
        self.policy_optimizer = tf.optimizers.Adam(policy_lr)
        self.q1_optimizer = tf.optimizers.Adam(q_lr)
        self.q2_optimizer = tf.optimizers.Adam(q_lr)
        self.alpha_optimizer = tf.optimizers.Adam(alpha_lr)
        
        # Entropy parameter
        self.log_alpha = tf.Variable(0.0, trainable=True)
        self.target_entropy = -np.prod([2])  # 2D action space
        
        self.discount = discount
        self.tau = tau
    
    @property
    def alpha(self):
        return tf.exp(self.log_alpha)
    
    def update_critic(self, batch):
        """Q-function 업데이트"""
        observations, actions, rewards, next_observations, dones = batch
        
        # Q 타겟 계산
        next_actions, next_log_probs = self.policy(next_observations)
        q1_targets = self.Q1_target(tf.concat([next_observations, next_actions], -1))
        q2_targets = self.Q2_target(tf.concat([next_observations, next_actions], -1))
        
        min_q_target = tf.minimum(q1_targets, q2_targets)
        q_target = rewards + (1 - dones) * self.discount * (
            min_q_target - self.alpha * next_log_probs
        )
        
        # Q1 업데이트
        with tf.GradientTape() as tape:
            q1_values = self.Q1(tf.concat([observations, actions], -1))
            q1_loss = tf.reduce_mean((q1_values - q_target) ** 2)
        
        q1_grad = tape.gradient(q1_loss, self.Q1.trainable_variables)
        self.q1_optimizer.apply_gradients(
            zip(q1_grad, self.Q1.trainable_variables))
        
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
        
        with tf.GradientTape() as tape:
            actions, log_probs = self.policy(observations)
            q1_values = self.Q1(tf.concat([observations, actions], -1))
            q2_values = self.Q2(tf.concat([observations, actions], -1))
            
            min_q = tf.minimum(q1_values, q2_values)
            policy_loss = tf.reduce_mean(
                self.alpha * log_probs - min_q
            )
        
        policy_grad = tape.gradient(policy_loss, self.policy.trainable_variables)
        self.policy_optimizer.apply_gradients(
            zip(policy_grad, self.policy.trainable_variables))
        
        return policy_loss
    
    def update_alpha(self, batch):
        """Entropy 계수 자동 조정"""
        observations, _, _, _, _ = batch
        
        with tf.GradientTape() as tape:
            _, log_probs = self.policy(observations)
            alpha_loss = -tf.reduce_mean(
                self.log_alpha * tf.stop_gradient(log_probs + self.target_entropy)
            )
        
        alpha_grad = tape.gradient(alpha_loss, [self.log_alpha])
        self.alpha_optimizer.apply_gradients(zip(alpha_grad, [self.log_alpha]))
        
        return alpha_loss
    
    def update_target_networks(self):
        """타겟 네트워크를 천천히 업데이트"""
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