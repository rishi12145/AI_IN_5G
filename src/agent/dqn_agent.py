import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from src.agent.models import QNetwork
from src.agent.replay_buffer import ReplayBuffer

class DQNAgent:
    def __init__(self, config):
        self.state_dim = config["agent"]["state_dim"]
        self.action_dim = config["agent"]["action_dim"]
        self.gamma = config["agent"]["gamma"]
        self.epsilon = config["agent"]["epsilon_start"]
        self.epsilon_end = config["agent"]["epsilon_end"]
        self.epsilon_decay = config["agent"]["epsilon_decay"]
        self.batch_size = config["agent"]["batch_size"]
        self.target_update = config["agent"]["target_update"]
        
        self.policy_net = QNetwork(self.state_dim, self.action_dim)
        self.target_net = QNetwork(self.state_dim, self.action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config["agent"]["lr"])
        self.memory = ReplayBuffer(10000)
        self.steps_done = 0

    def select_action(self, state):
        if random.random() > self.epsilon:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.policy_net(state_tensor)
                return q_values.max(1)[1].item()
        else:
            return random.randrange(self.action_dim)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
            
        states, actions, rewards, next_states = self.memory.sample(self.batch_size)
        
        state_batch = torch.FloatTensor(states)
        action_batch = torch.LongTensor(actions).unsqueeze(1)
        reward_batch = torch.FloatTensor(rewards)
        next_state_batch = torch.FloatTensor(next_states)
        
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        with torch.no_grad():
            next_state_values = self.target_net(next_state_batch).max(1)[0]
            
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        
        loss = nn.MSELoss()(state_action_values, expected_state_action_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
