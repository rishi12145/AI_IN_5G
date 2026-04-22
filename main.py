import argparse
import json
import logging
from src.env.network_env import AdvancedNetworkEnv
from src.agent.dqn_agent import DQNAgent
from src.utils.plotter import plot_advanced_metrics

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def run_simulation(config):
    env = AdvancedNetworkEnv(config)
    agents = [DQNAgent(config) for _ in range(config["env"]["num_base_stations"])]
    
    total_rewards_history = []
    total_power_history = []
    total_throughput_history = []
    
    episodes = config["training"]["episodes"]
    max_steps = config["env"]["max_steps"]
    
    logging.info(f"Starting Multi-Agent Deep Q-Network training over {episodes} episodes...")
    
    for episode in range(episodes):
        states = env.reset()
        episode_reward = 0
        episode_power = 0
        episode_throughput = 0
        
        for step in range(max_steps):
            actions = [agents[i].select_action(states[i]) for i in range(env.num_agents)]
            
            next_states, rewards = env.step(actions)
            
            for i in range(env.num_agents):
                agents[i].memory.push(states[i], actions[i], rewards[i], next_states[i])
                agents[i].optimize_model()
                
            states = next_states
            episode_reward += sum(rewards)
            episode_power += sum(env.power_levels)
            episode_throughput += sum(env.throughput)
            
        for agent in agents:
            agent.update_epsilon()
            
        if episode % config["agent"]["target_update"] == 0:
            for agent in agents:
                agent.update_target_network()
                
        total_rewards_history.append(episode_reward)
        total_power_history.append(episode_power)
        total_throughput_history.append(episode_throughput)
        
        if (episode + 1) % 10 == 0:
            logging.info(f"Episode {episode + 1}/{episodes} | Avg Reward: {episode_reward:.2f} | Pwr: {episode_power:.2f} | TPut: {episode_throughput:.2f}")

    logging.info("Training complete. Generating advanced metrics...")
    plot_advanced_metrics(total_rewards_history, total_power_history, total_throughput_history)
    logging.info("Metrics saved to advanced_marl_metrics.png.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Advanced 5G MARL Simulator')
    parser.add_argument('--config', type=str, default='config/settings.json')
    args = parser.parse_args()
    
    setup_logger()
    config_data = load_config(args.config)
    run_simulation(config_data)
