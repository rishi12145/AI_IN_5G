import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def setup_style():
    sns.set_theme(style="darkgrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['font.family'] = 'sans-serif'

def plot_advanced_metrics(rewards, power_history, throughput_history):
    setup_style()
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 18))
    
    sns.lineplot(ax=axes[0], data=rewards, color='#1f77b4', linewidth=2)
    axes[0].set_title('DQN Cumulative Rewards convergence', fontsize=16, pad=15)
    axes[0].set_xlabel('Episodes')
    axes[0].set_ylabel('Total Reward')
    
    window = 20
    rolling_mean = np.convolve(rewards, np.ones(window)/window, mode='valid')
    sns.lineplot(ax=axes[0], x=range(window-1, len(rewards)), y=rolling_mean, color='#d62728', linewidth=2.5, linestyle='--')
    
    sns.lineplot(ax=axes[1], data=power_history, color='#2ca02c', linewidth=2)
    axes[1].set_title('Network Power Efficiency (Total Watts)', fontsize=16, pad=15)
    axes[1].set_xlabel('Episodes')
    axes[1].set_ylabel('Power Consumption')
    
    sns.lineplot(ax=axes[2], data=throughput_history, color='#9467bd', linewidth=2)
    axes[2].set_title('Network Throughput Optimization', fontsize=16, pad=15)
    axes[2].set_xlabel('Episodes')
    axes[2].set_ylabel('Aggregated Throughput (Mbps)')
    
    plt.tight_layout()
    plt.savefig('advanced_marl_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
