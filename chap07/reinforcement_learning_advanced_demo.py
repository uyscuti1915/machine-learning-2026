#!/usr/bin/env python3
"""
强化学习高级演示
基于第7章强化学习应用内容
包含：RL基础、订单执行优化、投资组合优化等
"""

import qlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import namedtuple, deque
import random
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import DatasetH
import warnings
warnings.filterwarnings('ignore')

# 初始化Qlib
print("初始化Qlib...")
qlib.init(mount_path="~/.qlib/qlib_data/cn_data", region="cn")

# RL环境基础类
class TradingEnvironment(ABC):
    """交易环境基类"""
    
    def __init__(self):
        self.state = None
        self.done = False
        self.step_count = 0
        
    @abstractmethod
    def reset(self):
        """重置环境"""
        pass
    
    @abstractmethod
    def step(self, action):
        """执行动作"""
        pass
    
    @abstractmethod
    def get_state(self):
        """获取当前状态"""
        pass
    
    @abstractmethod
    def get_reward(self, action):
        """计算奖励"""
        pass

class PortfolioOptimizationEnv(TradingEnvironment):
    """投资组合优化环境"""
    
    def __init__(self, data, initial_capital=100000, max_steps=252):
        super().__init__()
        self.data = data
        self.initial_capital = initial_capital
        self.max_steps = max_steps
        self.current_step = 0
        self.portfolio_value = initial_capital
        self.positions = np.zeros(len(data.columns))
        self.cash = initial_capital
        self.price_history = []
        
    def reset(self):
        """重置环境"""
        self.current_step = 0
        self.portfolio_value = self.initial_capital
        self.positions = np.zeros(len(self.data.columns))
        self.cash = self.initial_capital
        self.price_history = []
        self.done = False
        return self.get_state()
    
    def step(self, action):
        """执行动作"""
        if self.done:
            return self.get_state(), 0, self.done, {}
        
        # 动作是权重分配向量
        weights = self.softmax(action)
        
        # 获取当前价格
        if self.current_step < len(self.data):
            current_prices = self.data.iloc[self.current_step].values
            self.price_history.append(current_prices)
            
            # 计算新的持仓
            total_value = self.portfolio_value
            new_positions = (weights * total_value) / current_prices
            
            # 计算交易成本（简化）
            transaction_cost = np.sum(np.abs(new_positions - self.positions)) * current_prices * 0.001
            
            # 更新持仓
            self.positions = new_positions
            self.cash = total_value - np.sum(self.positions * current_prices) - transaction_cost
            
            # 计算奖励
            if self.current_step > 0:
                prev_prices = self.price_history[-2] if len(self.price_history) > 1 else current_prices
                returns = (current_prices - prev_prices) / prev_prices
                portfolio_return = np.sum(weights * returns)
                reward = portfolio_return - 0.001 * np.sum(np.abs(weights))  # 加入正则化
            else:
                reward = 0
            
            # 更新组合价值
            self.portfolio_value = self.cash + np.sum(self.positions * current_prices)
            
            self.current_step += 1
            self.done = self.current_step >= min(self.max_steps, len(self.data))
            
            try:
                if hasattr(self.portfolio_value, 'shape') and self.portfolio_value.shape == ():
                    portfolio_val_scalar = self.portfolio_value.item()
                elif hasattr(self.portfolio_value, 'size') and self.portfolio_value.size == 1:
                    portfolio_val_scalar = self.portfolio_value.item()
                else:
                    portfolio_val_scalar = float(np.sum(self.portfolio_value))
            except:
                portfolio_val_scalar = self.initial_capital
            return self.get_state(), reward, self.done, {'portfolio_value': portfolio_val_scalar}
        
        else:
            self.done = True
            return self.get_state(), 0, self.done, {}
    
    def get_state(self):
        """获取当前状态"""
        if self.current_step < len(self.data):
            # 状态包含：当前价格、历史收益率、当前权重
            current_prices = self.data.iloc[self.current_step].values
            
            if len(self.price_history) > 0:
                prev_prices = self.price_history[-1]
                returns = (current_prices - prev_prices) / prev_prices
            else:
                returns = np.zeros_like(current_prices)
            
            try:
                portfolio_val = float(self.portfolio_value)
                if portfolio_val > 0:
                    current_weights = (self.positions * current_prices) / portfolio_val
                else:
                    current_weights = np.zeros_like(self.positions)
            except (TypeError, ValueError):
                current_weights = np.zeros_like(self.positions)
            
            state = np.concatenate([
                current_prices / np.max(current_prices),  # 归一化价格
                returns,
                current_weights
            ])
            
            return state
        else:
            return np.zeros(len(self.data.columns) * 3)
    
    def get_reward(self, action):
        """计算奖励"""
        if self.current_step == 0:
            return 0
        
        weights = self.softmax(action)
        if len(self.price_history) >= 2:
            current_prices = self.price_history[-1]
            prev_prices = self.price_history[-2]
            returns = (current_prices - prev_prices) / prev_prices
            portfolio_return = np.sum(weights * returns)
            return portfolio_return - 0.001 * np.sum(np.abs(weights))
        else:
            return 0
    
    def softmax(self, x):
        """Softmax函数"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

class OrderExecutionEnv(TradingEnvironment):
    """订单执行环境"""
    
    def __init__(self, target_quantity, price_data, max_steps=20):
        super().__init__()
        self.target_quantity = target_quantity
        self.price_data = price_data
        self.max_steps = max_steps
        self.current_step = 0
        self.executed_quantity = 0
        self.vwap = 0
        self.market_impact = 0.001  # 市场冲击系数
        
    def reset(self):
        """重置环境"""
        self.current_step = 0
        self.executed_quantity = 0
        self.vwap = 0
        self.done = False
        return self.get_state()
    
    def step(self, action):
        """执行动作"""
        if self.done:
            return self.get_state(), 0, self.done, {}
        
        # 动作是执行比例（0-1之间）
        execution_ratio = max(0, min(1, action))
        remaining_quantity = self.target_quantity - self.executed_quantity
        execute_now = remaining_quantity * execution_ratio
        
        if self.current_step < len(self.price_data):
            current_price = self.price_data[self.current_step]
            
            # 计算市场冲击
            impact = self.market_impact * (execute_now / self.target_quantity)
            execution_price = current_price * (1 + impact)
            
            # 更新已执行数量和VWAP
            if self.executed_quantity + execute_now > 0:
                self.vwap = (self.vwap * self.executed_quantity + execution_price * execute_now) / (self.executed_quantity + execute_now)
            
            self.executed_quantity += execute_now
            
            # 计算奖励（负的执行成本和市场冲击）
            benchmark_price = np.mean(self.price_data)  # 基准价格
            cost = abs(execution_price - benchmark_price) * execute_now
            reward = -cost / self.target_quantity  # 归一化
            
            self.current_step += 1
            self.done = (self.current_step >= self.max_steps) or (self.executed_quantity >= self.target_quantity * 0.99)
            
            return self.get_state(), reward, self.done, {
                'executed_quantity': self.executed_quantity,
                'vwap': self.vwap,
                'current_price': current_price
            }
        
        else:
            self.done = True
            return self.get_state(), 0, self.done, {}
    
    def get_state(self):
        """获取当前状态"""
        if self.current_step < len(self.price_data):
            remaining_ratio = (self.target_quantity - self.executed_quantity) / self.target_quantity
            time_ratio = self.current_step / self.max_steps
            current_price = self.price_data[self.current_step]
            avg_price = np.mean(self.price_data[:self.current_step+1])
            
            state = np.array([
                remaining_ratio,
                time_ratio,
                current_price / avg_price,
                self.vwap / current_price if current_price > 0 else 1.0
            ])
            
            return state
        else:
            return np.zeros(4)
    
    def get_reward(self, action):
        """计算奖励"""
        if self.current_step >= len(self.price_data):
            return 0
        
        execution_ratio = max(0, min(1, action))
        remaining_quantity = self.target_quantity - self.executed_quantity
        execute_now = remaining_quantity * execution_ratio
        
        if execute_now > 0:
            current_price = self.price_data[self.current_step]
            impact = self.market_impact * (execute_now / self.target_quantity)
            execution_price = current_price * (1 + impact)
            benchmark_price = np.mean(self.price_data)
            cost = abs(execution_price - benchmark_price) * execute_now
            return -cost / self.target_quantity
        else:
            return 0

# 深度Q网络
class DQN(nn.Module):
    """深度Q网络"""
    
    def __init__(self, state_size, action_size, hidden_size=64):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 策略网络（用于Actor-Critic）
class PolicyNetwork(nn.Module):
    """策略网络"""
    
    def __init__(self, state_size, action_size, hidden_size=64):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=-1)
        return x

# 价值网络
class ValueNetwork(nn.Module):
    """价值网络"""
    
    def __init__(self, state_size, hidden_size=64):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# DQN智能体
class DQNAgent:
    """DQN智能体"""
    
    def __init__(self, state_size, action_size, lr=0.001, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q网络
        self.q_network = DQN(state_size, action_size)
        self.target_q_network = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # 经验回放
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        
        # 更新目标网络
        self.update_target_network()
    
    def update_target_network(self):
        """更新目标网络"""
        self.target_q_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """存储经验"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """选择动作"""
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        
        with torch.no_grad():
            q_values = self.q_network(torch.FloatTensor(state).unsqueeze(0))
            return q_values.argmax().item()
    
    def replay(self):
        """经验回放"""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_q_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (0.99 * next_q_values * ~dones)
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Actor-Critic智能体
class ActorCriticAgent:
    """Actor-Critic智能体"""
    
    def __init__(self, state_size, action_size, lr=0.001):
        self.state_size = state_size
        self.action_size = action_size
        
        # Actor和Critic网络
        self.policy_network = PolicyNetwork(state_size, action_size)
        self.value_network = ValueNetwork(state_size)
        
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=lr)
        
        # 存储轨迹
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
    
    def act(self, state):
        """选择动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.policy_network(state_tensor)
        action_dist = Categorical(action_probs)
        action = action_dist.sample()
        return action.item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """存储转换"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
    
    def learn(self):
        """学习"""
        if len(self.states) == 0:
            return
        
        states = torch.FloatTensor(self.states)
        actions = torch.LongTensor(self.actions)
        rewards = torch.FloatTensor(self.rewards)
        next_states = torch.FloatTensor(self.next_states)
        dones = torch.BoolTensor(self.dones)
        
        # 计算价值和优势
        values = self.value_network(states).squeeze()
        next_values = self.value_network(next_states).squeeze()
        targets = rewards + 0.99 * next_values * ~dones
        advantages = targets - values
        
        # 更新Critic
        value_loss = nn.MSELoss()(values, targets.detach())
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        
        # 更新Actor
        action_probs = self.policy_network(states)
        action_dist = Categorical(action_probs)
        log_probs = action_dist.log_prob(actions)
        policy_loss = -(log_probs * advantages.detach()).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # 清空存储
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

def train_dqn_portfolio_optimization():
    """训练DQN进行投资组合优化"""
    print("\n=== DQN投资组合优化训练 ===")
    
    # 创建模拟数据
    np.random.seed(42)
    n_assets = 5
    n_days = 100
    
    # 生成相关的价格序列
    returns = np.random.multivariate_normal(
        mean=[0.001] * n_assets,
        cov=np.random.rand(n_assets, n_assets) * 0.0001,
        size=n_days
    )
    
    prices = pd.DataFrame(100 * np.exp(np.cumsum(returns, axis=0)))
    
    # 创建环境
    env = PortfolioOptimizationEnv(prices, initial_capital=100000, max_steps=50)
    
    # 创建智能体
    state_size = len(env.get_state())
    action_size = 10  # 离散化的动作空间
    agent = DQNAgent(state_size, action_size)
    
    # 训练参数
    episodes = 100
    scores = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        
        while not env.done:
            # 将连续动作空间离散化
            action_idx = agent.act(state)
            
            # 将离散动作转换为连续权重
            action = np.random.random(n_assets)
            action = action / np.sum(action)  # 归一化为权重
            
            next_state, reward, done, info = env.step(action)
            
            agent.remember(state, action_idx, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            if len(agent.memory) > agent.batch_size:
                agent.replay()
        
        scores.append(total_reward)
        
        if episode % 20 == 0:
            agent.update_target_network()
        
        if episode % 10 == 0:
            avg_score = np.mean(scores[-10:])
            print(f"Episode {episode}, Average Score: {avg_score:.4f}, Epsilon: {agent.epsilon:.3f}")
    
    return agent, scores

def train_ac_order_execution():
    """训练Actor-Critic进行订单执行优化"""
    print("\n=== Actor-Critic订单执行优化训练 ===")
    
    # 创建模拟价格数据
    np.random.seed(42)
    price_trend = np.linspace(100, 105, 50)  # 上涨趋势
    price_noise = np.random.normal(0, 0.5, 50)
    prices = price_trend + price_noise
    
    # 创建环境
    target_quantity = 1000
    env = OrderExecutionEnv(target_quantity, prices, max_steps=20)
    
    # 创建智能体
    state_size = len(env.get_state())
    action_size = 11  # 0到1之间，0.1间隔
    agent = ActorCriticAgent(state_size, action_size)
    
    # 训练参数
    episodes = 200
    scores = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        
        while not env.done:
            action_idx = agent.act(state)
            action = action_idx / (action_size - 1)  # 转换为0-1之间的连续值
            
            next_state, reward, done, info = env.step(action)
            
            agent.store_transition(state, action_idx, reward, next_state, done)
            
            state = next_state
            total_reward += reward
        
        # 每个episode结束后学习
        agent.learn()
        scores.append(total_reward)
        
        if episode % 20 == 0:
            avg_score = np.mean(scores[-20:])
            print(f"Episode {episode}, Average Reward: {avg_score:.4f}")
    
    return agent, scores

def test_trained_agents():
    """测试训练好的智能体"""
    print("\n=== 测试训练好的智能体 ===")
    
    # 测试投资组合优化
    print("测试投资组合优化智能体...")
    
    # 创建新的测试数据
    n_assets = 5
    n_days = 30
    test_returns = np.random.multivariate_normal(
        mean=[0.0005] * n_assets,
        cov=np.random.rand(n_assets, n_assets) * 0.0001,
        size=n_days
    )
    test_prices = pd.DataFrame(100 * np.exp(np.cumsum(test_returns, axis=0)))
    
    # 创建测试环境
    test_env = PortfolioOptimizationEnv(test_prices, initial_capital=100000, max_steps=25)
    
    state = test_env.reset()
    portfolio_values = [test_env.portfolio_value]
    
    while not test_env.done:
        # 简单的随机策略作为基线
        action = np.random.random(n_assets)
        action = action / np.sum(action)
        
        next_state, reward, done, info = test_env.step(action)
        portfolio_values.append(info['portfolio_value'])
        state = next_state
    
    print(f"初始资金: {test_env.initial_capital}")
    print(f"最终组合价值: {portfolio_values[-1]:.2f}")
    print(f"总收益率: {(portfolio_values[-1] - test_env.initial_capital) / test_env.initial_capital:.4f}")
    
    # 测试订单执行
    print("\n测试订单执行智能体...")
    
    test_prices = np.linspace(100, 102, 30) + np.random.normal(0, 0.3, 30)
    target_quantity = 500
    test_exec_env = OrderExecutionEnv(target_quantity, test_prices, max_steps=15)
    
    state = test_exec_env.reset()
    execution_record = []
    
    while not test_exec_env.done:
        # 简单的均匀执行策略
        remaining_steps = test_exec_env.max_steps - test_exec_env.current_step
        action = 1.0 / remaining_steps if remaining_steps > 0 else 1.0
        
        next_state, reward, done, info = test_exec_env.step(action)
        execution_record.append({
            'step': test_exec_env.current_step,
            'executed': info['executed_quantity'],
            'vwap': info['vwap'],
            'current_price': info['current_price']
        })
        state = next_state
    
    print(f"目标执行数量: {target_quantity}")
    print(f"实际执行数量: {execution_record[-1]['executed']:.0f}")
    print(f"执行完成率: {execution_record[-1]['executed']/target_quantity:.2%}")
    print(f"VWAP: {execution_record[-1]['vwap']:.2f}")
    print(f"基准价格（均价）: {np.mean(test_prices):.2f}")

def visualize_rl_results():
    """可视化强化学习结果"""
    print("\n=== 可视化强化学习结果 ===")
    
    try:
        # 训练DQN
        dqn_agent, dqn_scores = train_dqn_portfolio_optimization()
        
        # 训练Actor-Critic
        ac_agent, ac_scores = train_ac_order_execution()
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # DQN学习曲线
        axes[0, 0].plot(dqn_scores)
        axes[0, 0].set_title('DQN Portfolio Optimization - Learning Curve')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].grid(True)
        
        # DQN滑动平均
        window_size = 10
        if len(dqn_scores) >= window_size:
            moving_avg = pd.Series(dqn_scores).rolling(window=window_size).mean()
            axes[0, 1].plot(moving_avg)
            axes[0, 1].set_title(f'DQN - {window_size}-Episode Moving Average')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Average Reward')
            axes[0, 1].grid(True)
        
        # Actor-Critic学习曲线
        axes[1, 0].plot(ac_scores)
        axes[1, 0].set_title('Actor-Critic Order Execution - Learning Curve')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Total Reward')
        axes[1, 0].grid(True)
        
        # Actor-Critic滑动平均
        if len(ac_scores) >= window_size:
            ac_moving_avg = pd.Series(ac_scores).rolling(window=window_size).mean()
            axes[1, 1].plot(ac_moving_avg)
            axes[1, 1].set_title(f'Actor-Critic - {window_size}-Episode Moving Average')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Average Reward')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # 保存图表
        plt.savefig('demo/chap07/rl_training_results.png', dpi=300, bbox_inches='tight')
        print("强化学习训练结果图表已保存到 demo/chap07/rl_training_results.png")
        plt.close(fig)
        
    except Exception as e:
        print(f"可视化过程出现错误: {e}")

def main():
    print("=== 强化学习高级演示 ===")
    
    print("\n=== RL环境测试 ===")
    
    # 测试投资组合优化环境
    print("测试投资组合优化环境...")
    n_assets = 3
    sample_data = pd.DataFrame(np.random.randn(10, n_assets) + 100)
    
    portfolio_env = PortfolioOptimizationEnv(sample_data, initial_capital=10000, max_steps=5)
    state = portfolio_env.reset()
    print(f"初始状态维度: {len(state)}")
    
    for step in range(3):
        action = np.random.random(n_assets)
        action = action / np.sum(action)  # 归一化
        next_state, reward, done, info = portfolio_env.step(action)
        print(f"步骤 {step+1}: 奖励={reward:.4f}, 组合价值={info['portfolio_value']:.2f}")
        if done:
            break
    
    # 测试订单执行环境
    print("\n测试订单执行环境...")
    sample_prices = [100, 101, 102, 103, 104]
    order_env = OrderExecutionEnv(target_quantity=100, price_data=sample_prices, max_steps=3)
    
    state = order_env.reset()
    print(f"初始状态: {state}")
    
    for step in range(3):
        action = 0.3  # 每次执行30%
        next_state, reward, done, info = order_env.step(action)
        print(f"步骤 {step+1}: 执行量={info['executed_quantity']:.1f}, VWAP={info['vwap']:.2f}")
        if done:
            break
    
    print("\n=== 神经网络架构测试 ===")
    
    # 测试DQN网络
    print("测试DQN网络...")
    dqn = DQN(state_size=10, action_size=5)
    sample_state = torch.randn(1, 10)
    q_values = dqn(sample_state)
    print(f"DQN输出形状: {q_values.shape}")
    print(f"Q值: {q_values.detach().numpy().flatten()}")
    
    # 测试策略网络
    print("\n测试策略网络...")
    policy_net = PolicyNetwork(state_size=8, action_size=4)
    sample_state = torch.randn(1, 8)
    action_probs = policy_net(sample_state)
    print(f"策略网络输出形状: {action_probs.shape}")
    print(f"动作概率: {action_probs.detach().numpy().flatten()}")
    
    # 测试价值网络
    print("\n测试价值网络...")
    value_net = ValueNetwork(state_size=8)
    state_value = value_net(sample_state)
    print(f"状态价值: {state_value.item():.4f}")
    
    # 运行完整的训练示例
    print("\n=== 运行强化学习训练 ===")
    
    try:
        # 小规模训练演示
        print("运行小规模DQN训练演示...")
        
        # 创建简单环境
        simple_data = pd.DataFrame(np.random.randn(20, 3) * 0.01 + 1).cumprod() * 100
        simple_env = PortfolioOptimizationEnv(simple_data, initial_capital=10000, max_steps=10)
        
        state_size = len(simple_env.get_state())
        action_size = 5
        simple_agent = DQNAgent(state_size, action_size)
        
        episode_rewards = []
        
        for episode in range(20):  # 短训练演示
            state = simple_env.reset()
            total_reward = 0
            
            while not simple_env.done:
                action_idx = simple_agent.act(state)
                action = np.random.random(3)
                action = action / np.sum(action)
                
                next_state, reward, done, info = simple_env.step(action)
                simple_agent.remember(state, action_idx, reward, next_state, done)
                
                state = next_state
                total_reward += reward
                
                if len(simple_agent.memory) > 10:
                    simple_agent.replay()
            
            episode_rewards.append(total_reward)
            
            if episode % 5 == 0:
                print(f"Episode {episode}: 总奖励={total_reward:.4f}, Epsilon={simple_agent.epsilon:.3f}")
        
        print(f"训练完成，平均奖励: {np.mean(episode_rewards):.4f}")
        
    except Exception as e:
        print(f"训练过程出现错误: {e}")
    
    # 测试智能体
    test_trained_agents()
    
    # 可视化结果
    visualize_rl_results()
    
    print("\n强化学习高级演示完成！")
    print("\n主要演示内容:")
    print("1. 投资组合优化环境和DQN智能体")
    print("2. 订单执行优化环境和Actor-Critic智能体") 
    print("3. 神经网络架构（DQN, Policy Network, Value Network）")
    print("4. 经验回放和目标网络更新")
    print("5. 强化学习训练过程和结果可视化")

if __name__ == "__main__":
    main()