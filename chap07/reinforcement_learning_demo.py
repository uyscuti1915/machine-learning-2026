#!/usr/bin/env python3
"""
强化学习应用演示
基于第7章强化学习应用内容
"""

import qlib
import numpy as np
import pandas as pd
from qlib.contrib.data.handler import Alpha158
import warnings
warnings.filterwarnings('ignore')

# 初始化Qlib
print("初始化Qlib...")
qlib.init(mount_path="~/.qlib/qlib_data/cn_data", region="cn")

class TradingEnvironment:
    """简化的交易环境"""
    
    def __init__(self, data, initial_balance=10000):
        self.data = data
        self.initial_balance = initial_balance
        self.reset()
    
    def reset(self):
        """重置环境"""
        self.current_step = 0
        self.balance = self.initial_balance
        self.positions = {}
        self.portfolio_value = self.initial_balance
        return self.get_state()
    
    def get_state(self):
        """获取当前状态"""
        if self.current_step >= len(self.data):
            return np.zeros(10)  # 简化状态
        
        # 简化：返回部分特征作为状态
        current_data = self.data.iloc[self.current_step]
        state = current_data.head(10).fillna(0).values
        return state
    
    def step(self, action):
        """执行动作"""
        if self.current_step >= len(self.data) - 1:
            return self.get_state(), 0, True, {}
        
        # 简化的动作执行
        # action: 0=hold, 1=buy, 2=sell
        reward = np.random.normal(0, 0.01)  # 简化的奖励
        
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        
        return self.get_state(), reward, done, {}

class SimpleQLearningAgent:
    """简化的Q学习智能体"""
    
    def __init__(self, state_size=10, action_size=3, learning_rate=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        # 简化的Q表（实际应使用神经网络）
        self.q_table = np.random.random((100, action_size)) * 0.01
    
    def get_state_index(self, state):
        """将状态转换为索引（简化）"""
        return int(np.sum(state * 10) % 100)
    
    def act(self, state):
        """选择动作"""
        if np.random.random() <= self.epsilon:
            return np.random.choice(self.action_size)
        
        state_idx = self.get_state_index(state)
        return np.argmax(self.q_table[state_idx])
    
    def learn(self, state, action, reward, next_state, done):
        """学习更新Q值"""
        state_idx = self.get_state_index(state)
        next_state_idx = self.get_state_index(next_state)
        
        target = reward
        if not done:
            target += 0.95 * np.max(self.q_table[next_state_idx])
        
        self.q_table[state_idx][action] += self.learning_rate * (
            target - self.q_table[state_idx][action]
        )
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_agent(env, agent, episodes=100):
    """训练智能体"""
    print(f"开始训练智能体，共 {episodes} 个回合...")
    
    rewards = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
        
        rewards.append(total_reward)
        
        if episode % 20 == 0:
            avg_reward = np.mean(rewards[-20:])
            print(f"回合 {episode}, 平均奖励: {avg_reward:.4f}, Epsilon: {agent.epsilon:.4f}")
    
    return rewards

class PortfolioOptimizationAgent:
    """投资组合优化智能体"""
    
    def __init__(self, n_assets=5):
        self.n_assets = n_assets
        self.weights = np.ones(n_assets) / n_assets
    
    def optimize_portfolio(self, returns, risk_aversion=1.0):
        """优化投资组合权重"""
        # 简化的均值回归优化
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        
        # 简化的权重计算
        scores = mean_returns / np.diag(cov_matrix)
        weights = np.maximum(scores, 0)
        weights = weights / weights.sum()
        
        self.weights = weights
        return weights

def main():
    print("准备数据...")
    
    # 准备数据处理器
    handler = Alpha158(
        instruments='csi300',
        start_time='2020-01-01',
        end_time='2020-06-30',
        freq='day'
    )
    
    # 获取数据
    data = handler.fetch()
    
    # 获取特征数据 (移除标签列)
    label_cols = [col for col in data.columns if 'LABEL' in str(col)]
    feature_cols = [col for col in data.columns if 'LABEL' not in str(col)]
    features = data[feature_cols]
    print(f"数据形状: {features.shape}")
    
    print("\n=== 强化学习交易智能体演示 ===")
    
    # 创建交易环境
    sample_data = features.head(100)  # 使用前100行数据进行演示
    env = TradingEnvironment(sample_data)
    
    # 创建Q学习智能体
    agent = SimpleQLearningAgent()
    
    # 训练智能体
    rewards = train_agent(env, agent, episodes=50)
    
    print(f"\n训练完成！")
    print(f"平均奖励: {np.mean(rewards):.4f}")
    print(f"最终Epsilon: {agent.epsilon:.4f}")
    
    print("\n=== 测试训练后的智能体 ===")
    
    # 测试训练后的智能体
    state = env.reset()
    total_reward = 0
    steps = 0
    done = False
    
    action_counts = {0: 0, 1: 0, 2: 0}
    
    while not done and steps < 50:
        action = agent.act(state)
        action_counts[action] += 1
        next_state, reward, done, _ = env.step(action)
        
        state = next_state
        total_reward += reward
        steps += 1
    
    print(f"测试步数: {steps}")
    print(f"总奖励: {total_reward:.4f}")
    print(f"动作分布: Hold={action_counts[0]}, Buy={action_counts[1]}, Sell={action_counts[2]}")
    
    print("\n=== 投资组合优化演示 ===")
    
    # 创建投资组合优化智能体
    portfolio_agent = PortfolioOptimizationAgent(n_assets=5)
    
    # 模拟收益数据
    np.random.seed(42)
    returns = pd.DataFrame(
        np.random.normal(0.001, 0.02, (50, 5)),
        columns=[f'Asset_{i}' for i in range(5)]
    )
    
    print("模拟收益数据统计:")
    print(returns.describe())
    
    # 优化投资组合
    optimal_weights = portfolio_agent.optimize_portfolio(returns)
    
    print(f"\n优化后的投资组合权重:")
    for i, weight in enumerate(optimal_weights):
        print(f"Asset_{i}: {weight:.4f}")
    
    # 计算组合表现
    portfolio_returns = (returns * optimal_weights).sum(axis=1)
    
    print(f"\n投资组合表现:")
    print(f"平均收益: {portfolio_returns.mean():.6f}")
    print(f"收益标准差: {portfolio_returns.std():.6f}")
    print(f"夏普比率: {portfolio_returns.mean() / portfolio_returns.std():.4f}")
    
    print("\n=== 订单执行优化演示 ===")
    
    class OrderExecutionAgent:
        """订单执行优化智能体"""
        
        def __init__(self):
            self.slippage_model = lambda size: size * 0.001  # 简化的滑点模型
        
        def optimize_execution(self, total_size, time_horizon=10):
            """优化订单执行"""
            # 简化的TWAP策略
            chunk_size = total_size / time_horizon
            execution_plan = [chunk_size] * time_horizon
            
            return execution_plan
        
        def execute_order(self, execution_plan):
            """执行订单"""
            total_cost = 0
            for chunk in execution_plan:
                slippage = self.slippage_model(chunk)
                total_cost += slippage
            
            return total_cost
    
    # 测试订单执行优化
    execution_agent = OrderExecutionAgent()
    
    # 优化大额订单执行
    total_order_size = 10000
    execution_plan = execution_agent.optimize_execution(total_order_size, time_horizon=20)
    execution_cost = execution_agent.execute_order(execution_plan)
    
    print(f"订单总大小: {total_order_size}")
    print(f"执行计划: 分 {len(execution_plan)} 次执行，每次 {execution_plan[0]:.0f}")
    print(f"预估执行成本: {execution_cost:.6f}")
    
    print("\n强化学习应用演示完成！")

if __name__ == "__main__":
    main()