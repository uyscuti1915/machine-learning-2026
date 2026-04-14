#!/usr/bin/env python3
"""
回测系统演示
基于第6章回测系统内容
"""

import qlib
import numpy as np
import pandas as pd
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import DatasetH
import warnings
warnings.filterwarnings('ignore')

# 初始化Qlib
print("初始化Qlib...")
qlib.init(mount_path="~/.qlib/qlib_data/cn_data", region="cn")

class SimpleBacktester:
    """简化的回测系统"""
    
    def __init__(self, initial_capital=100000, commission=0.001):
        self.initial_capital = initial_capital
        self.commission = commission
        self.positions = {}
        self.cash = initial_capital
        self.portfolio_value = initial_capital
        self.trade_history = []
    
    def execute_trades(self, signals, prices):
        """执行交易"""
        for stock, signal in signals.items():
            if stock in prices:
                price = prices[stock]
                if signal > 0:  # 买入信号
                    shares = int((self.cash * signal) / price)
                    if shares > 0:
                        cost = shares * price * (1 + self.commission)
                        if cost <= self.cash:
                            self.positions[stock] = self.positions.get(stock, 0) + shares
                            self.cash -= cost
                            self.trade_history.append({
                                'stock': stock,
                                'action': 'buy',
                                'shares': shares,
                                'price': price,
                                'cost': cost
                            })
                elif signal < 0 and stock in self.positions:  # 卖出信号
                    shares = self.positions[stock]
                    revenue = shares * price * (1 - self.commission)
                    self.cash += revenue
                    del self.positions[stock]
                    self.trade_history.append({
                        'stock': stock,
                        'action': 'sell',
                        'shares': shares,
                        'price': price,
                        'revenue': revenue
                    })
    
    def calculate_portfolio_value(self, prices):
        """计算投资组合价值"""
        value = self.cash
        for stock, shares in self.positions.items():
            if stock in prices:
                value += shares * prices[stock]
        self.portfolio_value = value
        return value

def main():
    print("准备数据...")
    
    # 准备数据处理器
    handler = Alpha158(
        instruments='csi300',
        start_time='2020-01-01',
        end_time='2020-12-31',
        freq='day'
    )
    
    # 创建数据集
    dataset = DatasetH(
        handler=handler,
        segments={
            'train': ('2020-01-01', '2020-08-31'),
            'test': ('2020-09-01', '2020-12-31')
        }
    )
    
    print("训练模型...")
    model = LGBModel(
        loss='mse',
        n_estimators=50,
        max_depth=6,
        learning_rate=0.1,
        verbose=-1
    )
    
    model.fit(dataset)
    
    print("生成预测...")
    predictions = model.predict(dataset, segment='test')
    
    print("\n=== 回测演示 ===")
    
    # 创建回测器
    backtester = SimpleBacktester(initial_capital=100000, commission=0.001)
    
    # 模拟回测过程
    print("执行模拟回测...")
    
    # 获取测试数据
    test_data = dataset.prepare('test')
    
    # 简化：使用前10个交易日进行演示
    sample_dates = predictions.index.get_level_values('datetime').unique()[:10]
    
    portfolio_values = []
    
    for date in sample_dates:
        # 获取当日预测
        daily_pred = predictions.loc[predictions.index.get_level_values('datetime') == date]
        
        if len(daily_pred) == 0:
            continue
        
        # 生成交易信号（简化）
        signals = {}
        top_stocks = daily_pred.nlargest(5)  # 选择前5只股票
        
        for stock_date, pred_value in top_stocks.items():
            stock = stock_date[0] if isinstance(stock_date, tuple) else stock_date
            signals[stock] = 0.2  # 每只股票分配20%资金
        
        # 模拟价格（实际应该从数据中获取）
        prices = {stock: 10 + np.random.normal(0, 1) for stock in signals.keys()}
        
        # 执行交易
        backtester.execute_trades(signals, prices)
        
        # 计算组合价值
        portfolio_value = backtester.calculate_portfolio_value(prices)
        portfolio_values.append(portfolio_value)
        
        print(f"日期: {date.strftime('%Y-%m-%d')}, 组合价值: {portfolio_value:.2f}")
    
    print(f"\n=== 回测结果 ===")
    print(f"初始资金: {backtester.initial_capital}")
    print(f"最终组合价值: {portfolio_values[-1] if portfolio_values else backtester.initial_capital:.2f}")
    
    if len(portfolio_values) > 1:
        total_return = (portfolio_values[-1] - backtester.initial_capital) / backtester.initial_capital
        print(f"总收益率: {total_return:.4f}")
    
    print(f"交易次数: {len(backtester.trade_history)}")
    
    if backtester.trade_history:
        print("\n前5笔交易记录:")
        for trade in backtester.trade_history[:5]:
            print(f"  {trade}")
    
    print("\n回测系统演示完成！")

if __name__ == "__main__":
    main()