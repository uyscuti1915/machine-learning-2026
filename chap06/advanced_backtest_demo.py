#!/usr/bin/env python3
"""
高级回测系统演示
基于第6章回测系统内容
包含：事件驱动回测、风险控制、回测分析、报告生成等
"""

import qlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import DatasetH
from qlib.data import D
import warnings
warnings.filterwarnings('ignore')

# 初始化Qlib
print("初始化Qlib...")
qlib.init(mount_path="~/.qlib/qlib_data/cn_data", region="cn")

@dataclass
class Event:
    """事件基类"""
    timestamp: datetime
    type: str
    data: Dict[str, Any]

class MarketDataEvent(Event):
    """市场数据事件"""
    def __init__(self, timestamp, symbol, price, volume):
        super().__init__(timestamp, 'MARKET_DATA', {
            'symbol': symbol,
            'price': price,
            'volume': volume
        })

class SignalEvent(Event):
    """信号事件"""
    def __init__(self, timestamp, signals):
        super().__init__(timestamp, 'SIGNAL', {
            'signals': signals
        })

class OrderEvent(Event):
    """订单事件"""
    def __init__(self, timestamp, symbol, order_type, quantity, price):
        super().__init__(timestamp, 'ORDER', {
            'symbol': symbol,
            'order_type': order_type,
            'quantity': quantity,
            'price': price
        })

class FillEvent(Event):
    """成交事件"""
    def __init__(self, timestamp, symbol, quantity, price, commission):
        super().__init__(timestamp, 'FILL', {
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'commission': commission
        })

class RiskController:
    """风险控制器"""
    
    def __init__(self, max_position_size=0.1, max_drawdown=0.2, max_leverage=2.0):
        self.max_position_size = max_position_size
        self.max_drawdown = max_drawdown
        self.max_leverage = max_leverage
        self.portfolio_value_history = []
    
    def check_order_risk(self, order, portfolio):
        """检查订单风险"""
        # 检查持仓集中度
        if not self.check_position_concentration(order, portfolio):
            return False, "持仓集中度过高"
        
        # 检查资金充足性
        if not self.check_sufficient_funds(order, portfolio):
            return False, "资金不足"
        
        return True, "风险检查通过"
    
    def check_position_concentration(self, order, portfolio):
        """检查持仓集中度"""
        current_position = portfolio.get('positions', {}).get(order.data['symbol'], 0)
        new_position = current_position + order.data['quantity']
        
        portfolio_value = portfolio.get('total_value', 100000)
        position_value = abs(new_position * order.data['price'])
        
        concentration = position_value / portfolio_value
        
        return concentration <= self.max_position_size
    
    def check_sufficient_funds(self, order, portfolio):
        """检查资金充足性"""
        required_cash = order.data['quantity'] * order.data['price']
        available_cash = portfolio.get('cash', 0)
        
        return available_cash >= required_cash
    
    def update_portfolio_value(self, portfolio_value):
        """更新组合价值历史"""
        self.portfolio_value_history.append(portfolio_value)
    
    def calculate_drawdown(self):
        """计算回撤"""
        if len(self.portfolio_value_history) < 2:
            return 0
        
        peak = max(self.portfolio_value_history)
        current = self.portfolio_value_history[-1]
        
        drawdown = (peak - current) / peak
        
        return drawdown

class EventDrivenBacktest:
    """事件驱动回测系统"""
    
    def __init__(self, initial_capital=100000):
        self.events = []
        self.current_time = None
        self.positions = {}
        self.cash = initial_capital
        self.initial_capital = initial_capital
        self.portfolio_value = initial_capital
        self.risk_controller = RiskController()
        self.trade_history = []
        self.portfolio_history = []
        self.commission_rate = 0.0005
    
    def add_event(self, event):
        """添加事件"""
        self.events.append(event)
        self.events.sort(key=lambda x: x.timestamp)
    
    def process_events(self):
        """处理事件"""
        for event in self.events:
            self.current_time = event.timestamp
            
            if event.type == 'MARKET_DATA':
                self.handle_market_data(event)
            elif event.type == 'SIGNAL':
                self.handle_signal(event)
            elif event.type == 'ORDER':
                self.handle_order(event)
            elif event.type == 'FILL':
                self.handle_fill(event)
    
    def handle_market_data(self, event):
        """处理市场数据事件"""
        # 更新组合价值
        self.update_portfolio_value(event.data)
    
    def handle_signal(self, event):
        """处理信号事件"""
        signals = event.data['signals']
        
        # 生成订单
        for symbol, signal in signals.items():
            if signal > 0:  # 买入信号
                order = OrderEvent(
                    event.timestamp,
                    symbol,
                    'BUY',
                    100,  # 简化：固定数量
                    10.0 + np.random.normal(0, 1)  # 模拟价格
                )
                self.handle_order(order)
            elif signal < 0:  # 卖出信号
                if symbol in self.positions:
                    order = OrderEvent(
                        event.timestamp,
                        symbol,
                        'SELL',
                        self.positions[symbol],
                        10.0 + np.random.normal(0, 1)  # 模拟价格
                    )
                    self.handle_order(order)
    
    def handle_order(self, event):
        """处理订单事件"""
        # 检查风险
        portfolio_info = {
            'positions': self.positions,
            'cash': self.cash,
            'total_value': self.portfolio_value
        }
        
        risk_ok, risk_msg = self.risk_controller.check_order_risk(event, portfolio_info)
        
        if risk_ok:
            # 执行订单
            fill = FillEvent(
                event.timestamp,
                event.data['symbol'],
                event.data['quantity'],
                event.data['price'],
                self.calculate_commission(event.data)
            )
            self.handle_fill(fill)
        else:
            print(f"订单被风控拒绝: {risk_msg}")
    
    def handle_fill(self, event):
        """处理成交事件"""
        symbol = event.data['symbol']
        quantity = event.data['quantity']
        price = event.data['price']
        commission = event.data['commission']
        
        # 更新持仓
        if symbol not in self.positions:
            self.positions[symbol] = 0
        
        self.positions[symbol] += quantity
        
        # 更新现金
        self.cash -= quantity * price + commission
        
        # 记录交易
        self.trade_history.append({
            'timestamp': event.timestamp,
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'commission': commission,
            'action': 'buy' if quantity > 0 else 'sell'
        })
        
        # 如果持仓为0，删除记录
        if self.positions[symbol] == 0:
            del self.positions[symbol]
    
    def calculate_commission(self, order_data):
        """计算手续费"""
        return order_data['quantity'] * order_data['price'] * self.commission_rate
    
    def update_portfolio_value(self, market_data):
        """更新组合价值"""
        value = self.cash
        for symbol, quantity in self.positions.items():
            # 使用市场数据更新价格
            price = market_data.get('price', 10.0)  # 简化实现
            value += quantity * price
        
        self.portfolio_value = value
        self.portfolio_history.append({
            'timestamp': self.current_time,
            'value': value,
            'cash': self.cash,
            'positions': self.positions.copy()
        })
        
        # 更新风险控制器
        self.risk_controller.update_portfolio_value(value)

class ReturnAnalyzer:
    """收益率分析器"""
    
    def __init__(self, portfolio_values, benchmark_values=None):
        self.portfolio_values = pd.Series(portfolio_values) if not isinstance(portfolio_values, pd.Series) else portfolio_values
        self.benchmark_values = pd.Series(benchmark_values) if benchmark_values is not None else None
        self.portfolio_returns = self.calculate_returns(self.portfolio_values)
        self.benchmark_returns = self.calculate_returns(self.benchmark_values) if benchmark_values is not None else None
    
    def calculate_returns(self, values):
        """计算收益率"""
        if values is None or len(values) < 2:
            return pd.Series(dtype='float64')
        returns = values.pct_change().dropna()
        return returns
    
    def calculate_cumulative_returns(self):
        """计算累积收益率"""
        cumulative_returns = (1 + self.portfolio_returns).cumprod() - 1
        return cumulative_returns
    
    def calculate_annualized_return(self):
        """计算年化收益率"""
        if len(self.portfolio_returns) == 0:
            return 0
        total_return = self.calculate_cumulative_returns().iloc[-1]
        years = len(self.portfolio_returns) / 252
        
        if years > 0:
            annualized_return = (1 + total_return) ** (1 / years) - 1
        else:
            annualized_return = 0
        return annualized_return
    
    def calculate_volatility(self):
        """计算波动率"""
        if len(self.portfolio_returns) == 0:
            return 0
        volatility = self.portfolio_returns.std() * np.sqrt(252)
        return volatility
    
    def calculate_sharpe_ratio(self, risk_free_rate=0.02):
        """计算夏普比率"""
        if len(self.portfolio_returns) == 0:
            return 0
        excess_returns = self.portfolio_returns - risk_free_rate / 252
        sharpe_ratio = excess_returns.mean() / self.portfolio_returns.std() * np.sqrt(252)
        return sharpe_ratio
    
    def calculate_max_drawdown(self):
        """计算最大回撤"""
        cumulative_returns = (1 + self.portfolio_returns).cumprod()
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = drawdown.min()
        return max_drawdown

class RiskAnalyzer:
    """风险分析器"""
    
    def __init__(self, returns):
        self.returns = returns
    
    def calculate_var(self, confidence_level=0.05):
        """计算VaR"""
        if len(self.returns) == 0:
            return 0
        var = np.percentile(self.returns, confidence_level * 100)
        return var
    
    def calculate_cvar(self, confidence_level=0.05):
        """计算CVaR"""
        if len(self.returns) == 0:
            return 0
        var = self.calculate_var(confidence_level)
        cvar = self.returns[self.returns <= var].mean()
        return cvar

class BacktestReportGenerator:
    """回测报告生成器"""
    
    def __init__(self, backtest_results):
        self.results = backtest_results
        self.portfolio_values = [item['value'] for item in backtest_results['portfolio_history']]
        self.timestamps = [item['timestamp'] for item in backtest_results['portfolio_history']]
    
    def generate_performance_chart(self):
        """生成绩效图表"""
        if len(self.portfolio_values) < 2:
            print("数据不足，无法生成图表")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 累积收益率
        portfolio_series = pd.Series(self.portfolio_values, index=self.timestamps)
        returns = portfolio_series.pct_change().dropna()
        cumulative_returns = (1 + returns).cumprod()
        
        axes[0, 0].plot(cumulative_returns.index, cumulative_returns.values, label='Portfolio')
        axes[0, 0].set_title('Cumulative Returns')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 收益率分布
        axes[0, 1].hist(returns, bins=20, alpha=0.7, label='Portfolio')
        axes[0, 1].set_title('Return Distribution')
        axes[0, 1].legend()
        
        # 回撤
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak
        axes[1, 0].fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
        axes[1, 0].set_title('Drawdown')
        axes[1, 0].grid(True)
        
        # 组合价值
        axes[1, 1].plot(self.timestamps, self.portfolio_values)
        axes[1, 1].set_title('Portfolio Value')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        return fig
    
    def generate_summary_statistics(self):
        """生成汇总统计"""
        if len(self.portfolio_values) < 2:
            return pd.Series(dtype='float64')
        
        analyzer = ReturnAnalyzer(self.portfolio_values)
        risk_analyzer = RiskAnalyzer(analyzer.portfolio_returns)
        
        stats = {
            'Total Return': (self.portfolio_values[-1] - self.portfolio_values[0]) / self.portfolio_values[0],
            'Annualized Return': analyzer.calculate_annualized_return(),
            'Volatility': analyzer.calculate_volatility(),
            'Sharpe Ratio': analyzer.calculate_sharpe_ratio(),
            'Max Drawdown': analyzer.calculate_max_drawdown(),
            'VaR (95%)': risk_analyzer.calculate_var(0.05),
            'CVaR (95%)': risk_analyzer.calculate_cvar(0.05)
        }
        
        return pd.Series(stats)

def create_sample_strategy_signals(dates, n_stocks=50):
    """创建示例策略信号"""
    signals = {}
    
    for date in dates:
        daily_signals = {}
        # 随机生成一些信号
        selected_stocks = np.random.choice([f'stock_{i}' for i in range(n_stocks)], 
                                         size=min(10, n_stocks), replace=False)
        
        for stock in selected_stocks:
            # 生成买入/卖出信号
            signal = np.random.choice([-1, 0, 1], p=[0.2, 0.6, 0.2])
            if signal != 0:
                daily_signals[stock] = signal
        
        signals[date] = daily_signals
    
    return signals

def main():
    print("=== 高级回测系统演示 ===")
    
    # 创建日期范围
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2020, 3, 31)
    date_range = pd.date_range(start_date, end_date, freq='B')  # 工作日
    
    print(f"回测期间: {start_date.date()} 至 {end_date.date()}")
    
    # 创建事件驱动回测系统
    backtest_system = EventDrivenBacktest(initial_capital=100000)
    
    print("\n=== 生成模拟信号和事件 ===")
    
    # 生成策略信号
    strategy_signals = create_sample_strategy_signals(date_range)
    
    # 生成事件序列
    for date in date_range:
        # 市场数据事件
        market_event = MarketDataEvent(
            date, 
            'market', 
            10.0 + np.random.normal(0, 1),
            1000
        )
        backtest_system.add_event(market_event)
        
        # 信号事件
        if date in strategy_signals:
            signal_event = SignalEvent(date, strategy_signals[date])
            backtest_system.add_event(signal_event)
    
    print(f"生成了 {len(backtest_system.events)} 个事件")
    
    print("\n=== 执行回测 ===")
    
    # 运行回测
    backtest_system.process_events()
    
    print(f"执行了 {len(backtest_system.trade_history)} 笔交易")
    
    # 整理回测结果
    backtest_results = {
        'portfolio_history': backtest_system.portfolio_history,
        'trade_history': backtest_system.trade_history,
        'initial_capital': backtest_system.initial_capital,
        'final_value': backtest_system.portfolio_value
    }
    
    print("\n=== 回测结果分析 ===")
    
    if len(backtest_results['portfolio_history']) > 1:
        # 生成报告
        report_generator = BacktestReportGenerator(backtest_results)
        
        # 生成统计报告
        summary_stats = report_generator.generate_summary_statistics()
        print("绩效统计:")
        for metric, value in summary_stats.items():
            print(f"  {metric}: {value:.4f}")
        
        print("\n=== 风险分析 ===")
        
        portfolio_values = [item['value'] for item in backtest_results['portfolio_history']]
        analyzer = ReturnAnalyzer(portfolio_values)
        
        print(f"最大回撤: {analyzer.calculate_max_drawdown():.4f}")
        print(f"波动率: {analyzer.calculate_volatility():.4f}")
        print(f"夏普比率: {analyzer.calculate_sharpe_ratio():.4f}")
        
        # 风险分析
        risk_analyzer = RiskAnalyzer(analyzer.portfolio_returns)
        print(f"VaR (95%): {risk_analyzer.calculate_var(0.05):.4f}")
        print(f"CVaR (95%): {risk_analyzer.calculate_cvar(0.05):.4f}")
        
        print("\n=== 交易分析 ===")
        
        if backtest_results['trade_history']:
            trade_df = pd.DataFrame(backtest_results['trade_history'])
            
            print(f"总交易次数: {len(trade_df)}")
            buy_trades = trade_df[trade_df['quantity'] > 0]
            sell_trades = trade_df[trade_df['quantity'] < 0]
            print(f"买入交易: {len(buy_trades)} 次")
            print(f"卖出交易: {len(sell_trades)} 次")
            
            total_commission = trade_df['commission'].sum()
            print(f"总手续费: {total_commission:.2f}")
            
            if len(trade_df) > 0:
                print("\n前5笔交易:")
                for i, trade in enumerate(trade_df.head().to_dict('records')):
                    print(f"  {i+1}. {trade['timestamp'].strftime('%Y-%m-%d')} "
                          f"{trade['action']} {trade['symbol']} "
                          f"{trade['quantity']}股 @ {trade['price']:.2f}")
        
        print("\n=== 图表生成 ===")
        
        try:
            # 生成绩效图表
            fig = report_generator.generate_performance_chart()
            if fig is not None:
                # 保存图表
                fig.savefig('demo/chap06/backtest_performance.png', dpi=300, bbox_inches='tight')
                print("绩效图表已保存到 demo/chap06/backtest_performance.png")
                plt.close(fig)
        except Exception as e:
            print(f"图表生成失败: {e}")
    
    else:
        print("组合历史数据不足，无法进行详细分析")
    
    print("\n=== 风险控制演示 ===")
    
    # 演示风险控制功能
    risk_controller = RiskController(max_position_size=0.1, max_drawdown=0.15)
    
    # 模拟一个高风险订单
    high_risk_order = OrderEvent(
        datetime.now(),
        'risky_stock',
        'BUY',
        10000,  # 大量持仓
        10.0
    )
    
    portfolio_info = {
        'positions': {'risky_stock': 0},
        'cash': 50000,
        'total_value': 100000
    }
    
    risk_ok, risk_msg = risk_controller.check_order_risk(high_risk_order, portfolio_info)
    print(f"高风险订单风控结果: {risk_msg}")
    
    print("\n=== 回测系统演示完成 ===")

if __name__ == "__main__":
    main()