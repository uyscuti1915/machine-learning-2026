#!/usr/bin/env python3
"""
在线服务高级演示
基于第9章在线服务与部署内容
包含：实时数据服务、模型服务API、事件驱动处理、高可用架构等
"""

import qlib
import numpy as np
import pandas as pd
import asyncio
import json
import time
import threading
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Callable
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

# 模拟相关模块（实际使用时需要安装对应包）
class MockWebSocket:
    """模拟WebSocket"""
    def __init__(self, uri):
        self.uri = uri
        self.connected = True
    
    async def send(self, message):
        print(f"WebSocket发送: {message}")
    
    async def recv(self):
        await asyncio.sleep(0.1)
        return json.dumps({
            'symbol': 'AAPL',
            'price': 150 + np.random.normal(0, 2),
            'volume': np.random.randint(1000, 10000),
            'timestamp': datetime.now().isoformat()
        })

class MockRedis:
    """模拟Redis"""
    def __init__(self):
        self._data = {}
    
    def set(self, key, value, ex=None):
        self._data[key] = value
        if ex:
            # 简化：忽略过期时间
            pass
    
    def get(self, key):
        return self._data.get(key)
    
    def exists(self, key):
        return key in self._data

class MockFlask:
    """模拟Flask应用"""
    def __init__(self, name):
        self.name = name
        self.routes = {}
    
    def route(self, path, methods=None):
        def decorator(func):
            self.routes[path] = func
            return func
        return decorator
    
    def run(self, host='localhost', port=5000, debug=False):
        print(f"Flask应用运行在 http://{host}:{port}")

# 初始化Qlib
print("初始化Qlib...")
qlib.init(mount_path="~/.qlib/qlib_data/cn_data", region="cn")

@dataclass
class MarketData:
    """市场数据结构"""
    symbol: str
    price: float
    volume: int
    timestamp: datetime
    bid: float = None
    ask: float = None

@dataclass
class TradingSignal:
    """交易信号结构"""
    symbol: str
    action: str  # 'buy', 'sell', 'hold'
    confidence: float
    price: float
    timestamp: datetime

class RealTimeDataService:
    """实时数据服务"""
    
    def __init__(self, data_sources: List[str]):
        self.data_sources = data_sources
        self.subscribers = []
        self.running = False
        self.data_buffer = deque(maxlen=1000)
    
    def subscribe(self, callback: Callable):
        """订阅数据更新"""
        self.subscribers.append(callback)
    
    def unsubscribe(self, callback: Callable):
        """取消订阅"""
        if callback in self.subscribers:
            self.subscribers.remove(callback)
    
    async def start_streaming(self):
        """开始数据流"""
        self.running = True
        print("启动实时数据流...")
        
        try:
            while self.running:
                # 模拟从多个数据源获取数据
                for source in self.data_sources:
                    try:
                        # 模拟实时数据
                        mock_ws = MockWebSocket(source)
                        data_str = await mock_ws.recv()
                        data = json.loads(data_str)
                        
                        market_data = MarketData(
                            symbol=data['symbol'],
                            price=data['price'],
                            volume=data['volume'],
                            timestamp=datetime.fromisoformat(data['timestamp']),
                            bid=data['price'] - 0.01,
                            ask=data['price'] + 0.01
                        )
                        
                        # 缓存数据
                        self.data_buffer.append(market_data)
                        
                        # 通知所有订阅者
                        for callback in self.subscribers:
                            await callback(market_data)
                        
                    except Exception as e:
                        print(f"数据源 {source} 错误: {e}")
                
                await asyncio.sleep(1)  # 1秒间隔
                
        except Exception as e:
            print(f"数据流错误: {e}")
        finally:
            self.running = False
    
    def stop_streaming(self):
        """停止数据流"""
        self.running = False
        print("停止实时数据流")
    
    def get_latest_data(self, symbol: str = None) -> List[MarketData]:
        """获取最新数据"""
        if symbol:
            return [data for data in self.data_buffer if data.symbol == symbol]
        return list(self.data_buffer)

class DataCacheManager:
    """数据缓存管理器"""
    
    def __init__(self):
        self.redis_client = MockRedis()
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0
        }
    
    def get(self, key: str):
        """获取缓存数据"""
        data = self.redis_client.get(key)
        if data:
            self.cache_stats['hits'] += 1
            return json.loads(data) if isinstance(data, str) else data
        else:
            self.cache_stats['misses'] += 1
            return None
    
    def set(self, key: str, value: Any, expiry: int = 3600):
        """设置缓存数据"""
        data = json.dumps(value, default=str) if not isinstance(value, str) else value
        self.redis_client.set(key, data, ex=expiry)
        self.cache_stats['sets'] += 1
    
    def get_cache_stats(self):
        """获取缓存统计"""
        total = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = self.cache_stats['hits'] / total if total > 0 else 0
        return {
            **self.cache_stats,
            'hit_rate': hit_rate
        }

class ModelServiceAPI:
    """模型服务API"""
    
    def __init__(self):
        self.app = MockFlask(__name__)
        self.cache = DataCacheManager()
        self.model = None
        self.model_version = "1.0.0"
        self.setup_routes()
    
    def setup_routes(self):
        """设置API路由"""
        
        @self.app.route('/predict', methods=['POST'])
        def predict():
            return self.handle_prediction_request()
        
        @self.app.route('/model/info', methods=['GET'])
        def model_info():
            return self.get_model_info()
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            return self.health_check()
    
    def handle_prediction_request(self):
        """处理预测请求"""
        try:
            # 模拟请求数据解析
            request_data = {
                'symbols': ['AAPL', 'GOOGL', 'MSFT'],
                'features': np.random.randn(3, 158).tolist()  # Alpha158特征
            }
            
            # 检查缓存
            cache_key = f"prediction_{hash(str(request_data))}"
            cached_result = self.cache.get(cache_key)
            
            if cached_result:
                print("返回缓存的预测结果")
                return {
                    'status': 'success',
                    'predictions': cached_result,
                    'cached': True,
                    'model_version': self.model_version
                }
            
            # 生成预测
            predictions = self.generate_predictions(request_data)
            
            # 缓存结果
            self.cache.set(cache_key, predictions, expiry=300)  # 5分钟过期
            
            return {
                'status': 'success',
                'predictions': predictions,
                'cached': False,
                'model_version': self.model_version
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def generate_predictions(self, request_data):
        """生成预测结果"""
        predictions = {}
        
        for i, symbol in enumerate(request_data['symbols']):
            # 模拟模型预测
            features = np.array(request_data['features'][i])
            prediction_score = np.random.randn() * 0.1  # 模拟预测分数
            
            predictions[symbol] = {
                'score': prediction_score,
                'confidence': abs(prediction_score),
                'timestamp': datetime.now().isoformat()
            }
        
        return predictions
    
    def get_model_info(self):
        """获取模型信息"""
        return {
            'model_name': 'LightGBM',
            'version': self.model_version,
            'features': 158,
            'last_trained': '2024-01-01T00:00:00',
            'performance': {
                'ic': 0.05,
                'rank_ic': 0.08,
                'sharpe_ratio': 1.2
            }
        }
    
    def health_check(self):
        """健康检查"""
        cache_stats = self.cache.get_cache_stats()
        
        return {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'model_version': self.model_version,
            'cache_stats': cache_stats
        }

class RealTimeDataProcessor:
    """实时数据处理器"""
    
    def __init__(self):
        self.indicators = {}
        self.signal_callbacks = []
    
    def add_indicator(self, name: str, calculator: Callable):
        """添加技术指标计算器"""
        self.indicators[name] = calculator
    
    def add_signal_callback(self, callback: Callable):
        """添加信号回调"""
        self.signal_callbacks.append(callback)
    
    async def process_market_data(self, market_data: MarketData):
        """处理市场数据"""
        try:
            # 计算技术指标
            indicators = {}
            for name, calculator in self.indicators.items():
                indicators[name] = await calculator(market_data)
            
            # 生成交易信号
            signal = self.generate_signal(market_data, indicators)
            
            if signal:
                # 通知所有信号回调
                for callback in self.signal_callbacks:
                    await callback(signal)
            
        except Exception as e:
            print(f"数据处理错误: {e}")
    
    def generate_signal(self, market_data: MarketData, indicators: Dict) -> Optional[TradingSignal]:
        """生成交易信号"""
        # 简单的信号生成逻辑
        ma_short = indicators.get('ma_short', 0)
        ma_long = indicators.get('ma_long', 0)
        
        if ma_short > ma_long:
            return TradingSignal(
                symbol=market_data.symbol,
                action='buy',
                confidence=0.7,
                price=market_data.price,
                timestamp=market_data.timestamp
            )
        elif ma_short < ma_long:
            return TradingSignal(
                symbol=market_data.symbol,
                action='sell',
                confidence=0.6,
                price=market_data.price,
                timestamp=market_data.timestamp
            )
        
        return None

class TechnicalIndicatorProcessor:
    """技术指标处理器"""
    
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.price_history = defaultdict(lambda: deque(maxlen=window_size))
    
    async def calculate_ma(self, market_data: MarketData, period: int = 10) -> float:
        """计算移动平均"""
        symbol = market_data.symbol
        self.price_history[symbol].append(market_data.price)
        
        if len(self.price_history[symbol]) >= period:
            return sum(list(self.price_history[symbol])[-period:]) / period
        else:
            return market_data.price
    
    async def calculate_rsi(self, market_data: MarketData, period: int = 14) -> float:
        """计算RSI"""
        symbol = market_data.symbol
        self.price_history[symbol].append(market_data.price)
        
        if len(self.price_history[symbol]) < period + 1:
            return 50.0  # 中性值
        
        prices = list(self.price_history[symbol])
        gains = []
        losses = []
        
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(-change)
        
        if len(gains) >= period:
            avg_gain = sum(gains[-period:]) / period
            avg_loss = sum(losses[-period:]) / period
            
            if avg_loss == 0:
                return 100.0
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        return 50.0

class EventHandler:
    """事件处理器"""
    
    def __init__(self):
        self.handlers = {}
    
    def register_handler(self, event_type: str, handler: Callable):
        """注册事件处理器"""
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)
    
    async def emit_event(self, event_type: str, data: Any):
        """发出事件"""
        if event_type in self.handlers:
            for handler in self.handlers[event_type]:
                try:
                    await handler(data)
                except Exception as e:
                    print(f"事件处理错误 {event_type}: {e}")

class RealTimeTradingSystem:
    """实时交易系统"""
    
    def __init__(self):
        self.data_service = RealTimeDataService(['wss://mock-data-feed.com'])
        self.model_api = ModelServiceAPI()
        self.data_processor = RealTimeDataProcessor()
        self.indicator_processor = TechnicalIndicatorProcessor()
        self.event_handler = EventHandler()
        
        self.positions = {}
        self.orders = []
        self.pnl = 0.0
        
        self.setup_system()
    
    def setup_system(self):
        """设置系统组件"""
        # 设置数据订阅
        self.data_service.subscribe(self.on_market_data)
        
        # 设置技术指标
        self.data_processor.add_indicator('ma_short', 
            lambda data: self.indicator_processor.calculate_ma(data, 5))
        self.data_processor.add_indicator('ma_long', 
            lambda data: self.indicator_processor.calculate_ma(data, 20))
        self.data_processor.add_indicator('rsi', 
            lambda data: self.indicator_processor.calculate_rsi(data, 14))
        
        # 设置信号回调
        self.data_processor.add_signal_callback(self.on_trading_signal)
        
        # 注册事件处理器
        self.event_handler.register_handler('order_fill', self.on_order_fill)
        self.event_handler.register_handler('position_update', self.on_position_update)
    
    async def on_market_data(self, market_data: MarketData):
        """处理市场数据"""
        print(f"收到市场数据: {market_data.symbol} @ {market_data.price:.2f}")
        
        # 处理数据并生成信号
        await self.data_processor.process_market_data(market_data)
        
        # 更新持仓盈亏
        if market_data.symbol in self.positions:
            position = self.positions[market_data.symbol]
            position['current_price'] = market_data.price
            position['unrealized_pnl'] = (market_data.price - position['avg_price']) * position['quantity']
    
    async def on_trading_signal(self, signal: TradingSignal):
        """处理交易信号"""
        print(f"收到交易信号: {signal.symbol} {signal.action} (置信度: {signal.confidence:.2f})")
        
        # 风险检查
        if signal.confidence < 0.5:
            print("信号置信度过低，忽略")
            return
        
        # 生成订单
        order = {
            'symbol': signal.symbol,
            'action': signal.action,
            'quantity': 100,  # 固定数量
            'price': signal.price,
            'timestamp': signal.timestamp,
            'status': 'pending'
        }
        
        self.orders.append(order)
        
        # 模拟订单执行
        await self.execute_order(order)
    
    async def execute_order(self, order: Dict):
        """执行订单"""
        try:
            # 模拟订单执行延迟
            await asyncio.sleep(0.1)
            
            symbol = order['symbol']
            action = order['action']
            quantity = order['quantity']
            price = order['price']
            
            # 更新持仓
            if symbol not in self.positions:
                self.positions[symbol] = {
                    'quantity': 0,
                    'avg_price': 0,
                    'unrealized_pnl': 0,
                    'realized_pnl': 0
                }
            
            position = self.positions[symbol]
            
            if action == 'buy':
                new_quantity = position['quantity'] + quantity
                if new_quantity != 0:
                    position['avg_price'] = (position['avg_price'] * position['quantity'] + price * quantity) / new_quantity
                position['quantity'] = new_quantity
                
            elif action == 'sell':
                if position['quantity'] >= quantity:
                    realized_pnl = (price - position['avg_price']) * quantity
                    position['realized_pnl'] += realized_pnl
                    position['quantity'] -= quantity
                    self.pnl += realized_pnl
                else:
                    print(f"持仓不足，无法卖出 {quantity} 股 {symbol}")
                    return
            
            order['status'] = 'filled'
            
            # 发出事件
            await self.event_handler.emit_event('order_fill', order)
            await self.event_handler.emit_event('position_update', position)
            
            print(f"订单执行: {action} {quantity} {symbol} @ {price:.2f}")
            
        except Exception as e:
            print(f"订单执行错误: {e}")
            order['status'] = 'failed'
    
    async def on_order_fill(self, order: Dict):
        """订单成交处理"""
        print(f"订单成交通知: {order['symbol']} {order['action']} {order['quantity']}")
    
    async def on_position_update(self, position: Dict):
        """持仓更新处理"""
        print(f"持仓更新: 数量={position['quantity']}, 均价={position['avg_price']:.2f}, 未实现盈亏={position['unrealized_pnl']:.2f}")
    
    def get_portfolio_summary(self):
        """获取组合摘要"""
        total_value = 0
        total_unrealized_pnl = 0
        
        for symbol, position in self.positions.items():
            if position['quantity'] > 0:
                market_value = position['quantity'] * position['current_price']
                total_value += market_value
                total_unrealized_pnl += position['unrealized_pnl']
        
        return {
            'total_positions': len([p for p in self.positions.values() if p['quantity'] > 0]),
            'total_value': total_value,
            'realized_pnl': self.pnl,
            'unrealized_pnl': total_unrealized_pnl,
            'total_pnl': self.pnl + total_unrealized_pnl,
            'total_orders': len(self.orders)
        }

class HighAvailabilitySystem:
    """高可用系统"""
    
    def __init__(self):
        self.services = {}
        self.health_checks = {}
        self.failover_enabled = True
    
    def register_service(self, name: str, service: Any, health_check: Callable):
        """注册服务"""
        self.services[name] = {
            'instance': service,
            'status': 'unknown',
            'last_check': None
        }
        self.health_checks[name] = health_check
    
    async def monitor_services(self):
        """监控服务健康状态"""
        while True:
            for service_name, health_check in self.health_checks.items():
                try:
                    is_healthy = await health_check()
                    self.services[service_name]['status'] = 'healthy' if is_healthy else 'unhealthy'
                    self.services[service_name]['last_check'] = datetime.now()
                    
                    if not is_healthy and self.failover_enabled:
                        await self.handle_service_failure(service_name)
                        
                except Exception as e:
                    print(f"服务 {service_name} 健康检查失败: {e}")
                    self.services[service_name]['status'] = 'error'
                    
                    if self.failover_enabled:
                        await self.handle_service_failure(service_name)
            
            await asyncio.sleep(10)  # 10秒检查间隔
    
    async def handle_service_failure(self, service_name: str):
        """处理服务故障"""
        print(f"处理服务故障: {service_name}")
        
        # 模拟故障转移逻辑
        if service_name == 'data_service':
            print("启动备用数据服务...")
            # 实际实现中会启动备用服务
        elif service_name == 'model_api':
            print("启动备用模型API...")
            # 实际实现中会启动备用API服务
        
        # 发送告警
        await self.send_alert(f"服务 {service_name} 故障，已启动故障转移")
    
    async def send_alert(self, message: str):
        """发送告警"""
        print(f"🚨 告警: {message}")
        # 实际实现中会发送邮件、短信或推送通知

async def demo_real_time_system():
    """演示实时交易系统"""
    print("\n=== 实时交易系统演示 ===")
    
    # 创建交易系统
    trading_system = RealTimeTradingSystem()
    
    # 启动高可用监控
    ha_system = HighAvailabilitySystem()
    
    # 注册服务到高可用系统
    ha_system.register_service('data_service', trading_system.data_service,
                              lambda: asyncio.create_task(asyncio.sleep(0.1)) and True)
    ha_system.register_service('model_api', trading_system.model_api,
                              lambda: True)
    
    print("启动实时交易系统组件...")
    
    # 创建任务
    tasks = [
        asyncio.create_task(trading_system.data_service.start_streaming()),
        asyncio.create_task(ha_system.monitor_services())
    ]
    
    try:
        # 运行系统一段时间
        await asyncio.sleep(10)  # 运行10秒
        
        # 显示系统状态
        portfolio_summary = trading_system.get_portfolio_summary()
        print(f"\n=== 系统状态摘要 ===")
        print(f"持仓数量: {portfolio_summary['total_positions']}")
        print(f"总市值: {portfolio_summary['total_value']:.2f}")
        print(f"已实现盈亏: {portfolio_summary['realized_pnl']:.2f}")
        print(f"未实现盈亏: {portfolio_summary['unrealized_pnl']:.2f}")
        print(f"总盈亏: {portfolio_summary['total_pnl']:.2f}")
        print(f"总订单数: {portfolio_summary['total_orders']}")
        
        # 显示服务状态
        print(f"\n=== 服务状态 ===")
        for service_name, service_info in ha_system.services.items():
            print(f"{service_name}: {service_info['status']} (最后检查: {service_info['last_check']})")
        
        # 显示缓存统计
        cache_stats = trading_system.model_api.cache.get_cache_stats()
        print(f"\n=== 缓存统计 ===")
        print(f"命中次数: {cache_stats['hits']}")
        print(f"未命中次数: {cache_stats['misses']}")
        print(f"命中率: {cache_stats['hit_rate']:.2%}")
        
    finally:
        # 停止所有任务
        trading_system.data_service.stop_streaming()
        for task in tasks:
            task.cancel()
        
        print("\n实时交易系统演示完成")

def demo_model_api():
    """演示模型API服务"""
    print("\n=== 模型API服务演示 ===")
    
    # 创建模型服务
    model_api = ModelServiceAPI()
    
    print("启动模型API服务...")
    model_api.app.run(host='localhost', port=5000)
    
    # 模拟API调用
    print("\n模拟API调用:")
    
    # 健康检查
    health_result = model_api.health_check()
    print(f"健康检查: {health_result}")
    
    # 获取模型信息
    model_info = model_api.get_model_info()
    print(f"模型信息: {model_info}")
    
    # 预测请求
    prediction_result = model_api.handle_prediction_request()
    print(f"预测结果: {prediction_result}")
    
    # 再次请求（应该返回缓存结果）
    prediction_result_cached = model_api.handle_prediction_request()
    print(f"缓存预测结果: {prediction_result_cached}")

def demo_data_cache():
    """演示数据缓存管理"""
    print("\n=== 数据缓存管理演示 ===")
    
    cache = DataCacheManager()
    
    print("测试缓存功能...")
    
    # 设置缓存
    test_data = {
        'symbol': 'AAPL',
        'price': 150.25,
        'timestamp': datetime.now().isoformat()
    }
    
    cache.set('market_data:AAPL', test_data)
    print(f"设置缓存: market_data:AAPL")
    
    # 获取缓存
    cached_data = cache.get('market_data:AAPL')
    print(f"获取缓存: {cached_data}")
    
    # 获取不存在的缓存
    non_existent = cache.get('market_data:GOOGL')
    print(f"不存在的缓存: {non_existent}")
    
    # 显示缓存统计
    stats = cache.get_cache_stats()
    print(f"缓存统计: {stats}")

async def main():
    """主函数"""
    print("=== 在线服务高级演示 ===")
    
    # 演示数据缓存
    demo_data_cache()
    
    # 演示模型API服务
    demo_model_api()
    
    # 演示技术指标处理器
    print("\n=== 技术指标处理器演示 ===")
    
    indicator_processor = TechnicalIndicatorProcessor()
    
    # 模拟市场数据
    test_prices = [100, 101, 99, 102, 105, 103, 106, 108, 107, 109]
    
    for i, price in enumerate(test_prices):
        market_data = MarketData(
            symbol='TEST',
            price=price,
            volume=1000,
            timestamp=datetime.now() + timedelta(seconds=i)
        )
        
        ma = await indicator_processor.calculate_ma(market_data, 5)
        rsi = await indicator_processor.calculate_rsi(market_data, 5)
        
        print(f"价格: {price:.2f}, MA(5): {ma:.2f}, RSI(5): {rsi:.2f}")
    
    # 演示事件处理器
    print("\n=== 事件处理器演示 ===")
    
    event_handler = EventHandler()
    
    async def on_price_update(data):
        print(f"价格更新事件: {data}")
    
    async def on_order_event(data):
        print(f"订单事件: {data}")
    
    # 注册事件处理器
    event_handler.register_handler('price_update', on_price_update)
    event_handler.register_handler('order', on_order_event)
    
    # 发出事件
    await event_handler.emit_event('price_update', {'symbol': 'AAPL', 'price': 150.25})
    await event_handler.emit_event('order', {'symbol': 'GOOGL', 'action': 'buy', 'quantity': 100})
    
    # 运行实时系统演示
    await demo_real_time_system()
    
    print("\n在线服务高级演示完成！")
    print("\n主要演示内容:")
    print("1. 实时数据服务和订阅机制")
    print("2. 模型服务API和缓存管理")
    print("3. 技术指标实时计算")
    print("4. 事件驱动架构")
    print("5. 实时交易系统集成")
    print("6. 高可用性监控和故障转移")

if __name__ == "__main__":
    asyncio.run(main())