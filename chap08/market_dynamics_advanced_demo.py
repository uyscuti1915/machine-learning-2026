#!/usr/bin/env python3
"""
市场动态适应高级演示
基于第8章市场动态适应内容
包含：概念漂移检测、元学习方法、自适应模型等
"""

import qlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# 初始化Qlib
print("初始化Qlib...")
qlib.init(mount_path="~/.qlib/qlib_data/cn_data", region="cn")

class ConceptDriftDetector(ABC):
    """概念漂移检测基类"""
    
    def __init__(self, window_size=252, threshold=0.05):
        self.window_size = window_size
        self.threshold = threshold
        self.history = []
        
    @abstractmethod
    def detect_drift(self, new_data: np.ndarray) -> bool:
        """检测概念漂移"""
        pass
    
    def update_history(self, data: np.ndarray):
        """更新历史数据"""
        self.history.append(data)
        if len(self.history) > self.window_size:
            self.history.pop(0)

class StatisticalDriftDetector(ConceptDriftDetector):
    """基于统计检验的概念漂移检测"""
    
    def __init__(self, test_method='ks', **kwargs):
        super().__init__(**kwargs)
        self.test_method = test_method
        
    def detect_drift(self, new_data: np.ndarray) -> Dict[str, Any]:
        """检测统计分布漂移"""
        if len(self.history) < 2:
            self.update_history(new_data)
            return {'drift_detected': False, 'p_value': 1.0, 'statistic': 0.0}
        
        # 获取历史数据基线
        baseline_data = np.concatenate(self.history[-self.window_size//2:])
        
        if self.test_method == 'ks':
            # Kolmogorov-Smirnov检验
            statistic, p_value = stats.ks_2samp(baseline_data, new_data)
        elif self.test_method == 'anderson':
            # Anderson-Darling检验
            statistic, critical_values, significance_level = stats.anderson_ksamp([baseline_data, new_data])
            p_value = significance_level
        else:
            # 默认使用t检验
            statistic, p_value = stats.ttest_ind(baseline_data, new_data)
        
        drift_detected = p_value < self.threshold
        
        self.update_history(new_data)
        
        return {
            'drift_detected': drift_detected,
            'p_value': p_value,
            'statistic': statistic,
            'test_method': self.test_method
        }

class PerformanceDriftDetector(ConceptDriftDetector):
    """基于模型性能的概念漂移检测"""
    
    def __init__(self, performance_threshold=0.1, **kwargs):
        super().__init__(**kwargs)
        self.performance_threshold = performance_threshold
        self.baseline_performance = None
        
    def detect_drift(self, predictions: np.ndarray, actuals: np.ndarray) -> Dict[str, Any]:
        """基于预测性能检测漂移"""
        current_mse = mean_squared_error(actuals, predictions)
        
        if self.baseline_performance is None:
            self.baseline_performance = current_mse
            return {
                'drift_detected': False,
                'performance_degradation': 0.0,
                'current_mse': current_mse
            }
        
        # 计算性能退化
        performance_degradation = (current_mse - self.baseline_performance) / self.baseline_performance
        drift_detected = performance_degradation > self.performance_threshold
        
        # 更新基线（指数移动平均）
        alpha = 0.1
        self.baseline_performance = alpha * current_mse + (1 - alpha) * self.baseline_performance
        
        return {
            'drift_detected': drift_detected,
            'performance_degradation': performance_degradation,
            'current_mse': current_mse,
            'baseline_mse': self.baseline_performance
        }

class AnomalyDriftDetector(ConceptDriftDetector):
    """基于异常检测的概念漂移检测"""
    
    def __init__(self, contamination=0.1, **kwargs):
        super().__init__(**kwargs)
        self.contamination = contamination
        self.detector = IsolationForest(contamination=contamination, random_state=42)
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def detect_drift(self, new_data: np.ndarray) -> Dict[str, Any]:
        """基于异常检测的漂移检测"""
        if not self.is_fitted and len(self.history) >= 10:
            # 训练异常检测器
            baseline_data = np.vstack(self.history)
            baseline_data_scaled = self.scaler.fit_transform(baseline_data.reshape(-1, 1))
            self.detector.fit(baseline_data_scaled)
            self.is_fitted = True
        
        if not self.is_fitted:
            self.update_history(new_data)
            return {
                'drift_detected': False,
                'anomaly_score': 0.0,
                'outlier_fraction': 0.0
            }
        
        # 检测异常
        new_data_scaled = self.scaler.transform(new_data.reshape(-1, 1))
        anomaly_scores = self.detector.decision_function(new_data_scaled)
        outlier_predictions = self.detector.predict(new_data_scaled)
        
        # 计算异常比例
        outlier_fraction = (outlier_predictions == -1).mean()
        avg_anomaly_score = anomaly_scores.mean()
        
        # 如果异常比例超过阈值，认为发生了概念漂移
        drift_detected = outlier_fraction > self.contamination * 2
        
        self.update_history(new_data)
        
        return {
            'drift_detected': drift_detected,
            'anomaly_score': avg_anomaly_score,
            'outlier_fraction': outlier_fraction
        }

class AdaptiveModelFramework:
    """自适应模型框架"""
    
    def __init__(self, base_model, drift_detector, adaptation_strategy='retrain'):
        self.base_model = base_model
        self.drift_detector = drift_detector
        self.adaptation_strategy = adaptation_strategy
        
        self.model_history = []
        self.performance_history = []
        self.drift_history = []
        
    def fit(self, X, y):
        """训练基础模型"""
        self.base_model.fit(X, y)
        self.model_history.append(self.base_model)
        
    def predict(self, X):
        """预测"""
        return self.base_model.predict(X)
    
    def adapt_model(self, X_new, y_new, X_val=None, y_val=None):
        """自适应模型更新"""
        # 检测概念漂移
        drift_result = self.drift_detector.detect_drift(X_new.flatten())
        self.drift_history.append(drift_result)
        
        if drift_result['drift_detected']:
            print(f"检测到概念漂移: {drift_result}")
            
            if self.adaptation_strategy == 'retrain':
                # 完全重新训练
                self.base_model.fit(X_new, y_new)
                
            elif self.adaptation_strategy == 'incremental':
                # 增量学习（如果模型支持）
                if hasattr(self.base_model, 'partial_fit'):
                    self.base_model.partial_fit(X_new, y_new)
                else:
                    # 使用新旧数据混合训练
                    if len(self.model_history) > 0:
                        # 简化：假设有历史数据
                        self.base_model.fit(X_new, y_new)
                    
            elif self.adaptation_strategy == 'ensemble':
                # 创建新模型并与旧模型集成
                from sklearn.base import clone
                new_model = clone(self.base_model)
                new_model.fit(X_new, y_new)
                
                # 简化的集成策略
                self.model_history.append(new_model)
                if len(self.model_history) > 3:  # 最多保持3个模型
                    self.model_history.pop(0)
        
        # 评估性能
        if X_val is not None and y_val is not None:
            predictions = self.predict(X_val)
            mse = mean_squared_error(y_val, predictions)
            self.performance_history.append(mse)
    
    def get_drift_statistics(self):
        """获取漂移统计信息"""
        if not self.drift_history:
            return {}
        
        drift_count = sum(1 for d in self.drift_history if d['drift_detected'])
        drift_rate = drift_count / len(self.drift_history)
        
        return {
            'total_checks': len(self.drift_history),
            'drift_detections': drift_count,
            'drift_rate': drift_rate,
            'recent_drifts': self.drift_history[-10:]  # 最近10次检测
        }

class MetaLearningFramework:
    """元学习框架"""
    
    def __init__(self, base_learners, meta_learner, task_window=50):
        self.base_learners = base_learners
        self.meta_learner = meta_learner
        self.task_window = task_window
        
        self.task_history = []
        self.meta_features = []
        self.performance_matrix = []
        
    def extract_meta_features(self, X, y):
        """提取元特征"""
        meta_features = {}
        
        # 数据统计特征
        meta_features['n_samples'] = len(X)
        meta_features['n_features'] = X.shape[1] if X.ndim > 1 else 1
        meta_features['mean_target'] = np.mean(y)
        meta_features['std_target'] = np.std(y)
        meta_features['skew_target'] = stats.skew(y)
        meta_features['kurtosis_target'] = stats.kurtosis(y)
        
        # 特征相关性
        if X.ndim > 1 and X.shape[1] > 1:
            correlation_matrix = np.corrcoef(X.T)
            meta_features['max_correlation'] = np.max(np.abs(correlation_matrix - np.eye(X.shape[1])))
            meta_features['mean_correlation'] = np.mean(np.abs(correlation_matrix - np.eye(X.shape[1])))
        else:
            meta_features['max_correlation'] = 0
            meta_features['mean_correlation'] = 0
        
        # 数据复杂度
        if len(y) > 1:
            meta_features['target_autocorr'] = np.corrcoef(y[:-1], y[1:])[0, 1] if len(y) > 2 else 0
        else:
            meta_features['target_autocorr'] = 0
        
        return meta_features
    
    def evaluate_base_learners(self, X_train, y_train, X_val, y_val):
        """评估基学习器性能"""
        performances = []
        
        for name, learner in self.base_learners.items():
            try:
                learner.fit(X_train, y_train)
                predictions = learner.predict(X_val)
                mse = mean_squared_error(y_val, predictions)
                performances.append(mse)
            except Exception as e:
                print(f"基学习器 {name} 评估失败: {e}")
                performances.append(float('inf'))
        
        return performances
    
    def learn_task(self, X_train, y_train, X_val, y_val):
        """学习新任务"""
        # 提取元特征
        meta_features = self.extract_meta_features(X_train, y_train)
        
        # 评估所有基学习器
        performances = self.evaluate_base_learners(X_train, y_train, X_val, y_val)
        
        # 记录任务
        self.task_history.append({
            'meta_features': meta_features,
            'performances': performances,
            'timestamp': datetime.now()
        })
        
        # 保持任务窗口大小
        if len(self.task_history) > self.task_window:
            self.task_history.pop(0)
        
        # 更新元学习器
        if len(self.task_history) >= 5:  # 至少需要5个任务来训练元学习器
            self.update_meta_learner()
        
        # 返回最佳基学习器
        best_learner_idx = np.argmin(performances)
        best_learner_name = list(self.base_learners.keys())[best_learner_idx]
        
        return best_learner_name, performances[best_learner_idx]
    
    def update_meta_learner(self):
        """更新元学习器"""
        try:
            # 准备训练数据
            X_meta = []
            y_meta = []
            
            for task in self.task_history:
                meta_features = list(task['meta_features'].values())
                best_learner_idx = np.argmin(task['performances'])
                
                X_meta.append(meta_features)
                y_meta.append(best_learner_idx)
            
            X_meta = np.array(X_meta)
            y_meta = np.array(y_meta)
            
            # 训练元学习器
            if len(set(y_meta)) > 1:  # 确保有多个类别
                self.meta_learner.fit(X_meta, y_meta)
                
        except Exception as e:
            print(f"元学习器更新失败: {e}")
    
    def predict_best_learner(self, X, y):
        """预测最佳学习器"""
        try:
            # 提取元特征
            meta_features = self.extract_meta_features(X, y)
            feature_vector = np.array(list(meta_features.values())).reshape(1, -1)
            
            # 使用元学习器预测
            if hasattr(self.meta_learner, 'predict'):
                predicted_idx = self.meta_learner.predict(feature_vector)[0]
                learner_names = list(self.base_learners.keys())
                
                if 0 <= predicted_idx < len(learner_names):
                    return learner_names[predicted_idx]
            
            # 如果预测失败，返回默认选择
            return list(self.base_learners.keys())[0]
            
        except Exception as e:
            print(f"预测最佳学习器失败: {e}")
            return list(self.base_learners.keys())[0]

class MarketRegimeDetector:
    """市场状态检测器"""
    
    def __init__(self, regime_indicators=['volatility', 'trend', 'volume']):
        self.regime_indicators = regime_indicators
        self.regime_history = []
        self.current_regime = None
        
    def calculate_market_indicators(self, price_data, volume_data=None):
        """计算市场指标"""
        indicators = {}
        
        if 'volatility' in self.regime_indicators:
            # 计算波动率
            returns = price_data.pct_change().dropna()
            indicators['volatility'] = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
            
        if 'trend' in self.regime_indicators:
            # 计算趋势强度
            ma_short = price_data.rolling(10).mean()
            ma_long = price_data.rolling(50).mean()
            trend_signal = (ma_short.iloc[-1] - ma_long.iloc[-1]) / ma_long.iloc[-1]
            indicators['trend'] = trend_signal
            
        if 'volume' in self.regime_indicators and volume_data is not None:
            # 计算成交量指标
            volume_ma = volume_data.rolling(20).mean()
            volume_ratio = volume_data.iloc[-1] / volume_ma.iloc[-1]
            indicators['volume'] = volume_ratio
            
        return indicators
    
    def detect_regime(self, price_data, volume_data=None):
        """检测市场状态"""
        indicators = self.calculate_market_indicators(price_data, volume_data)
        
        # 简化的状态分类规则
        volatility = indicators.get('volatility', 0.2)
        trend = indicators.get('trend', 0)
        volume = indicators.get('volume', 1.0)
        
        if volatility > 0.3:
            regime = 'high_volatility'
        elif abs(trend) > 0.05:
            regime = 'trending' if trend > 0 else 'declining'
        elif volume > 1.5:
            regime = 'high_activity'
        else:
            regime = 'normal'
        
        # 记录状态变化
        if self.current_regime != regime:
            self.regime_history.append({
                'previous_regime': self.current_regime,
                'new_regime': regime,
                'timestamp': datetime.now(),
                'indicators': indicators
            })
            self.current_regime = regime
            
        return regime, indicators
    
    def get_regime_statistics(self):
        """获取状态统计"""
        if not self.regime_history:
            return {}
        
        regime_counts = {}
        for record in self.regime_history:
            regime = record['new_regime']
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        return {
            'current_regime': self.current_regime,
            'regime_changes': len(self.regime_history),
            'regime_distribution': regime_counts,
            'recent_changes': self.regime_history[-5:]  # 最近5次变化
        }

def demo_concept_drift_detection():
    """演示概念漂移检测"""
    print("\n=== 概念漂移检测演示 ===")
    
    # 生成模拟数据：正常数据 + 漂移数据
    np.random.seed(42)
    
    # 正常期间数据（均值0，标准差1）
    normal_data = np.random.normal(0, 1, 1000)
    
    # 漂移期间数据（均值2，标准差1.5）
    drift_data = np.random.normal(2, 1.5, 500)
    
    # 测试统计漂移检测器
    print("统计漂移检测器测试:")
    stat_detector = StatisticalDriftDetector(test_method='ks', threshold=0.05)
    
    # 添加正常数据
    for i in range(0, len(normal_data), 100):
        batch = normal_data[i:i+100]
        result = stat_detector.detect_drift(batch)
        if result['drift_detected']:
            print(f"  批次 {i//100 + 1}: 检测到漂移 (p-value: {result['p_value']:.4f})")
    
    # 添加漂移数据
    print("  添加漂移数据...")
    for i in range(0, len(drift_data), 100):
        batch = drift_data[i:i+100]
        result = stat_detector.detect_drift(batch)
        if result['drift_detected']:
            print(f"  漂移批次 {i//100 + 1}: 检测到漂移 (p-value: {result['p_value']:.4f})")
    
    # 测试异常检测漂移检测器
    print("\n异常检测漂移检测器测试:")
    anomaly_detector = AnomalyDriftDetector(contamination=0.1)
    
    # 正常数据
    for i in range(0, len(normal_data), 100):
        batch = normal_data[i:i+100]
        result = anomaly_detector.detect_drift(batch)
        if result['drift_detected']:
            print(f"  批次 {i//100 + 1}: 检测到异常漂移 (异常比例: {result['outlier_fraction']:.4f})")
    
    # 漂移数据
    print("  添加漂移数据...")
    for i in range(0, len(drift_data), 100):
        batch = drift_data[i:i+100]
        result = anomaly_detector.detect_drift(batch)
        if result['drift_detected']:
            print(f"  漂移批次 {i//100 + 1}: 检测到异常漂移 (异常比例: {result['outlier_fraction']:.4f})")

def demo_adaptive_model():
    """演示自适应模型"""
    print("\n=== 自适应模型演示 ===")
    
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    
    # 生成模拟数据
    np.random.seed(42)
    
    # 训练数据
    X_train = np.random.randn(1000, 5)
    y_train = X_train[:, 0] + 0.5 * X_train[:, 1] + np.random.randn(1000) * 0.1
    
    # 创建自适应模型框架
    base_model = LinearRegression()
    drift_detector = StatisticalDriftDetector(threshold=0.01)
    adaptive_model = AdaptiveModelFramework(
        base_model=base_model,
        drift_detector=drift_detector,
        adaptation_strategy='retrain'
    )
    
    # 训练基础模型
    adaptive_model.fit(X_train, y_train)
    print("基础模型训练完成")
    
    # 模拟数据流和概念漂移
    for step in range(10):
        # 生成新数据（逐渐引入概念漂移）
        shift = step * 0.1
        X_new = np.random.randn(100, 5) + shift
        y_new = X_new[:, 0] + 0.5 * X_new[:, 1] + shift + np.random.randn(100) * 0.1
        
        # 验证数据
        X_val = np.random.randn(50, 5) + shift
        y_val = X_val[:, 0] + 0.5 * X_val[:, 1] + shift + np.random.randn(50) * 0.1
        
        # 自适应更新
        adaptive_model.adapt_model(X_new, y_new, X_val, y_val)
        
        # 预测性能
        predictions = adaptive_model.predict(X_val)
        mse = mean_squared_error(y_val, predictions)
        
        print(f"步骤 {step + 1}: MSE = {mse:.4f}")
    
    # 显示漂移统计
    drift_stats = adaptive_model.get_drift_statistics()
    print(f"\n漂移统计:")
    print(f"总检测次数: {drift_stats['total_checks']}")
    print(f"检测到漂移次数: {drift_stats['drift_detections']}")
    print(f"漂移率: {drift_stats['drift_rate']:.2%}")

def demo_meta_learning():
    """演示元学习"""
    print("\n=== 元学习演示 ===")
    
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.naive_bayes import GaussianNB
    
    # 基学习器
    base_learners = {
        'linear': LinearRegression(),
        'ridge': Ridge(),
        'rf': RandomForestRegressor(n_estimators=10, random_state=42),
        'tree': DecisionTreeRegressor(random_state=42)
    }
    
    # 元学习器
    meta_learner = GaussianNB()
    
    # 创建元学习框架
    meta_framework = MetaLearningFramework(base_learners, meta_learner)
    
    # 模拟多个学习任务
    np.random.seed(42)
    
    for task in range(10):
        print(f"\n任务 {task + 1}:")
        
        # 生成不同特性的数据
        n_samples = np.random.randint(100, 500)
        n_features = np.random.randint(2, 8)
        noise_level = np.random.uniform(0.1, 0.5)
        
        X = np.random.randn(n_samples, n_features)
        
        # 不同的数据生成机制
        if task % 3 == 0:
            # 线性关系
            y = np.sum(X, axis=1) + np.random.randn(n_samples) * noise_level
        elif task % 3 == 1:
            # 非线性关系
            y = np.sum(X**2, axis=1) + np.random.randn(n_samples) * noise_level
        else:
            # 复杂关系
            y = np.sum(X[:, :2], axis=1) * np.sum(X[:, 2:], axis=1) + np.random.randn(n_samples) * noise_level
        
        # 分割数据
        split = n_samples // 2
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        
        # 元学习
        best_learner, best_performance = meta_framework.learn_task(X_train, y_train, X_val, y_val)
        
        print(f"  数据: {n_samples}样本, {n_features}特征, 噪声={noise_level:.3f}")
        print(f"  最佳学习器: {best_learner}")
        print(f"  最佳性能: MSE = {best_performance:.4f}")
        
        # 如果有足够的历史任务，测试元学习器预测
        if len(meta_framework.task_history) >= 5:
            predicted_learner = meta_framework.predict_best_learner(X_train, y_train)
            print(f"  元学习器预测: {predicted_learner}")

def demo_market_regime_detection():
    """演示市场状态检测"""
    print("\n=== 市场状态检测演示 ===")
    
    # 生成模拟市场数据
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=252, freq='D')
    
    # 基础价格趋势
    base_trend = np.cumsum(np.random.randn(252) * 0.01)
    
    # 添加不同的市场状态
    price_data = pd.Series(index=dates, dtype=float)
    volume_data = pd.Series(index=dates, dtype=float)
    
    for i, date in enumerate(dates):
        if i < 60:  # 正常期
            price = 100 + base_trend[i] + np.random.randn() * 0.5
            volume = 1000 + np.random.randn() * 100
        elif i < 120:  # 高波动期
            price = 100 + base_trend[i] + np.random.randn() * 2.0
            volume = 1200 + np.random.randn() * 200
        elif i < 180:  # 趋势期
            price = 100 + base_trend[i] + (i - 120) * 0.1 + np.random.randn() * 0.3
            volume = 800 + np.random.randn() * 80
        else:  # 高成交量期
            price = 100 + base_trend[i] + np.random.randn() * 0.8
            volume = 2000 + np.random.randn() * 300
            
        price_data[date] = price
        volume_data[date] = volume
    
    # 创建市场状态检测器
    regime_detector = MarketRegimeDetector(['volatility', 'trend', 'volume'])
    
    # 检测状态变化
    print("检测市场状态变化:")
    for i in range(50, len(dates), 20):  # 每20天检测一次
        end_idx = min(i + 1, len(dates))
        price_window = price_data.iloc[:end_idx]
        volume_window = volume_data.iloc[:end_idx]
        
        regime, indicators = regime_detector.detect_regime(price_window, volume_window)
        
        print(f"日期 {dates[i].strftime('%Y-%m-%d')}: {regime}")
        print(f"  波动率: {indicators.get('volatility', 0):.4f}")
        print(f"  趋势: {indicators.get('trend', 0):.4f}")
        print(f"  成交量: {indicators.get('volume', 0):.4f}")
    
    # 显示状态统计
    regime_stats = regime_detector.get_regime_statistics()
    print(f"\n市场状态统计:")
    print(f"当前状态: {regime_stats['current_regime']}")
    print(f"状态变化次数: {regime_stats['regime_changes']}")
    print(f"状态分布: {regime_stats['regime_distribution']}")

def main():
    """主函数"""
    print("=== 市场动态适应高级演示 ===")
    
    # 运行各个演示
    demo_concept_drift_detection()
    demo_adaptive_model()
    demo_meta_learning()
    demo_market_regime_detection()
    
    print("\n=== 演示总结 ===")
    print("1. 概念漂移检测: 实现了统计检验、性能监控、异常检测多种方法")
    print("2. 自适应模型: 展示了模型如何响应概念漂移并自动调整")
    print("3. 元学习: 演示了如何基于任务特征选择最佳学习算法")
    print("4. 市场状态检测: 展示了市场状态识别和分析方法")
    
    print("\n=== 实用建议 ===")
    print("1. 在生产环境中结合多种漂移检测方法以提高准确性")
    print("2. 根据业务需求选择合适的模型适应策略")
    print("3. 建立完整的监控体系来跟踪模型性能和市场变化")
    print("4. 定期评估和更新漂移检测阈值")
    print("5. 保持足够的历史数据用于对比分析")
    
    print("\nmarket_dynamics_advanced_demo.py 演示完成！")
    print("\n主要演示内容:")
    print("1. 多种概念漂移检测算法")
    print("2. 自适应模型框架和策略")
    print("3. 元学习算法选择框架")
    print("4. 市场状态检测和分析")
    print("5. 完整的市场动态适应解决方案")

if __name__ == "__main__":
    main()