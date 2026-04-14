#!/usr/bin/env python3
"""
概念漂移检测演示
基于第8章市场动态适应内容
"""

import qlib
import numpy as np
import pandas as pd
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.data.handler import Alpha158
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 初始化Qlib
print("初始化Qlib...")
qlib.init(mount_path="~/.qlib/qlib_data/cn_data", region="cn")

class ConceptDriftDetector:
    """概念漂移检测器"""
    
    def __init__(self, window_size=252, threshold=0.05):
        self.window_size = window_size
        self.threshold = threshold
        self.performance_history = []
        self.drift_points = []
    
    def add_performance(self, performance):
        """添加性能数据"""
        self.performance_history.append(performance)
        
        if len(self.performance_history) > self.window_size:
            self.performance_history.pop(0)
    
    def detect_drift(self):
        """检测概念漂移"""
        if len(self.performance_history) < self.window_size:
            return False
        
        # 计算最近性能与历史平均的差异
        recent_performance = np.mean(self.performance_history[-self.window_size//4:])
        historical_performance = np.mean(self.performance_history[:-self.window_size//4])
        
        performance_drop = historical_performance - recent_performance
        
        if performance_drop > self.threshold:
            self.drift_points.append(len(self.performance_history))
            return True
        
        return False
    
    def get_drift_statistics(self):
        """获取漂移统计信息"""
        return {
            'drift_count': len(self.drift_points),
            'drift_points': self.drift_points,
            'avg_performance': np.mean(self.performance_history) if self.performance_history else 0
        }

class AdaptiveModel:
    """自适应模型"""
    
    def __init__(self, base_model_class=None, retrain_frequency=63):
        self.base_model_class = base_model_class or GradientBoostingRegressor
        self.retrain_frequency = retrain_frequency
        self.model = None
        self.scaler = StandardScaler()
        self.steps_since_retrain = 0
        self.training_data = []
        self.training_labels = []
        self.is_fitted = False
    
    def fit(self, X, y):
        """初始训练"""
        # 处理NaN值
        X_clean = X.fillna(X.mean()).fillna(0)
        y_clean = y.fillna(y.mean()).fillna(0)
        
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X_clean)
        
        # 创建模型
        if self.base_model_class == GradientBoostingRegressor:
            self.model = self.base_model_class(
                n_estimators=50,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        else:
            self.model = self.base_model_class()
        
        # 训练模型
        self.model.fit(X_scaled, y_clean)
        
        # 保存训练数据
        self.training_data = [X_clean]
        self.training_labels = [y_clean]
        self.is_fitted = True
    
    def predict(self, X):
        """预测"""
        if not self.is_fitted:
            raise ValueError("模型未训练")
        
        # 处理NaN值
        X_clean = X.fillna(0)  # 对于新数据，用0填充
        
        # 标准化
        X_scaled = self.scaler.transform(X_clean)
        
        return self.model.predict(X_scaled)
    
    def update(self, X, y):
        """更新模型"""
        self.steps_since_retrain += 1
        
        # 处理NaN值
        X_clean = X.fillna(0)
        y_clean = y.fillna(y.mean()).fillna(0)
        
        # 添加新数据
        self.training_data.append(X_clean)
        self.training_labels.append(y_clean)
        
        # 保持数据窗口大小
        if len(self.training_data) > 5:
            self.training_data.pop(0)
            self.training_labels.pop(0)
        
        # 检查是否需要重训练
        if self.steps_since_retrain >= self.retrain_frequency:
            self.retrain()
            self.steps_since_retrain = 0
    
    def retrain(self):
        """重新训练模型"""
        if len(self.training_data) > 0:
            # 合并所有训练数据
            combined_X = pd.concat(self.training_data, ignore_index=True)
            combined_y = pd.concat(self.training_labels, ignore_index=True)
            
            # 重新标准化
            X_scaled = self.scaler.fit_transform(combined_X)
            
            # 重新训练
            if self.base_model_class == GradientBoostingRegressor:
                self.model = self.base_model_class(
                    n_estimators=50,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                )
            else:
                self.model = self.base_model_class()
            
            self.model.fit(X_scaled, combined_y)
            print(f"模型已重新训练（数据点数量: {len(combined_y)}）")

def calculate_ic(predictions, returns):
    """计算IC值"""
    valid_mask = ~(np.isnan(predictions) | np.isnan(returns))
    pred_clean = predictions[valid_mask]
    ret_clean = returns[valid_mask]
    
    if len(pred_clean) <= 1:
        return 0
    
    correlation = np.corrcoef(pred_clean, ret_clean)[0, 1]
    return correlation if not np.isnan(correlation) else 0

def simulate_market_regime_change(data, change_point=0.5):
    """模拟市场机制变化"""
    change_idx = int(len(data) * change_point)
    
    # 前半部分保持原样
    regime1_data = data[:change_idx].copy()
    
    # 后半部分添加噪声模拟市场变化
    regime2_data = data[change_idx:].copy()
    noise = np.random.normal(0, 0.1, regime2_data.shape)
    regime2_data = regime2_data + noise
    
    combined_data = pd.concat([regime1_data, regime2_data])
    return combined_data, change_idx

def main():
    print("准备数据...")
    
    # 快速演示版本：使用模拟数据
    print("使用模拟数据进行快速演示...")
    
    # 生成模拟的金融时间序列数据
    np.random.seed(42)
    n_samples = 1000
    n_features = 50
    
    # 模拟特征数据
    feature_data = np.random.randn(n_samples, n_features)
    
    # 模拟标签（收益率）
    labels = np.random.normal(0, 0.01, n_samples)
    
    # 创建DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    train_data = pd.DataFrame(feature_data[:600], columns=feature_names)
    train_data['LABEL0'] = labels[:600]
    
    test_data = pd.DataFrame(feature_data[600:], columns=feature_names)
    test_data['LABEL0'] = labels[600:]
    
    # 分离特征和标签
    label_cols = [col for col in train_data.columns if 'LABEL' in str(col)]
    feature_cols = [col for col in train_data.columns if 'LABEL' not in str(col)]
    
    X_train = train_data[feature_cols]
    y_train = train_data[label_cols[0]] if label_cols else train_data.iloc[:, -1]  # Use last column if no LABEL found
    X_test = test_data[feature_cols]
    y_test = test_data[label_cols[0]] if label_cols else test_data.iloc[:, -1]
    
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    
    print("\n=== 模拟市场机制变化 ===")
    
    # 模拟市场机制变化
    X_test_changed, change_point = simulate_market_regime_change(X_test, change_point=0.6)
    print(f"市场机制变化点: {change_point}")
    
    print("\n=== 概念漂移检测演示 ===")
    
    # 创建概念漂移检测器
    drift_detector = ConceptDriftDetector(window_size=50, threshold=0.02)
    
    # 训练基础模型
    print("训练基础模型...")
    base_model = GradientBoostingRegressor(
        n_estimators=50,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    # 处理NaN值
    X_train_clean = X_train.fillna(X_train.mean()).fillna(0)
    y_train_clean = y_train.fillna(y_train.mean()).fillna(0)
    
    base_model.fit(X_train_clean, y_train_clean)
    
    # 创建自适应模型
    print("创建自适应模型...")
    adaptive_model = AdaptiveModel(GradientBoostingRegressor, retrain_frequency=30)
    adaptive_model.fit(X_train, y_train)
    
    print("\n=== 逐步预测和漂移检测 ===")
    
    # 分批处理测试数据
    batch_size = 20
    base_model_performance = []
    adaptive_model_performance = []
    drift_detected = []
    
    for i in range(0, len(X_test_changed), batch_size):
        end_idx = min(i + batch_size, len(X_test_changed))
        X_batch = X_test_changed.iloc[i:end_idx]
        y_batch = y_test.iloc[i:end_idx]
        
        # 基础模型预测
        X_batch_clean = X_batch.fillna(0)
        base_pred = base_model.predict(X_batch_clean)
        base_ic = calculate_ic(base_pred, y_batch.values)
        base_model_performance.append(base_ic)
        
        # 自适应模型预测
        adaptive_pred = adaptive_model.predict(X_batch)
        adaptive_ic = calculate_ic(adaptive_pred, y_batch.values)
        adaptive_model_performance.append(adaptive_ic)
        
        # 概念漂移检测
        drift_detector.add_performance(base_ic)
        is_drift = drift_detector.detect_drift()
        drift_detected.append(is_drift)
        
        # 更新自适应模型（模拟在线学习）
        adaptive_model.update(X_batch, y_batch)
        
        if is_drift:
            print(f"批次 {i//batch_size + 1}: 检测到概念漂移！")
        
        if (i//batch_size + 1) % 5 == 0:
            print(f"批次 {i//batch_size + 1}: 基础模型IC={base_ic:.4f}, 自适应模型IC={adaptive_ic:.4f}")
    
    print("\n=== 性能对比分析 ===")
    
    # 计算整体性能
    base_avg_ic = np.mean(base_model_performance)
    adaptive_avg_ic = np.mean(adaptive_model_performance)
    
    print(f"基础模型平均IC: {base_avg_ic:.4f}")
    print(f"自适应模型平均IC: {adaptive_avg_ic:.4f}")
    print(f"性能提升: {(adaptive_avg_ic - base_avg_ic) / abs(base_avg_ic) * 100:.2f}%")
    
    # 漂移统计
    drift_stats = drift_detector.get_drift_statistics()
    print(f"检测到的漂移次数: {drift_stats['drift_count']}")
    print(f"漂移点: {drift_stats['drift_points']}")
    
    print("\n=== 分段性能分析 ===")
    
    # 分析变化前后的性能
    mid_point = len(base_model_performance) // 2
    
    base_early_ic = np.mean(base_model_performance[:mid_point])
    base_late_ic = np.mean(base_model_performance[mid_point:])
    adaptive_early_ic = np.mean(adaptive_model_performance[:mid_point])
    adaptive_late_ic = np.mean(adaptive_model_performance[mid_point:])
    
    print("前半段性能:")
    print(f"  基础模型IC: {base_early_ic:.4f}")
    print(f"  自适应模型IC: {adaptive_early_ic:.4f}")
    
    print("后半段性能:")
    print(f"  基础模型IC: {base_late_ic:.4f}")
    print(f"  自适应模型IC: {adaptive_late_ic:.4f}")
    
    print("性能变化:")
    print(f"  基础模型变化: {(base_late_ic - base_early_ic):.4f}")
    print(f"  自适应模型变化: {(adaptive_late_ic - adaptive_early_ic):.4f}")
    
    print("\n=== 元学习方法演示 ===")
    
    class MetaLearningAdapter:
        """元学习适应器"""
        
        def __init__(self):
            self.adaptation_history = []
            self.performance_history = []
        
        def adapt_hyperparameters(self, current_performance, base_params):
            """基于历史表现调整超参数"""
            adapted_params = base_params.copy()
            
            if len(self.performance_history) > 5:
                # 如果性能下降，增加学习率
                recent_perf = np.mean(self.performance_history[-3:])
                historical_perf = np.mean(self.performance_history[:-3])
                
                if recent_perf < historical_perf:
                    adapted_params['learning_rate'] = min(0.3, base_params['learning_rate'] * 1.5)
                    adapted_params['n_estimators'] = max(30, base_params['n_estimators'] - 20)
                else:
                    adapted_params['learning_rate'] = max(0.05, base_params['learning_rate'] * 0.9)
            
            self.adaptation_history.append(adapted_params)
            self.performance_history.append(current_performance)
            
            return adapted_params
    
    # 测试元学习适应器
    meta_adapter = MetaLearningAdapter()
    base_params = {'learning_rate': 0.1, 'n_estimators': 50, 'max_depth': 6}
    
    print("元学习超参数适应演示:")
    for i, perf in enumerate(base_model_performance[-5:]):
        adapted_params = meta_adapter.adapt_hyperparameters(perf, base_params)
        print(f"步骤 {i+1}: 性能={perf:.4f}, 学习率={adapted_params['learning_rate']:.3f}")
    
    print("\n概念漂移检测演示完成！")

if __name__ == "__main__":
    main()