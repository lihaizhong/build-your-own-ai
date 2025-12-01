# Prophet v8 重构方案设计

## 🎯 v8重构目标

### 问题诊断
基于v8首次运行结果，发现以下关键问题：

**性能恶化**:
- 申购MAPE: 42.64% → 53.91% (恶化11.27%)
- 赎回MAPE: 99.43% → 110.57% (恶化11.14%)
- 净流入: 预测异常负值(-4.58亿)

**技术问题**:
1. **过拟合**: 122维特征 vs 427样本，特征/样本比过高
2. **参数过度保守**: changepoint=0.001，趋势变化检测不足
3. **特征质量**: 包含噪声和冗余特征
4. **未来预测**: 30天特征填充策略不准确

## 🚀 重构策略

### 策略一：精简特征工程 (50维)

#### 核心原则
- **特征数量**: 从122维减至50维左右
- **特征质量**: 优先选择业务相关性强、预测能力强的特征
- **避免过拟合**: 特征/样本比控制在合理范围

#### 核心特征选择 (50维)

**1. 基础时间特征 (15维)**
```python
time_features = [
    'year', 'month', 'day', 'weekday', 'week_of_year', 'day_of_year',
    'quarter', 'is_quarter_start', 'is_quarter_end',
    'is_month_start', 'is_month_mid', 'is_month_end',
    'is_weekend', 'is_friday', 'is_monday'
]
```

**2. 业务洞察特征 (10维)**
```python
business_features = [
    'pay_cycle',                    # 薪资周期
    'pay_preparation',              # 薪资准备期
    'investment_cycle',             # 投资周期
    'month_end_fund',               # 月末资金调度
    'month_start_fund',             # 月初资金调度
    'quarter_end_fund',             # 季度末资金调度
    'is_business_day',              # 业务日期
    'is_month_end_business',        # 月末业务日
    'weekend_pay_cycle',            # 周末薪资周期
    'month_start_business'          # 月初业务周期
]
```

**3. 市场数据特征 (12维)**
```python
market_features = [
    'shibor_o_n',                   # 隔夜利率
    'shibor_1w',                    # 1周利率
    'shibor_1m',                    # 1月利率
    'shibor_o_n_change',            # 隔夜利率变化
    'shibor_1w_change',             # 1周利率变化
    'shibor_1m_change',             # 1月利率变化
    'daily_yield',                  # 日收益率
    'yield_7d',                     # 7日年化收益率
    'yield_change',                 # 收益率变化
    'rate_environment',             # 利率环境
    'yield_environment',            # 收益率环境
    'rate_spread_1w_1m'             # 利差特征
]
```

**4. 滞后窗口特征 (10维 - 申购)**
```python
purchase_lag_features = [
    'purchase_lag_1',               # 1天滞后
    'purchase_lag_2',               # 2天滞后
    'purchase_lag_3',               # 3天滞后
    'purchase_rolling_mean_7',      # 7天均值
    'purchase_rolling_mean_14',     # 14天均值
    'purchase_rolling_std_7',       # 7天标准差
    'purchase_rolling_min_7',       # 7天最小值
    'purchase_rolling_max_7',       # 7天最大值
    'purchase_pct_change_7',        # 7天变化率
    'purchase_pct_change_14'        # 14天变化率
]
```

**5. 滞后窗口特征 (10维 - 赎回)**
```python
redeem_lag_features = [
    'redeem_lag_1',                 # 1天滞后
    'redeem_lag_2',                 # 2天滞后
    'redeem_lag_3',                 # 3天滞后
    'redeem_rolling_mean_7',        # 7天均值
    'redeem_rolling_mean_14',       # 14天均值
    'redeem_rolling_std_7',         # 7天标准差
    'redeem_rolling_min_7',         # 7天最小值
    'redeem_rolling_max_7',         # 7天最大值
    'redeem_pct_change_7',          # 7天变化率
    'redeem_pct_change_14'          # 14天变化率
]
```

**6. 交互特征 (3维)**
```python
interaction_features = [
    'weekend_pay_cycle',            # 周末×薪资周期
    'rate_environment_weekday',     # 利率环境×工作日
    'yield_month_end'               # 收益率环境×月末
]
```

**总计**: 15 + 10 + 12 + 10 + 10 + 3 = **60维特征** (相比122维减少51%)

### 策略二：智能参数设置

#### 基于经验的参数设置
```python
# 申购模型参数 - 平衡趋势和季节性
purchase_params = {
    'changepoint_prior_scale': 0.01,      # 适度趋势检测
    'seasonality_prior_scale': 5.0,       # 中等季节性强度
    'holidays_prior_scale': 10.0,         # 强节假日效应
    'interval_width': 0.90,               # 标准置信区间
    'seasonality_mode': 'additive'        # 加性季节性
}

# 赎回模型参数 - 重视季节性变化
redeem_params = {
    'changepoint_prior_scale': 0.02,      # 较强趋势检测
    'seasonality_prior_scale': 8.0,       # 强季节性强度
    'holidays_prior_scale': 10.0,         # 强节假日效应
    'interval_width': 0.95,               # 较宽置信区间
    'seasonality_mode': 'additive'        # 加性季节性
}
```

#### 简化网格搜索
```python
# 减少搜索空间，只测试关键参数组合
param_grid = {
    'changepoint_prior_scale': [0.005, 0.01, 0.02, 0.03],
    'seasonality_prior_scale': [2.0, 5.0, 8.0, 10.0],
    'seasonality_mode': ['additive', 'multiplicative']
}
```

### 策略三：改进未来特征预测

#### 精准特征预测
```python
def predict_future_features(df, selected_features, future_dates):
    """
    改进的30天特征预测策略
    """
    future_features = {}
    
    # 1. 时间特征 (精确计算)
    time_features = ['year', 'month', 'day', 'weekday', 'week_of_year', 'day_of_year', 
                    'quarter', 'is_quarter_start', 'is_quarter_end', 'is_month_start', 
                    'is_month_mid', 'is_month_end', 'is_weekend', 'is_friday', 'is_monday']
    
    for feature in time_features:
        if feature in selected_features:
            if feature == 'year':
                future_features[feature] = future_dates.dt.year
            elif feature == 'month':
                future_features[feature] = future_dates.dt.month
            elif feature == 'day':
                future_features[feature] = future_dates.dt.day
            elif feature == 'weekday':
                future_features[feature] = future_dates.dt.dayofweek
            elif feature == 'week_of_year':
                future_features[feature] = future_dates.dt.isocalendar().week
            elif feature == 'day_of_year':
                future_features[feature] = future_dates.dt.dayofyear
            elif feature == 'quarter':
                future_features[feature] = future_dates.dt.quarter
            elif feature == 'is_quarter_start':
                future_features[feature] = future_dates.dt.is_quarter_start.astype(int)
            elif feature == 'is_quarter_end':
                future_features[feature] = future_dates.dt.is_quarter_end.astype(int)
            elif feature == 'is_month_start':
                future_features[feature] = (future_dates.dt.day <= 3).astype(int)
            elif feature == 'is_month_mid':
                future_features[feature] = ((future_dates.dt.day >= 14) & (future_dates.dt.day <= 16)).astype(int)
            elif feature == 'is_month_end':
                future_features[feature] = (future_dates.dt.day >= 28).astype(int)
            elif feature == 'is_weekend':
                future_features[feature] = (future_dates.dt.dayofweek >= 5).astype(int)
            elif feature == 'is_friday':
                future_features[feature] = (future_dates.dt.dayofweek == 4).astype(int)
            elif feature == 'is_monday':
                future_features[feature] = (future_dates.dt.dayofweek == 0).astype(int)
    
    # 2. 业务特征 (基于时间特征计算)
    business_derived_features = ['pay_cycle', 'pay_preparation', 'investment_cycle', 
                                'month_end_fund', 'month_start_fund', 'quarter_end_fund',
                                'is_business_day', 'is_month_end_business']
    
    for feature in business_derived_features:
        if feature in selected_features:
            # 基于已经计算的时间特征来推导
            if feature == 'pay_cycle':
                future_features[feature] = ((future_dates.dt.day >= 25) | (future_dates.dt.day <= 5)).astype(int)
            elif feature == 'pay_preparation':
                future_features[feature] = ((future_dates.dt.day >= 20) & (future_dates.dt.day <= 24)).astype(int)
            elif feature == 'investment_cycle':
                future_features[feature] = (future_dates.dt.day.isin([1, 15])).astype(int)
            elif feature == 'month_end_fund':
                future_features[feature] = ((future_dates.dt.day >= 25) & (future_dates.dt.day <= 31)).astype(int)
            elif feature == 'month_start_fund':
                future_features[feature] = (future_dates.dt.day <= 7).astype(int)
            elif feature == 'quarter_end_fund':
                future_features[feature] = ((future_dates.dt.month.isin([3, 6, 9, 12])) & (future_dates.dt.day >= 25)).astype(int)
            elif feature == 'is_business_day':
                future_features[feature] = (~future_dates.dt.dayofweek.isin([5, 6])).astype(int)
            elif feature == 'is_month_end_business':
                future_features[feature] = future_features['is_business_day'] * future_features['month_end_fund']
    
    # 3. 滞后特征 (趋势外推)
    lag_features = [col for col in selected_features if '_lag_' in col or '_rolling_' in col or '_pct_change_' in col]
    for feature in lag_features:
        if feature in df.columns:
            # 使用最近7天数据进行简单外推
            recent_values = df[feature].dropna().tail(7)
            if len(recent_values) >= 3:
                # 使用加权平均（最近值权重更高）
                weights = np.array([0.1, 0.15, 0.2, 0.25, 0.15, 0.1, 0.05])
                forecast_value = np.average(recent_values.values, weights=weights)
                future_features[feature] = forecast_value
            else:
                future_features[feature] = recent_values.mean()
    
    # 4. 市场特征 (使用最后已知值)
    market_features = ['shibor_o_n', 'shibor_1w', 'shibor_1m', 'daily_yield', 'yield_7d', 
                      'rate_environment', 'yield_environment']
    for feature in market_features:
        if feature in selected_features and feature in df.columns:
            # 使用最后已知值作为未来预测
            future_features[feature] = df[feature].iloc[-1]
    
    return pd.DataFrame(future_features)
```

### 策略四：增加模型验证

#### 交叉验证
```python
def cross_validation_evaluation(enhanced_df, regressors, target_column, cv=5):
    """
    交叉验证评估模型稳定性
    """
    from sklearn.model_selection import TimeSeriesSplit
    
    prophet_df = enhanced_df[['ds', target_column]].copy()
    prophet_df.rename(columns={target_column: 'y'}, inplace=True)
    
    # 添加外生变量
    for regressor in regressors:
        prophet_df[regressor] = enhanced_df[regressor].fillna(0)
    
    # 时间序列交叉验证
    tscv = TimeSeriesSplit(n_splits=cv)
    scores = []
    
    for train_idx, test_idx in tscv.split(prophet_df):
        train_data = prophet_df.iloc[train_idx]
        test_data = prophet_df.iloc[test_idx]
        
        # 训练模型
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.01,
            seasonality_prior_scale=5.0,
            holidays_prior_scale=10.0,
            interval_width=0.90,
            seasonality_mode='additive',
            holidays=create_optimized_holidays()
        )
        
        model.fit(train_data)
        
        # 预测
        forecast = model.predict(test_data)
        
        # 计算MAE
        mae = mean_absolute_error(test_data['y'], forecast['yhat'])
        scores.append(mae)
    
    return np.mean(scores), np.std(scores)
```

## 📋 重构实施计划

### Phase 1: 特征工程重构 (核心)
- [ ] **重新设计60维特征**: 精简而高质量的特征组合
- [ ] **特征相关性检查**: 移除高度相关的特征
- [ ] **业务逻辑验证**: 确保特征的业务合理性

### Phase 2: 参数优化升级
- [ ] **智能参数设置**: 基于经验的平衡参数
- [ ] **简化网格搜索**: 16种组合的精准搜索
- [ ] **交叉验证**: 5折时间序列交叉验证

### Phase 3: 预测策略改进
- [ ] **精准特征预测**: 改进30天特征预测策略
- [ ] **趋势外推优化**: 更合理的滞后特征预测
- [ ] **市场特征处理**: 合理的利率和收益率预测

### Phase 4: 模型验证与优化
- [ ] **性能基准对比**: 与v7性能进行详细对比
- [ ] **稳定性测试**: 不同参数配置的稳定性
- [ ] **业务逻辑检查**: 确保预测结果的合理性

## 🎯 预期改进效果

### 技术指标提升
- **特征数量**: 122维 → 60维 (减少51%)
- **申购MAPE**: 53.91% → 40.0% (改善13.91%)
- **赎回MAPE**: 110.57% → 92.0% (改善18.57%)
- **模型稳定性**: 交叉验证分数方差 < 5%

### 业务指标改善
- **净流入预测**: 恢复正常正值范围
- **预测逻辑**: 符合资金流动的业务规律
- **线上分数**: 103分 → 108-112分

## 🔧 核心技术要点

### 1. 特征工程策略
- **精选特征**: 从122维减至60维，保留最有价值的特征
- **避免冗余**: 移除高度相关的特征
- **业务导向**: 确保特征有明确的业务意义

### 2. 参数优化策略
- **平衡设置**: 避免过度保守或激进的参数
- **验证驱动**: 通过交叉验证选择最优参数
- **稳定性优先**: 重视模型稳定性而非单次最优

### 3. 预测优化策略
- **精准预测**: 基于业务逻辑的30天特征预测
- **趋势外推**: 更合理的滞后特征处理
- **市场建模**: 合理的利率和收益率特征预测

### 4. 验证策略
- **多重验证**: 时间序列交叉验证
- **稳定性测试**: 确保模型在不同参数下的稳定性
- **业务检查**: 确保预测结果的业务合理性

---

**重构价值**: 通过精简特征、智能参数、精准预测，重构Prophet v8实现从过拟合到优化的技术突破！

*制定时间: 2025年12月2日*
*实施目标: 申购MAPE < 40%, 赎回MAPE < 92%, 分数 > 108分*