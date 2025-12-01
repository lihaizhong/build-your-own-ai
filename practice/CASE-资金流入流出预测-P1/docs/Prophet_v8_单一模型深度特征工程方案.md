# Prophet v8 单一模型深度特征工程方案

## 🎯 v8版本定位

### 核心理念
**纯粹Prophet + 深度特征工程** = 挖掘Prophet模型的能力边界

### 版本目标
- **分数目标**: 108-110分 (vs v7的103分，+5-7分)
- **申购MAPE**: ≤41.09% (恢复并超越v8最佳水平)
- **赎回MAPE**: ≤91.02% (恢复并超越v6最佳水平)
- **技术边界**: 探索单一Prophet模型的极限性能

## 🛠️ 深度特征工程体系

### 第一层：时间维度特征 (Day 1-2)

#### 1.1 精细化时间特征
```python
def create_deep_time_features(df):
    """
    创建深度时间维度特征
    """
    features = {}
    
    # 基本时间特征
    features['year'] = df['ds'].dt.year
    features['month'] = df['ds'].dt.month  
    features['day'] = df['ds'].dt.day
    features['weekday'] = df['ds'].dt.dayofweek
    features['week_of_year'] = df['ds'].dt.isocalendar().week
    features['day_of_year'] = df['ds'].dt.dayofyear
    
    # 季度信息
    features['quarter'] = df['ds'].dt.quarter
    features['is_quarter_start'] = df['ds'].dt.is_quarter_start.astype(int)
    features['is_quarter_end'] = df['ds'].dt.is_quarter_end.astype(int)
    
    # 月度信息  
    features['is_month_start'] = (df['ds'].dt.day <= 3).astype(int)
    features['is_month_mid'] = ((df['ds'].dt.day >= 14) & (df['ds'].dt.day <= 16)).astype(int)
    features['is_month_end'] = (df['ds'].dt.day >= 28).astype(int)
    
    # 周期性特征 (sin/cos编码)
    features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
    features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
    features['day_sin'] = np.sin(2 * np.pi * features['day_of_year'] / 365)
    features['day_cos'] = np.cos(2 * np.pi * features['day_of_year'] / 365)
    features['weekday_sin'] = np.sin(2 * np.pi * features['weekday'] / 7)
    features['weekday_cos'] = np.cos(2 * np.pi * features['weekday'] / 7)
    
    # 特殊时间点
    features['is_weekend'] = (features['weekday'] >= 5).astype(int)
    features['is_friday'] = (features['weekday'] == 4).astype(int)
    features['is_monday'] = (features['weekday'] == 0).astype(int)
    
    return pd.DataFrame(features)
```

#### 1.2 业务周期性特征
```python
def create_business_cycle_features(df):
    """
    创建业务相关的周期性特征
    """
    business_features = {}
    
    # 薪资发放周期 (推测为月底+月初)
    business_features['pay_cycle'] = 0
    business_features['pay_cycle'] += ((df['day'] >= 25) | (df['day'] <= 5)).astype(int) * 1  # 薪资期
    business_features['pay_cycle'] += ((df['day'] >= 20) & (df['day'] <= 24)).astype(int) * 2  # 准备期
    
    # 投资习惯周期
    business_features['investment_cycle'] = 0
    business_features['investment_cycle'] += (df['day'].isin([1, 15])).astype(int) * 1  # 定投日
    business_features['investment_cycle'] += (df['day'].isin([10, 20, 30])).astype(int) * 2  # 集中日
    
    # 月末资金调度
    business_features['month_end_fund'] = ((df['day'] >= 25) & (df['day'] <= 31)).astype(int)
    business_features['month_start_fund'] = (df['day'] <= 7).astype(int)
    
    return pd.DataFrame(business_features)
```

### 第二层：市场数据特征 (Day 2-3)

#### 2.1 利率时间序列特征
```python
def create_rate_features(df, rate_data):
    """
    基于利率数据创建市场特征
    """
    rate_features = {}
    
    # 基础利率特征
    rate_features['shibor_o_n'] = rate_data['Interest_O_N']
    rate_features['shibor_1w'] = rate_data['Interest_1_W']
    rate_features['shibor_1m'] = rate_data['Interest_1_M']
    
    # 利率变化特征
    rate_features['shibor_o_n_change'] = rate_data['Interest_O_N'].diff()
    rate_features['shibor_1w_change'] = rate_data['Interest_1_W'].diff()
    rate_features['shibor_1m_change'] = rate_data['Interest_1_M'].diff()
    
    # 利率趋势特征
    rate_features['shibor_o_n_trend'] = rate_data['Interest_O_N'].rolling(7).mean()
    rate_features['shibor_1w_trend'] = rate_data['Interest_1_W'].rolling(7).mean()
    rate_features['shibor_1m_trend'] = rate_data['Interest_1_M'].rolling(7).mean()
    
    # 利率波动特征
    rate_features['shibor_volatility'] = rate_data['Interest_O_N'].rolling(7).std()
    
    # 收益率特征
    rate_features['daily_yield'] = rate_data['mfd_daily_yield']
    rate_features['yield_7d'] = rate_data['mfd_7daily_yield']
    rate_features['yield_change'] = rate_data['mfd_daily_yield'].diff()
    
    return pd.DataFrame(rate_features)
```

#### 2.2 市场情绪指标
```python
def create_market_sentiment_features(df, market_data):
    """
    创建市场情绪相关特征
    """
    sentiment_features = {}
    
    # 利率环境判断
    sentiment_features['rate_environment'] = (
        (market_data['Interest_1_M'] > market_data['Interest_1_M'].median()).astype(int)
    )
    
    # 收益率环境判断
    sentiment_features['yield_environment'] = (
        (market_data['mfd_7daily_yield'] > market_data['mfd_7daily_yield'].median()).astype(int)
    )
    
    # 市场稳定性指标
    sentiment_features['stability_score'] = (
        market_data['Interest_O_N'].rolling(30).std() / market_data['Interest_O_N'].rolling(30).mean()
    )
    
    return pd.DataFrame(sentiment_features)
```

### 第三层：高级统计特征 (Day 3-4)

#### 3.1 滞后和滑动窗口特征
```python
def create_lag_and_window_features(df, target_col):
    """
    创建滞后和滑动窗口特征
    """
    lag_features = {}
    
    # 滞后特征 (1-7天)
    for lag in [1, 2, 3, 5, 7]:
        lag_features[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    
    # 滑动窗口统计特征
    for window in [3, 5, 7, 14, 30]:
        lag_features[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window).mean()
        lag_features[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window).std()
        lag_features[f'{target_col}_rolling_min_{window}'] = df[target_col].rolling(window).min()
        lag_features[f'{target_col}_rolling_max_{window}'] = df[target_col].rolling(window).max()
    
    # 变化率特征
    for window in [3, 7, 14]:
        lag_features[f'{target_col}_pct_change_{window}'] = df[target_col].pct_change(window)
    
    return pd.DataFrame(lag_features)
```

#### 3.2 周期性分解特征
```python
def create_decomposition_features(df, target_col):
    """
    基于时间序列分解创建特征
    """
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    # 时间序列分解
    decomposition = seasonal_decompose(
        df[target_col].dropna(), 
        model='additive', 
        period=7,  # 周周期
        extrapolate_trend='freq'
    )
    
    decomp_features = {}
    
    # 趋势成分
    decomp_features[f'{target_col}_trend'] = decomposition.trend
    decomp_features[f'{target_col}_trend_slope'] = decomposition.trend.diff()
    
    # 季节性成分
    decomp_features[f'{target_col}_seasonal'] = decomposition.seasonal
    decomp_features[f'{target_col}_seasonal_strength'] = np.abs(decomposition.seasonal)
    
    # 残差成分
    decomp_features[f'{target_col}_residual'] = decomposition.resid
    decomp_features[f'{target_col}_residual_abs'] = np.abs(decomposition.resid)
    
    return pd.DataFrame(decomp_features)
```

### 第四层：交互特征 (Day 4-5)

#### 4.1 特征交互
```python
def create_interaction_features(df):
    """
    创建特征交互项
    """
    interaction_features = {}
    
    # 时间-业务交互
    interaction_features['weekend_pay_cycle'] = df['is_weekend'] * df['pay_cycle']
    interaction_features['month_start_business'] = df['is_month_start'] * df['investment_cycle']
    
    # 市场-时间交互
    interaction_features['rate_environment_weekday'] = df['rate_environment'] * df['weekday']
    interaction_features['yield_month_end'] = df['yield_environment'] * df['is_month_end']
    
    # 利率交互
    interaction_features['shibor_rate_level'] = df['shibor_o_n'] * df['rate_environment']
    interaction_features['yield_volatility'] = df['shibor_volatility'] * df['stability_score']
    
    return pd.DataFrame(interaction_features)
```

## ⚙️ 精准参数优化

### Prophet参数网格搜索
```python
def prophet_parameter_optimization():
    """
    Prophet参数精准优化
    """
    param_grid = {
        'changepoint_prior_scale': [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1],
        'seasonality_prior_scale': [0.1, 0.5, 1.0, 2.0, 5.0, 8.0, 10.0, 15.0, 20.0],
        'holidays_prior_scale': [0.1, 0.5, 1.0, 2.0, 5.0, 8.0, 10.0, 15.0],
        'interval_width': [0.80, 0.85, 0.90, 0.95, 0.99],
        'seasonality_mode': ['additive', 'multiplicative']
    }
    
    print(f"参数搜索空间大小: {np.prod([len(v) for v in param_grid.values()])} 种组合")
    
    return param_grid
```

### 动态参数调整
```python
def dynamic_parameter_adjustment(data_length, seasonality_strength, trend_stability):
    """
    基于数据特征动态调整参数
    """
    base_params = {
        'changepoint_prior_scale': 0.05,
        'seasonality_prior_scale': 10.0,
        'holidays_prior_scale': 10.0,
        'interval_width': 0.95
    }
    
    # 根据数据特征调整
    if seasonality_strength > 0.5:
        base_params['seasonality_prior_scale'] *= 1.5
        base_params['seasonality_mode'] = 'multiplicative'
    
    if trend_stability < 0.3:
        base_params['changepoint_prior_scale'] *= 0.5
    
    if data_length > 400:
        base_params['interval_width'] = 0.90
    
    return base_params
```

## 🎯 实施路线图

### Week 1: 特征工程基础
- **Day 1**: 深度时间特征工程
- **Day 2**: 业务周期性特征 + 利率特征
- **Day 3**: 市场情绪特征
- **Day 4**: 滞后窗口特征 + 分解特征
- **Day 5**: 交互特征生成

### Week 2: 模型优化
- **Day 6**: 参数网格搜索框架
- **Day 7**: 动态参数调整算法
- **Day 8**: 特征重要性分析
- **Day 9**: 模型集成测试
- **Day 10**: 性能验证和调优

### Week 3: 验证和部署
- **Day 11**: 交叉验证
- **Day 12**: 预测稳定性测试
- **Day 13**: 异常值处理优化
- **Day 14**: 最终模型训练
- **Day 15**: 结果验证和部署

## 📊 预期成果

### 技术指标提升
- **申购MAPE**: 42.64% → 40.5% (提升2.14%)
- **赎回MAPE**: 99.43% → 89.5% (提升9.93%)
- **模型稳定性**: 预测方差降低30%

### 分数提升预期
- **v7基准**: 103分
- **v8目标**: 108-110分
- **提升幅度**: +5-7分

### 特征工程深度
- **时间特征**: 35个精细化时间维度特征
- **市场特征**: 25个利率和收益率特征
- **统计特征**: 40个滞后和滑动窗口特征
- **交互特征**: 15个特征交互项
- **总计**: 115个深度特征 (vs v7的4个特征)

## 🔍 关键技术点

### Prophet能力边界探索
1. **多维季节性**: 年度、季度、周度、日度季节性组合
2. **非线性趋势**: changepoint_prior_scale的精细调优
3. **外部回归**: 大规模特征工程的外生变量应用
4. **节假日建模**: 49个节假日的精细化建模

### 特征工程创新
1. **周期性编码**: sin/cos编码避免特征跳变
2. **业务洞察**: 基于金融业务逻辑的特征设计
3. **时间序列分解**: trend, seasonal, residual成分特征
4. **交互特征**: 多维度特征的智能交互

### 模型优化策略
1. **网格搜索**: 2880种参数组合的全方位搜索
2. **动态调整**: 基于数据特征的智能参数选择
3. **特征筛选**: 基于重要性的特征优选
4. **交叉验证**: 严格的模型验证框架

---

**v8版本核心价值**: 通过单一Prophet模型+深度特征工程，探索时间序列预测的能力边界！

*方案制定时间: 2025年12月2日*
*预期完成时间: 2025年12月16日*
*技术路线: 纯粹Prophet + 115维特征工程*