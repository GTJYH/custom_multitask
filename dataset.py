"""基于猴子实验CSV的数据集加载与到trial参数的映射。

读取 `custom_multitask/data/*.csv`，筛选需要的task类型与有效trial，
并将 direction 与 error_type 转换为 psychometric 模式可用的参数。
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import os

# 任务ID到规则名称的映射
TASK_ID_TO_RULE = {
    0: 'anti_saccade',
    1: 'pro_saccade', 
    2: 'delay_pro',
    3: 'delay_anti'
}

def load_stage_data(data_dir: str, stage: str, subject: Optional[str] = None) -> Dict[str, List[Tuple[int, int]]]:
    """
    按阶段加载数据：
    - stage='phase1': 只读取 {subject}Task3.csv 的所有任务
    - stage='phase2': 只读取 {subject}Task4.csv 的所有任务
    
    Args:
        data_dir: 数据目录路径
        stage: 'phase1' 或 'phase2'
        subject: 被试名称 ('DD' 或 'Evender')，None表示加载所有被试
    
    Returns:
        Dict[rule_name, List[Tuple[direction, error_type]]]
    """
    if stage not in ['phase1', 'phase2']:
        raise ValueError("stage must be 'phase1' or 'phase2'")
    
    # 确定要读取的文件
    if subject:
        filename = f"{subject}Task{3 if stage == 'phase1' else 4}.csv"
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        files_to_read = [filepath]
    else:
        # 读取所有被试的对应阶段文件
        subjects = ['DD', 'Evender']
        files_to_read = []
        for subj in subjects:
            filename = f"{subj}Task{3 if stage == 'phase1' else 4}.csv"
            filepath = os.path.join(data_dir, filename)
            if os.path.exists(filepath):
                files_to_read.append(filepath)
        
        if not files_to_read:
            raise FileNotFoundError(f"No {stage} data files found in {data_dir}")
    
    # 加载数据（不按任务过滤，读取所有任务）
    all_data = []
    for filepath in files_to_read:
        df = pd.read_csv(filepath)
        all_data.append(df)
    
    if not all_data:
        raise ValueError(f"No data found for {stage}")
    
    # 合并所有数据，保持原始顺序
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # 按任务分组，只保留我们关心的任务
    per_rule: Dict[str, List[Tuple[int, int]]] = {}
    for task_id in TASK_ID_TO_RULE.keys():
        task_data = combined_df[combined_df['task'] == task_id]
        if len(task_data) > 0:  # 只有当有数据时才添加
            rule_name = TASK_ID_TO_RULE[task_id]
            # 转换为 (direction, error_type) 元组列表
            params = [(row['direction'], row['error_type']) for _, row in task_data.iterrows()]
            per_rule[rule_name] = params
    
    return per_rule

def sample_params_for_stage(data_dir: str, stage: str, rule_name: str, 
                           max_samples: int = None, shuffle: bool = False,
                           subject: Optional[str] = None, start_idx: int = 0) -> List[Tuple[int, int]]:
    """
    从指定阶段的数据中采样参数
    
    Args:
        data_dir: 数据目录路径
        stage: 'phase1' 或 'phase2'
        rule_name: 任务规则名称
        max_samples: 最大采样数量，None表示全部
        shuffle: 是否打乱顺序
        subject: 被试名称
        start_idx: 起始索引
    
    Returns:
        List[Tuple[direction, error_type]]
    """
    # 加载阶段数据
    stage_data = load_stage_data(data_dir, stage, subject)
    
    if rule_name not in stage_data:
        available_rules = list(stage_data.keys())
        raise ValueError(f"Rule '{rule_name}' not found in {stage} data. Available rules: {available_rules}")
    
    params = stage_data[rule_name]
    
    # 应用起始索引
    if start_idx >= len(params):
        return []
    
    params = params[start_idx:]
    
    # 应用最大采样数量
    if max_samples is not None:
        params = params[:max_samples]
    
    # 是否打乱
    if shuffle:
        params = params.copy()
        np.random.shuffle(params)
    
    return params

def get_stage_info(data_dir: str, stage: str, subject: Optional[str] = None) -> Dict[str, int]:
    """
    获取指定阶段的数据信息
    
    Returns:
        Dict[rule_name, data_count]
    """
    stage_data = load_stage_data(data_dir, stage, subject)
    return {rule_name: len(params) for rule_name, params in stage_data.items()}

# 保持向后兼容的函数
def load_all_data(data_dir: str) -> Dict[str, List[Tuple[int, int]]]:
    """
    加载所有数据（保持向后兼容）
    """
    print("Warning: load_all_data is deprecated. Use load_stage_data instead.")
    # 合并两个阶段的数据
    phase1_data = load_stage_data(data_dir, 'phase1')
    phase2_data = load_stage_data(data_dir, 'phase2')
    
    # 合并数据
    all_data = {}
    for rule_name, params in phase1_data.items():
        all_data[rule_name] = params
    for rule_name, params in phase2_data.items():
        if rule_name in all_data:
            all_data[rule_name].extend(params)
        else:
            all_data[rule_name] = params
    
    return all_data

def filter_and_group_trials(rows: List[Tuple[int, int]]) -> Dict[str, List[Tuple[int, int]]]:
    """
    过滤和分组试验数据（保持向后兼容）
    """
    print("Warning: filter_and_group_trials is deprecated. Use load_stage_data instead.")
    per_rule = {}
    for task_id, direction, error_type in rows:
        if task_id in TASK_ID_TO_RULE:
            rule_name = TASK_ID_TO_RULE[task_id]
            if rule_name not in per_rule:
                per_rule[rule_name] = []
            per_rule[rule_name].append((direction, error_type))
    return per_rule

def build_params_for_rule(data_dir: str, rule_name: str, subject: Optional[str] = None) -> List[Tuple[int, int]]:
    """
    构建指定规则的参数（保持向后兼容）
    """
    print("Warning: build_params_for_rule is deprecated. Use sample_params_for_stage instead.")
    # 尝试从两个阶段都获取数据
    all_params = []
    for stage in ['phase1', 'phase2']:
        try:
            stage_data = load_stage_data(data_dir, stage, subject)
            if rule_name in stage_data:
                all_params.extend(stage_data[rule_name])
        except (FileNotFoundError, ValueError):
            continue
    
    return all_params


