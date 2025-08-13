"""配置管理模块，统一管理超参数和实验配置"""

import json
import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class HyperParameters:
    """超参数数据类"""
    
    # 基础参数
    dt: float = 20.0                    # 时间步长 (ms)
    tau: float = 100.0                  # 时间常数 (ms)
    alpha: float = 0.2                  # alpha = dt/tau
    sigma_x: float = 0.1                # 输入噪声
    
    # 网络结构
    n_eachring: int = 32                # 环单元数
    n_input: int = 37                   # 输入维度 (1注视 + 32环 + 4任务信号)
    n_output: int = 33                  # 输出维度 (1注视 + 32环)
    
    # 训练参数
    learning_rate: float = 0.001        # 学习率
    batch_size: int = 32                # 批次大小
    loss_type: str = 'lsq'              # 损失函数类型
    
    # 训练阶段参数
    phase1_epochs: int = 400            # 第一阶段训练轮数
    phase2_epochs: int = 40             # 第二阶段训练轮数
    transfer_lr_factor: float = 0.1     # 迁移学习学习率因子
    
    # 评估参数
    eval_interval: int = 10             # 评估间隔
    num_eval_trials: int = 100          # 评估试验数量
    
    # 随机种子
    random_seed: int = 42               # 随机种子
    
    # 持续学习参数
    continual_learning: dict = None     # 持续学习配置
    
    def __post_init__(self):
        """初始化后处理"""
        # 计算alpha
        self.alpha = self.dt / self.tau
        
        # 设置持续学习默认配置
        if self.continual_learning is None:
            self.continual_learning = {
                'enabled': False,
                'c_intsyn': 1.0,
                'ksi_intsyn': 0.01,
                'ewc_lambda': 100.0
            }
        
        # 设置随机种子
        np.random.seed(self.random_seed)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        # 确保持续学习配置被正确序列化
        if 'continual_learning' in data and data['continual_learning'] is None:
            data['continual_learning'] = {
                'enabled': False,
                'c_intsyn': 1.0,
                'ksi_intsyn': 0.01,
                'ewc_lambda': 100.0
            }
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HyperParameters':
        """从字典创建"""
        # 处理持续学习配置
        if 'continual_learning' in data and data['continual_learning'] is None:
            data['continual_learning'] = {
                'enabled': False,
                'c_intsyn': 1.0,
                'ksi_intsyn': 0.01,
                'ewc_lambda': 100.0
            }
        return cls(**data)


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_dir: str = "configs"):
        """
        初始化配置管理器
        
        Args:
            config_dir: 配置文件目录
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        # 默认配置
        self.default_config = {
            "random_training": {
                "description": "随机数据训练配置",
                "hyperparameters": HyperParameters(
                    phase1_epochs=400,
                    phase2_epochs=40,
                    batch_size=32,
                    learning_rate=0.001
                ).to_dict()
            },
            "psychometric_training": {
                "description": "真实数据训练配置",
                "hyperparameters": HyperParameters(
                    phase1_epochs=400,
                    phase2_epochs=40,
                    batch_size=32,
                    learning_rate=0.001,
                    eval_interval=10
                ).to_dict(),
                "data_config": {
                    "data_dir": "data",
                    "subjects": ["DD", "Evender"],
                    "stages": ["phase1", "phase2"]
                }
            },

        }
    
    def save_config(self, config_name: str, config: Dict[str, Any], 
                   format: str = "json") -> str:
        """
        保存配置到文件
        
        Args:
            config_name: 配置名称
            config: 配置字典
            format: 文件格式 ("json" 或 "yaml")
        
        Returns:
            配置文件路径
        """
        config_file = self.config_dir / f"{config_name}.{format}"
        
        if format.lower() == "json":
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        elif format.lower() == "yaml":
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        else:
            raise ValueError(f"不支持的格式: {format}")
        
        return str(config_file)
    
    def load_config(self, config_name: str, format: str = "json") -> Dict[str, Any]:
        """
        从文件加载配置
        
        Args:
            config_name: 配置名称
            format: 文件格式
        
        Returns:
            配置字典
        """
        config_file = self.config_dir / f"{config_name}.{format}"
        
        if not config_file.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_file}")
        
        if format.lower() == "json":
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif format.lower() == "yaml":
            with open(config_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            raise ValueError(f"不支持的格式: {format}")
    
    def get_default_config(self, config_type: str) -> Dict[str, Any]:
        """
        获取默认配置
        
        Args:
            config_type: 配置类型
        
        Returns:
            默认配置字典
        """
        if config_type not in self.default_config:
            raise ValueError(f"未知的配置类型: {config_type}")
        
        return self.default_config[config_type]
    
    def create_experiment_config(self, experiment_name: str, 
                               base_config: str = "random_training",
                               **overrides) -> Dict[str, Any]:
        """
        创建实验配置
        
        Args:
            experiment_name: 实验名称
            base_config: 基础配置类型
            **overrides: 覆盖参数
        
        Returns:
            实验配置字典
        """
        # 获取基础配置
        config = self.get_default_config(base_config).copy()
        
        # 添加实验信息
        config["experiment_name"] = experiment_name
        config["base_config"] = base_config
        config["created_at"] = str(np.datetime64('now'))
        
        # 应用覆盖参数
        if "hyperparameters" in overrides:
            hp_dict = config["hyperparameters"].copy()
            hp_dict.update(overrides["hyperparameters"])
            config["hyperparameters"] = hp_dict
        
        # 保存配置
        config_file = self.save_config(experiment_name, config)
        
        return config
    
    def list_configs(self) -> list:
        """列出所有配置文件"""
        configs = []
        for file in self.config_dir.glob("*.json"):
            configs.append(file.stem)
        for file in self.config_dir.glob("*.yaml"):
            configs.append(file.stem)
        return sorted(list(set(configs)))


class ExperimentConfig:
    """实验配置类"""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """
        初始化实验配置
        
        Args:
            config_dict: 配置字典
        """
        self.config_dict = config_dict
        self.experiment_name = config_dict.get("experiment_name", "unknown")
        self.description = config_dict.get("description", "")
        
        # 解析超参数
        hp_dict = config_dict.get("hyperparameters", {})
        self.hyperparameters = HyperParameters(**hp_dict)
        
        # 其他配置
        self.data_config = config_dict.get("data_config", {})
        self.training_config = config_dict.get("training_config", {})
    
    def get_hp(self) -> Dict[str, Any]:
        """获取超参数字典（用于兼容现有代码）"""
        hp_dict = self.hyperparameters.to_dict()
        # 添加随机数生成器
        hp_dict['rng'] = np.random.RandomState(self.hyperparameters.random_seed)
        return hp_dict
    
    def save(self, filepath: str):
        """保存配置到文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.config_dict, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, filepath: str) -> 'ExperimentConfig':
        """从文件加载配置"""
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls(config_dict)