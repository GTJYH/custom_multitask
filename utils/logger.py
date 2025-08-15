"""日志系统模块，提供统一的日志记录功能"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path


class ColoredFormatter(logging.Formatter):
    """彩色日志格式化器"""
    
    # 颜色代码
    COLORS = {
        'DEBUG': '\033[36m',    # 青色
        'INFO': '\033[32m',     # 绿色
        'WARNING': '\033[33m',  # 黄色
        'ERROR': '\033[31m',    # 红色
        'CRITICAL': '\033[35m', # 紫色
        'RESET': '\033[0m'      # 重置
    }
    
    def format(self, record):
        # 添加颜色
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
        
        return super().format(record)


def setup_logger(
    name: str = "custom_multitask",
    level: int = logging.INFO,
    log_dir: Optional[str] = None,
    console_output: bool = True,
    file_output: bool = True,
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    date_format: str = "%Y-%m-%d %H:%M:%S"
) -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        name: 日志记录器名称
        level: 日志级别
        log_dir: 日志文件目录
        console_output: 是否输出到控制台
        file_output: 是否输出到文件
        log_format: 日志格式
        date_format: 日期格式
    
    Returns:
        配置好的日志记录器
    """
    logger = logging.getLogger(name)
    
    # 避免重复添加处理器
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # 创建格式化器
    formatter = logging.Formatter(log_format, date_format)
    colored_formatter = ColoredFormatter(log_format, date_format)
    
    # 控制台处理器
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(colored_formatter)
        logger.addHandler(console_handler)
    
    # 文件处理器
    if file_output and log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成日志文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{name}_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"日志文件创建: {log_file}")
    
    return logger


def get_logger(name: str = "custom_multitask") -> logging.Logger:
    """
    获取日志记录器
    
    Args:
        name: 日志记录器名称
    
    Returns:
        日志记录器
    """
    return logging.getLogger(name)


class TrainingLogger:
    """训练专用日志记录器"""
    
    def __init__(self, log_dir: str, experiment_name: str):
        """
        初始化训练日志记录器
        
        Args:
            log_dir: 日志目录
            experiment_name: 实验名称
        """
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        
        # 清理实验名称，确保路径安全
        safe_experiment_name = self._sanitize_experiment_name(experiment_name)
        
        # 创建日志目录
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志记录器
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_name = f"training_{safe_experiment_name}_{timestamp}"
        self.logger = setup_logger(
            name=log_name,
            log_dir=str(self.log_dir),
            console_output=True,
            file_output=True
        )
        
        # 记录实验开始
        self.logger.info(f"实验开始: {experiment_name}")
        self.logger.info(f"日志目录: {self.log_dir}")
    
    def _sanitize_experiment_name(self, name: str) -> str:
        """清理实验名称，确保路径安全"""
        import re
        # 移除或替换不安全的字符
        safe_name = re.sub(r'[<>:"/\\|?*]', '_', name)
        # 移除前后空格
        safe_name = safe_name.strip()
        # 如果为空，使用默认名称
        if not safe_name:
            safe_name = "default_experiment"
        return safe_name
        

    
    def log_hyperparameters(self, hp: Dict[str, Any]):
        """记录超参数"""
        self.logger.info("超参数配置:")
        for key, value in hp.items():
            self.logger.info(f"  {key}: {value}")
    
    def log_training_start(self, phase: str, epochs: int, tasks: list):
        """记录训练开始"""
        self.logger.info(f"=== {phase} 训练开始 ===")
        self.logger.info(f"训练轮数: {epochs}")
        self.logger.info(f"训练任务: {tasks}")
    
    def log_epoch(self, phase: str, epoch: int, total_epochs: int, loss: float, 
                  metrics: Optional[Dict[str, float]] = None):
        """记录每个epoch的信息"""
        self.logger.info(f"{phase} Epoch {epoch}/{total_epochs}, 损失: {loss:.6f}")
        if metrics:
            metrics_str = ", ".join([f"{k}: {v:.3f}" for k, v in metrics.items()])
            self.logger.info(f"  性能指标: {metrics_str}")
    
    def log_training_end(self, phase: str, final_loss: float, final_metrics: Optional[Dict[str, float]] = None):
        """记录训练结束"""
        self.logger.info(f"=== {phase} 训练完成 ===")
        self.logger.info(f"最终损失: {final_loss:.6f}")
        if final_metrics:
            metrics_str = ", ".join([f"{k}: {v:.3f}" for k, v in final_metrics.items()])
            self.logger.info(f"最终性能: {metrics_str}")
    
    def log_error(self, error: Exception, context: str = ""):
        """记录错误信息"""
        self.logger.error(f"错误 {context}: {str(error)}", exc_info=True)
    
    def log_model_save(self, model_path: str):
        """记录模型保存"""
        self.logger.info(f"模型保存: {model_path}")
    
    def log_evaluation(self, task_name: str, performance: float):
        """记录评估结果"""
        self.logger.info(f"任务 {task_name} 性能: {performance:.3f}")
    
    def log_transfer_learning(self, existing_tasks: Dict[str, float], new_tasks: Dict[str, float]):
        """记录迁移学习结果"""
        self.logger.info("迁移学习评估结果:")
        if existing_tasks:
            existing_str = ", ".join([f"{k}: {v:.3f}" for k, v in existing_tasks.items()])
            self.logger.info(f"  已有任务: {existing_str}")
        if new_tasks:
            new_str = ", ".join([f"{k}: {v:.3f}" for k, v in new_tasks.items()])
            self.logger.info(f"  新任务: {new_str}")


class AnalysisLogger:
    """分析模块专用日志记录器"""
    
    def __init__(self, name: str = "analysis"):
        """
        初始化分析日志记录器
        
        Args:
            name: 日志记录器名称
        """
        self.name = name
        
        # 设置日志记录器
        self.logger = setup_logger(
            name=name,
            console_output=True,
            file_output=False
        )
        
        # 记录分析开始
        self.logger.info(f"分析模块启动: {name}")
    
    def log_analysis_start(self, analysis_type: str, model_path: str):
        """记录分析开始"""
        self.logger.info(f"=== 开始 {analysis_type} 分析 ===")
        self.logger.info(f"模型路径: {model_path}")
    
    def log_analysis_progress(self, current: int, total: int, description: str = ""):
        """记录分析进度"""
        percentage = (current / total) * 100
        self.logger.info(f"进度: {current}/{total} ({percentage:.1f}%) - {description}")
    
    def log_analysis_complete(self, analysis_type: str, results_summary: str = ""):
        """记录分析完成"""
        self.logger.info(f"=== {analysis_type} 分析完成 ===")
        if results_summary:
            self.logger.info(f"结果摘要: {results_summary}")
    
    def log_error(self, error: Exception, context: str = ""):
        """记录错误信息"""
        self.logger.error(f"错误 {context}: {str(error)}", exc_info=True)
    
    def log_warning(self, message: str):
        """记录警告信息"""
        self.logger.warning(message)
    
    def log_info(self, message: str):
        """记录一般信息"""
        self.logger.info(message)
    
    def log_debug(self, message: str):
        """记录调试信息"""
        self.logger.debug(message)