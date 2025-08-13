"""可视化管理模块，提供统一的图表生成和保存功能"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
from datetime import datetime

from .logger import get_logger


class VisualizationManager:
    """可视化管理器"""
    
    def __init__(self, output_dir: str = "figures", style: str = "default"):
        """
        初始化可视化管理器
        
        Args:
            output_dir: 输出目录
            style: 图表样式
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = get_logger("visualization")
        
        # 设置图表样式
        self._setup_style(style)
        
        # 颜色方案 - 参考 train_model.py 的简洁配色
        self.colors = {
            'phase1': 'blue',  # 蓝色
            'phase2': 'red',   # 红色
            'existing': 'blue', # 已有任务用蓝色
            'new': 'red',      # 新任务用红色
            'pro_saccade': 'blue',
            'anti_saccade': 'red',
            'delay_pro': 'green',
            'delay_anti': 'orange'
        }
    
    def _setup_style(self, style: str):
        """设置图表样式"""
        if style == "default":
            plt.style.use('default')
        elif style == "seaborn":
            plt.style.use('seaborn-v0_8')
        elif style == "ggplot":
            plt.style.use('ggplot')
        
        # 设置英文字体
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 设置图表大小
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['figure.dpi'] = 100
    
    def plot_training_loss(self, phase1_losses: List[float], phase2_losses: List[float] = None,
                          save_path: Optional[str] = None) -> str:
        """
        绘制训练损失曲线
        
        Args:
            phase1_losses: 第一阶段损失
            phase2_losses: 第二阶段损失
            save_path: 保存路径
        
        Returns:
            保存的文件路径
        """
        fig, axes = plt.subplots(1, 2 if phase2_losses else 1, figsize=(15, 6))
        
        if phase2_losses:
            # 两个子图
            # 第一阶段损失
            axes[0].plot(phase1_losses, label='Phase 1', color='blue')
            axes[0].set_title('Phase 1 Training Loss')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].legend()
            axes[0].grid(True)
            
            # 第二阶段损失
            axes[1].plot(phase2_losses, label='Phase 2', color='red')
            axes[1].set_title('Phase 2 Transfer Learning Loss')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Loss')
            axes[1].legend()
            axes[1].grid(True)
        else:
            # 单个图
            axes.plot(phase1_losses, label='Phase 1', color='blue')
            axes.set_title('Training Loss Curve')
            axes.set_xlabel('Epoch')
            axes.set_ylabel('Loss')
            axes.legend()
            axes.grid(True)
        
        plt.tight_layout()
        
        # 保存图表
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = self.output_dir / f"training_loss_{timestamp}.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"训练损失图表保存: {save_path}")
        return str(save_path)
    
    def plot_task_performance(self, evaluations: List[Dict[str, float]], 
                            task_names: List[str], phase: str = "phase1",
                            save_path: Optional[str] = None) -> str:
        """
        绘制任务性能曲线
        
        Args:
            evaluations: 评估结果列表
            task_names: 任务名称列表
            phase: 训练阶段
            save_path: 保存路径
        
        Returns:
            保存的文件路径
        """
        if not evaluations:
            raise ValueError("评估结果为空")
        
        # 提取性能数据
        epochs = list(range(0, len(evaluations) * 10, 10))  # 假设每10个epoch评估一次
        
        plt.figure(figsize=(10, 6))
        
        for task in task_names:
            performances = []
            for eval_result in evaluations:
                perf = eval_result.get(task, 0.0)
                performances.append(perf)
            
            color = self.colors.get(task, '#666666')
            plt.plot(epochs, performances, label=task, marker='o', 
                    color=color, linewidth=2, markersize=6)
        
        plt.title(f'{phase} Task Performance', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Performance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 保存图表
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = self.output_dir / f"{phase}_performance_{timestamp}.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"任务性能图表保存: {save_path}")
        return str(save_path)
    
    def plot_transfer_learning_results(self, evaluations: List[Dict[str, Dict[str, float]]],
                                     save_path: Optional[str] = None) -> str:
        """
        绘制迁移学习结果
        
        Args:
            evaluations: 迁移学习评估结果
            save_path: 保存路径
        
        Returns:
            保存的文件路径
        """
        if not evaluations:
            raise ValueError("评估结果为空")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        epochs = list(range(0, len(evaluations) * 5, 5))  # 假设每5个epoch评估一次
        
        # 已有任务性能
        axes[0, 0].set_title('Existing Tasks Performance (Transfer)')
        existing_tasks = list(evaluations[0].get('existing_tasks', {}).keys())
        for task in existing_tasks:
            performances = [eval_result.get('existing_tasks', {}).get(task, 0) 
                          for eval_result in evaluations]
            axes[0, 0].plot(epochs, performances, label=f'{task} (existing)', marker='s')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Performance')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 新任务性能
        axes[0, 1].set_title('New Tasks Performance (Transfer)')
        new_tasks = list(evaluations[0].get('new_tasks', {}).keys())
        for task in new_tasks:
            performances = [eval_result.get('new_tasks', {}).get(task, 0) 
                          for eval_result in evaluations]
            axes[0, 1].plot(epochs, performances, label=f'{task} (new)', marker='^', color='red')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Performance')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 迁移学习汇总
        axes[1, 0].set_title('Transfer Learning Summary')
        avg_existing = [eval_result.get('summary', {}).get('avg_existing', 0) 
                       for eval_result in evaluations]
        avg_new = [eval_result.get('summary', {}).get('avg_new', 0) 
                  for eval_result in evaluations]
        
        axes[1, 0].plot(epochs, avg_existing, label='Avg Existing Tasks', marker='o', color='blue')
        axes[1, 0].plot(epochs, avg_new, label='Avg New Tasks', marker='s', color='red')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Average Performance')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 性能对比柱状图
        axes[1, 1].set_title('Final Performance Comparison', fontsize=12, fontweight='bold')
        if evaluations:
            final_eval = evaluations[-1]
            existing_perfs = list(final_eval.get('existing_tasks', {}).values())
            new_perfs = list(final_eval.get('new_tasks', {}).values())
            
            x = np.arange(len(existing_perfs) + len(new_perfs))
            all_perfs = existing_perfs + new_perfs
            colors = [self.colors['existing']] * len(existing_perfs) + [self.colors['new']] * len(new_perfs)
            
            bars = axes[1, 1].bar(x, all_perfs, color=colors, alpha=0.7)
            axes[1, 1].set_xlabel('Tasks')
            axes[1, 1].set_ylabel('Final Performance')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(list(final_eval.get('existing_tasks', {}).keys()) + 
                                     list(final_eval.get('new_tasks', {}).keys()), 
                                     rotation=45)
            axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # 保存图表
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = self.output_dir / f"transfer_learning_{timestamp}.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"迁移学习结果图表保存: {save_path}")
        return str(save_path)
    
    def plot_comprehensive_results(self, results: Dict[str, Any], 
                                 save_path: Optional[str] = None) -> str:
        """
        绘制综合结果图表
        
        Args:
            results: 训练结果字典
            save_path: 保存路径
        
        Returns:
            保存的文件路径
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. 第一阶段训练损失
        if results.get('phase1_losses'):
            axes[0, 0].plot(results['phase1_losses'], color=self.colors['phase1'], linewidth=2)
            axes[0, 0].set_title('Phase 1 Training Loss', fontsize=12, fontweight='bold')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 第二阶段迁移学习损失
        if results.get('phase2_losses'):
            axes[0, 1].plot(results['phase2_losses'], color=self.colors['phase2'], linewidth=2)
            axes[0, 1].set_title('Phase 2 Transfer Learning Loss', fontsize=12, fontweight='bold')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 第一阶段任务性能
        if results.get('phase1_evaluations'):
            phase1_evals = results['phase1_evaluations']
            if phase1_evals:
                # 过滤掉非数字字段，只保留任务性能指标
                first_eval = phase1_evals[0]
                task_names = [k for k, v in first_eval.items() 
                            if k != 'error_analysis' and isinstance(v, (int, float))]
                epochs = list(range(0, len(phase1_evals) * 10, 10))
                
                for task in task_names:
                    performances = [eval_result.get(task, 0) for eval_result in phase1_evals]
                    color = self.colors.get(task, '#666666')
                    axes[0, 2].plot(epochs, performances, label=task, marker='o', color=color)
                
                axes[0, 2].set_title('Phase 1 Task Performance', fontsize=12, fontweight='bold')
                axes[0, 2].set_xlabel('Epoch')
                axes[0, 2].set_ylabel('Performance')
                axes[0, 2].legend()
                axes[0, 2].grid(True, alpha=0.3)
        
        # 4. 迁移学习效果 - 已有任务性能
        if results.get('phase2_evaluations'):
            phase2_evals = results['phase2_evaluations']
            if phase2_evals:
                epochs = list(range(0, len(phase2_evals) * 5, 5))
                
                # 检查数据结构，支持新的评估格式
                first_eval = phase2_evals[0]
                if 'existing_tasks' in first_eval:
                    # 旧格式：existing_tasks/new_tasks
                    existing_tasks = list(first_eval.get('existing_tasks', {}).keys())
                    for i, task in enumerate(existing_tasks):
                        performances = [eval_result.get('existing_tasks', {}).get(task, 0) 
                                      for eval_result in phase2_evals]
                        color = self.colors.get(task, f'C{i}')
                        axes[1, 0].plot(epochs, performances, label=f'{task} (existing)', 
                                       marker='s', color=color)
                else:
                    # 新格式：直接使用任务名称
                    # 第一阶段任务（已有任务）
                    phase1_tasks = ['pro_saccade', 'anti_saccade', 'delay_pro']
                    
                    for i, task in enumerate(phase1_tasks):
                        # 过滤掉非数字字段
                        performances = []
                        for eval_result in phase2_evals:
                            if task in eval_result and isinstance(eval_result.get(task, 0), (int, float)):
                                performances.append(eval_result.get(task, 0))
                            else:
                                performances.append(0)
                        color = self.colors.get(task, f'C{i}')
                        axes[1, 0].plot(epochs, performances, label=f'{task} (existing)', 
                                       marker='s', color=color)
                
                # 只有在有数据时才设置标题和图例
                if phase1_tasks:
                    axes[1, 0].set_title('Existing Tasks Performance (Transfer)', fontsize=12, fontweight='bold')
                    axes[1, 0].set_xlabel('Epoch')
                    axes[1, 0].set_ylabel('Performance')
                    axes[1, 0].legend()
                    axes[1, 0].grid(True, alpha=0.3)
        
        # 5. 迁移学习效果 - 新任务性能
        if results.get('phase2_evaluations'):
            phase2_evals = results['phase2_evaluations']
            if phase2_evals:
                epochs = list(range(0, len(phase2_evals) * 5, 5))
                
                # 检查数据结构，支持新的评估格式
                first_eval = phase2_evals[0]
                if 'new_tasks' in first_eval:
                    # 旧格式：existing_tasks/new_tasks
                    new_tasks = list(first_eval.get('new_tasks', {}).keys())
                    for i, task in enumerate(new_tasks):
                        performances = [eval_result.get('new_tasks', {}).get(task, 0) 
                                      for eval_result in phase2_evals]
                        color = self.colors.get(task, f'C{i+len(existing_tasks)}')
                        axes[1, 1].plot(epochs, performances, label=f'{task} (new)', 
                                       marker='^', color=color)
                else:
                    # 新格式：第二阶段任务（新任务）
                    phase2_tasks = ['delay_anti']  # 第二阶段主要训练的任务
                    
                    for i, task in enumerate(phase2_tasks):
                        # 过滤掉非数字字段
                        performances = []
                        for eval_result in phase2_evals:
                            if task in eval_result and isinstance(eval_result.get(task, 0), (int, float)):
                                performances.append(eval_result.get(task, 0))
                            else:
                                performances.append(0)
                        color = self.colors.get(task, f'C{i+len(phase1_tasks)}')
                        axes[1, 1].plot(epochs, performances, label=f'{task} (new)', 
                                       marker='^', color=color)
                
                # 只有在有数据时才设置标题和图例
                if phase2_tasks:
                    axes[1, 1].set_title('New Tasks Performance (Transfer)', fontsize=12, fontweight='bold')
                    axes[1, 1].set_xlabel('Epoch')
                    axes[1, 1].set_ylabel('Performance')
                    axes[1, 1].legend()
                    axes[1, 1].grid(True, alpha=0.3)
        
        # 6. 迁移学习汇总
        if results.get('phase2_evaluations'):
            phase2_evals = results['phase2_evaluations']
            if phase2_evals:
                epochs = list(range(0, len(phase2_evals) * 5, 5))
                
                # 计算平均性能
                avg_existing = []
                avg_new = []
                
                for eval_result in phase2_evals:
                    if 'summary' in eval_result:
                        # 旧格式
                        avg_existing.append(eval_result.get('summary', {}).get('avg_existing', 0))
                        avg_new.append(eval_result.get('summary', {}).get('avg_new', 0))
                    else:
                        # 新格式：分别计算第一阶段和第二阶段任务的平均性能
                        phase1_tasks = ['pro_saccade', 'anti_saccade', 'delay_pro']
                        phase2_tasks = ['delay_anti']
                        
                        # 计算第一阶段任务平均性能（过滤掉非数字字段）
                        phase1_performances = []
                        for task in phase1_tasks:
                            if task in eval_result and isinstance(eval_result.get(task, 0), (int, float)):
                                phase1_performances.append(eval_result.get(task, 0))
                        avg_phase1 = sum(phase1_performances) / len(phase1_performances) if phase1_performances else 0
                        avg_existing.append(avg_phase1)
                        
                        # 计算第二阶段任务平均性能（过滤掉非数字字段）
                        phase2_performances = []
                        for task in phase2_tasks:
                            if task in eval_result and isinstance(eval_result.get(task, 0), (int, float)):
                                phase2_performances.append(eval_result.get(task, 0))
                        avg_phase2 = sum(phase2_performances) / len(phase2_performances) if phase2_performances else 0
                        avg_new.append(avg_phase2)
                
                # 只有在有数据时才绘制
                if avg_existing and avg_new:
                    axes[1, 2].plot(epochs, avg_existing, label='Avg Existing Tasks', 
                                   marker='o', color=self.colors['existing'], linewidth=2)
                    axes[1, 2].plot(epochs, avg_new, label='Avg New Tasks', 
                                   marker='s', color=self.colors['new'], linewidth=2)
                    
                    axes[1, 2].set_title('Transfer Learning Summary', fontsize=12, fontweight='bold')
                    axes[1, 2].set_xlabel('Epoch')
                    axes[1, 2].set_ylabel('Average Performance')
                    axes[1, 2].legend()
                    axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = self.output_dir / f"comprehensive_results_{timestamp}.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"综合结果图表保存: {save_path}")
        return str(save_path)
    
    def save_results_summary(self, results: Dict[str, Any], 
                           save_path: Optional[str] = None) -> str:
        """
        保存结果摘要
        
        Args:
            results: 训练结果字典
            save_path: 保存路径
        
        Returns:
            保存的文件路径
        """
        summary = {
            'timestamp': datetime.now().isoformat(),
            'model_path': results.get('model_path', 'N/A'),
            'phase1_summary': {
                'total_epochs': len(results.get('phase1_losses', [])),
                'final_loss': results.get('phase1_losses', [0])[-1] if results.get('phase1_losses') else 0,
                'final_evaluation': results.get('phase1_evaluations', [{}])[-1] if results.get('phase1_evaluations') else {}
            },
            'phase2_summary': {
                'total_epochs': len(results.get('phase2_losses', [])),
                'final_loss': results.get('phase2_losses', [0])[-1] if results.get('phase2_losses') else 0,
                'final_evaluation': results.get('phase2_evaluations', [{}])[-1] if results.get('phase2_evaluations') else {}
            }
        }
        
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = self.output_dir / f"results_summary_{timestamp}.json"
        else:
            save_path = Path(save_path)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"结果摘要保存: {save_path}")
        return str(save_path)