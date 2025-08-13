"""指标跟踪模块，用于记录和分析训练过程中的各种指标"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import json
from pathlib import Path
from datetime import datetime

from .logger import get_logger


class MetricsTracker:
    """指标跟踪器"""
    
    def __init__(self, experiment_name: str, output_dir: str = "metrics"):
        """
        初始化指标跟踪器
        
        Args:
            experiment_name: 实验名称
            output_dir: 输出目录
        """
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = get_logger("metrics")
        
        # 初始化指标存储
        self.metrics = defaultdict(list)
        self.phase_metrics = defaultdict(lambda: defaultdict(list))
        
        # 记录开始时间
        self.start_time = datetime.now()
    
    def log_metric(self, phase: str, metric_name: str, value: float, 
                  epoch: Optional[int] = None, task: Optional[str] = None):
        """
        记录指标
        
        Args:
            phase: 训练阶段 ('phase1', 'phase2')
            metric_name: 指标名称
            value: 指标值
            epoch: 当前epoch
            task: 任务名称
        """
        timestamp = datetime.now()
        
        # 记录到全局指标
        self.metrics[metric_name].append({
            'phase': phase,
            'value': value,
            'epoch': epoch,
            'task': task,
            'timestamp': timestamp.isoformat()
        })
        
        # 记录到阶段指标
        if epoch is not None:
            self.phase_metrics[phase][metric_name].append({
                'epoch': epoch,
                'value': value,
                'task': task,
                'timestamp': timestamp.isoformat()
            })
        
        self.logger.debug(f"记录指标: {phase}.{metric_name} = {value:.6f}")
    
    def log_training_loss(self, phase: str, epoch: int, loss: float):
        """记录训练损失"""
        self.log_metric(phase, 'training_loss', loss, epoch)
    
    def log_task_performance(self, phase: str, epoch: int, task: str, performance: float):
        """记录任务性能"""
        self.log_metric(phase, 'task_performance', performance, epoch, task)
    
    def log_transfer_learning_metrics(self, phase: str, epoch: int, 
                                    existing_tasks: Dict[str, float], 
                                    new_tasks: Dict[str, float]):
        """记录迁移学习指标"""
        # 记录已有任务平均性能
        if existing_tasks:
            avg_existing = np.mean(list(existing_tasks.values()))
            self.log_metric(phase, 'avg_existing_performance', avg_existing, epoch)
        
        # 记录新任务平均性能
        if new_tasks:
            avg_new = np.mean(list(new_tasks.values()))
            self.log_metric(phase, 'avg_new_performance', avg_new, epoch)
        
        # 记录遗忘程度（已有任务性能下降）
        if existing_tasks and 'phase1' in self.phase_metrics:
            # 计算与第一阶段最终性能的差异
            phase1_final_perfs = {}
            for metric in self.phase_metrics['phase1']['task_performance']:
                if metric['task'] in existing_tasks:
                    phase1_final_perfs[metric['task']] = metric['value']
            
            if phase1_final_perfs:
                forgetting = []
                for task in existing_tasks:
                    if task in phase1_final_perfs:
                        forgetting.append(phase1_final_perfs[task] - existing_tasks[task])
                
                if forgetting:
                    avg_forgetting = np.mean(forgetting)
                    self.log_metric(phase, 'avg_forgetting', avg_forgetting, epoch)
    
    def log_error_analysis(self, phase: str, epoch: int, task: str, error_analysis: Dict[str, Any]):
        """记录错误分析结果
        
        Args:
            phase: 训练阶段 ('phase1', 'phase2')
            epoch: 当前epoch
            task: 任务名称
            error_analysis: 错误分析结果字典
        """
        # 记录各种错误率
        error_metrics = [
            'fixation_error_rate',
            'direction_error_rate', 
            'both_error_rate',
            'correct_rate',
            'total_error_rate'
        ]
        
        for metric in error_metrics:
            if metric in error_analysis:
                self.log_metric(phase, f'error_{metric}', error_analysis[metric], epoch, task)
        
        # 记录错误计数
        count_metrics = [
            'fixation_errors',
            'direction_errors',
            'both_errors',
            'correct_trials',
            'total_trials'
        ]
        
        for metric in count_metrics:
            if metric in error_analysis:
                self.log_metric(phase, f'error_count_{metric}', error_analysis[metric], epoch, task)
    
    def get_metric_series(self, metric_name: str, phase: Optional[str] = None) -> pd.DataFrame:
        """
        获取指标时间序列
        
        Args:
            metric_name: 指标名称
            phase: 训练阶段（可选）
        
        Returns:
            指标时间序列DataFrame
        """
        if phase:
            data = self.phase_metrics[phase][metric_name]
        else:
            data = self.metrics[metric_name]
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df.sort_values('timestamp')
    
    def get_latest_metrics(self, phase: Optional[str] = None) -> Dict[str, float]:
        """
        获取最新指标
        
        Args:
            phase: 训练阶段（可选）
        
        Returns:
            最新指标字典
        """
        latest_metrics = {}
        
        if phase:
            for metric_name, values in self.phase_metrics[phase].items():
                if values:
                    latest_metrics[metric_name] = values[-1]['value']
        else:
            for metric_name, values in self.metrics.items():
                if values:
                    latest_metrics[metric_name] = values[-1]['value']
        
        return latest_metrics
    
    def calculate_summary_statistics(self) -> Dict[str, Any]:
        """
        计算汇总统计
        
        Returns:
            汇总统计字典
        """
        summary = {
            'experiment_name': self.experiment_name,
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'duration': (datetime.now() - self.start_time).total_seconds(),
            'phases': {}
        }
        
        for phase in ['phase1', 'phase2']:
            if phase in self.phase_metrics:
                phase_summary = {
                    'total_epochs': len(self.phase_metrics[phase].get('training_loss', [])),
                    'metrics': {}
                }
                
                for metric_name, values in self.phase_metrics[phase].items():
                    if values:
                        values_list = [v['value'] for v in values]
                        phase_summary['metrics'][metric_name] = {
                            'count': len(values_list),
                            'mean': np.mean(values_list),
                            'std': np.std(values_list),
                            'min': np.min(values_list),
                            'max': np.max(values_list),
                            'latest': values_list[-1]
                        }
                
                summary['phases'][phase] = phase_summary
        
        return summary
    
    def save_metrics(self, filename: Optional[str] = None) -> str:
        """
        保存指标到文件
        
        Args:
            filename: 文件名（可选）
        
        Returns:
            保存的文件路径
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.experiment_name}_metrics_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        # 准备保存数据
        save_data = {
            'experiment_name': self.experiment_name,
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'metrics': dict(self.metrics),
            'phase_metrics': dict(self.phase_metrics),
            'summary': self.calculate_summary_statistics()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"指标保存: {filepath}")
        return str(filepath)
    
    def load_metrics(self, filepath: str):
        """
        从文件加载指标
        
        Args:
            filepath: 文件路径
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.experiment_name = data.get('experiment_name', self.experiment_name)
        self.metrics = defaultdict(list, data.get('metrics', {}))
        self.phase_metrics = defaultdict(lambda: defaultdict(list), data.get('phase_metrics', {}))
        
        self.logger.info(f"指标加载: {filepath}")
    
    def export_to_csv(self, filename: Optional[str] = None) -> str:
        """
        导出指标到CSV文件
        
        Args:
            filename: 文件名（可选）
        
        Returns:
            保存的文件路径
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.experiment_name}_metrics_{timestamp}.csv"
        
        filepath = self.output_dir / filename
        
        # 转换为DataFrame
        rows = []
        for metric_name, values in self.metrics.items():
            for value in values:
                row = {
                    'metric_name': metric_name,
                    'phase': value['phase'],
                    'value': value['value'],
                    'epoch': value['epoch'],
                    'task': value['task'],
                    'timestamp': value['timestamp']
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False, encoding='utf-8')
        
        self.logger.info(f"指标导出CSV: {filepath}")
        return str(filepath)
    
    def plot_metrics(self, metric_names: Optional[List[str]] = None, 
                    phases: Optional[List[str]] = None) -> Dict[str, str]:
        """
        绘制指标图表
        
        Args:
            metric_names: 要绘制的指标名称列表
            phases: 要绘制的阶段列表
        
        Returns:
            图表文件路径字典
        """
        import matplotlib.pyplot as plt
        
        if metric_names is None:
            metric_names = list(self.metrics.keys())
        
        if phases is None:
            phases = list(self.phase_metrics.keys())
        
        plot_files = {}
        
        for metric_name in metric_names:
            if metric_name not in self.metrics:
                continue
            
            plt.figure(figsize=(12, 6))
            
            for phase in phases:
                if phase in self.phase_metrics and metric_name in self.phase_metrics[phase]:
                    values = self.phase_metrics[phase][metric_name]
                    if values:
                        epochs = [v['epoch'] for v in values if v['epoch'] is not None]
                        metric_values = [v['value'] for v in values if v['epoch'] is not None]
                        
                        if epochs and metric_values:
                            plt.plot(epochs, metric_values, label=f'{phase} - {metric_name}', 
                                   marker='o', markersize=4)
            
            plt.title(f'{metric_name} 指标变化', fontsize=14, fontweight='bold')
            plt.xlabel('Epoch')
            plt.ylabel(metric_name)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 保存图表
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_filename = f"{self.experiment_name}_{metric_name}_{timestamp}.png"
            plot_filepath = self.output_dir / plot_filename
            
            plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            plot_files[metric_name] = str(plot_filepath)
        
        return plot_files


class PerformanceAnalyzer:
    """性能分析器"""
    
    def __init__(self, metrics_tracker: MetricsTracker):
        """
        初始化性能分析器
        
        Args:
            metrics_tracker: 指标跟踪器
        """
        self.metrics_tracker = metrics_tracker
        self.logger = get_logger("performance_analyzer")
    
    def analyze_learning_curves(self) -> Dict[str, Any]:
        """
        分析学习曲线
        
        Returns:
            学习曲线分析结果
        """
        analysis = {}
        
        # 分析训练损失
        loss_data = self.metrics_tracker.get_metric_series('training_loss')
        if not loss_data.empty:
            analysis['training_loss'] = {
                'convergence_rate': self._calculate_convergence_rate(loss_data['value']),
                'final_loss': loss_data['value'].iloc[-1],
                'loss_reduction': loss_data['value'].iloc[0] - loss_data['value'].iloc[-1]
            }
        
        # 分析任务性能
        perf_data = self.metrics_tracker.get_metric_series('task_performance')
        if not perf_data.empty:
            analysis['task_performance'] = {
                'final_performance': perf_data['value'].iloc[-1],
                'performance_improvement': perf_data['value'].iloc[-1] - perf_data['value'].iloc[0],
                'stability': 1 - perf_data['value'].std()  # 稳定性指标
            }
        
        return analysis
    
    def analyze_transfer_learning(self) -> Dict[str, Any]:
        """
        分析迁移学习效果
        
        Returns:
            迁移学习分析结果
        """
        analysis = {}
        
        # 分析遗忘程度
        forgetting_data = self.metrics_tracker.get_metric_series('avg_forgetting', 'phase2')
        if not forgetting_data.empty:
            analysis['forgetting'] = {
                'total_forgetting': forgetting_data['value'].sum(),
                'max_forgetting': forgetting_data['value'].max(),
                'forgetting_stability': 1 - forgetting_data['value'].std()
            }
        
        # 分析新任务学习效果
        new_perf_data = self.metrics_tracker.get_metric_series('avg_new_performance', 'phase2')
        if not new_perf_data.empty:
            analysis['new_task_learning'] = {
                'final_performance': new_perf_data['value'].iloc[-1],
                'learning_speed': new_perf_data['value'].iloc[-1] / len(new_perf_data),
                'learning_curve': new_perf_data['value'].tolist()
            }
        
        return analysis
    
    def _calculate_convergence_rate(self, values: pd.Series) -> float:
        """
        计算收敛率
        
        Args:
            values: 数值序列
        
        Returns:
            收敛率
        """
        if len(values) < 2:
            return 0.0
        
        # 计算相邻值的差异
        diffs = values.diff().abs()
        
        # 计算收敛率（差异的减少趋势）
        if diffs.sum() > 0:
            return 1 - (diffs.iloc[-10:].mean() / diffs.iloc[:10].mean())
        else:
            return 0.0
    
    def generate_report(self) -> str:
        """
        生成分析报告
        
        Returns:
            报告文件路径
        """
        report = {
            'experiment_name': self.metrics_tracker.experiment_name,
            'analysis_time': datetime.now().isoformat(),
            'learning_curves': self.analyze_learning_curves(),
            'transfer_learning': self.analyze_transfer_learning(),
            'summary_statistics': self.metrics_tracker.calculate_summary_statistics()
        }
        
        # 保存报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"{self.metrics_tracker.experiment_name}_analysis_{timestamp}.json"
        report_filepath = self.metrics_tracker.output_dir / report_filename
        
        with open(report_filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"分析报告生成: {report_filepath}")
        return str(report_filepath)