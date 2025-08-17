"""
错误分析可视化脚本

分析训练日志中的错误数据，生成美观的可视化图表来展示：
1. 各任务错误率随时间变化
"""

import os
import re
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import pandas as pd
from datetime import datetime
import argparse




class ErrorAnalysisVisualizer:
    """错误分析可视化器"""
    
    def __init__(self, log_file_path: str, output_dir: str = "temp"):
        """
        初始化可视化器
        
        Args:
            log_file_path: 训练日志文件路径
            output_dir: 输出目录
        """
        self.log_file_path = log_file_path
        self.output_dir = output_dir
        self.data = self._parse_log_file()
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # English labels mapping
        self.english_labels = {
            'pro_saccade': 'Pro Saccade',
            'anti_saccade': 'Anti Saccade', 
            'delay_pro': 'Delay Pro',
            'delay_anti': 'Delay Anti'
        }
    
    def _parse_log_file(self) -> Dict:
        """解析日志文件，提取错误分析数据"""
        data = {
            'epochs': [],
            'tasks': defaultdict(list),
            'error_rates': defaultdict(lambda: defaultdict(list)),
            'performance': defaultdict(list),
            'loss': [],
            'phase': 'phase1'
        }
        
        current_epoch = None
        current_task = None
        phase1_epochs = 0
        
        with open(self.log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                # 检测阶段变化
                if '=== 第二阶段 训练开始 ===' in line:
                    data['phase'] = 'phase2'
                    phase1_epochs = len(data['epochs'])
                
                # 提取epoch信息
                epoch_match = re.search(r'Epoch (\d+)/(\d+)', line)
                if epoch_match:
                    current_epoch = int(epoch_match.group(1))
                    if data['phase'] == 'phase2':
                        # 第二阶段：epoch编号从1000开始
                        adjusted_epoch = 1000 + current_epoch
                    else:
                        # 第一阶段：直接使用epoch编号
                        adjusted_epoch = current_epoch
                    data['epochs'].append(adjusted_epoch)
                
                # 重置current_epoch，避免跨阶段污染
                if '=== 第二阶段 训练开始 ===' in line:
                    current_epoch = None
                
                # 提取损失信息
                loss_match = re.search(r'损失: ([\d.]+)', line)
                if loss_match and current_epoch is not None:
                    data['loss'].append(float(loss_match.group(1)))
                
                # 提取任务错误分析
                task_match = re.search(r'任务 (\w+) 错误分析:', line)
                if task_match:
                    current_task = task_match.group(1)
                    if current_epoch is not None:
                        # 使用调整后的epoch编号
                        if data['phase'] == 'phase2':
                            adjusted_epoch = 1000 + current_epoch
                        else:
                            adjusted_epoch = current_epoch
                        data['tasks'][current_task].append(adjusted_epoch)
                
                # 提取错误率数据
                if current_task and current_epoch is not None:
                    # 总试验数
                    total_match = re.search(r'总试验数: (\d+)', line)
                    if total_match:
                        data['error_rates'][current_task]['total_trials'].append(int(total_match.group(1)))
                    
                    # 正确率
                    correct_match = re.search(r'正确率: ([\d.]+)', line)
                    if correct_match:
                        data['error_rates'][current_task]['correct_rate'].append(float(correct_match.group(1)))
                    
                    # 注视阶段没有注视错误率
                    fixation_match = re.search(r'注视阶段没有注视错误率: ([\d.]+)', line)
                    if fixation_match:
                        data['error_rates'][current_task]['fixation_error_rate'].append(float(fixation_match.group(1)))
                    
                    # 眼跳方向错误率
                    direction_match = re.search(r'眼跳方向错误率: ([\d.]+)', line)
                    if direction_match:
                        data['error_rates'][current_task]['direction_error_rate'].append(float(direction_match.group(1)))
                    
                    # 眼跳阶段仍注视错误率
                    both_match = re.search(r'眼跳阶段仍注视错误率: ([\d.]+)', line)
                    if both_match:
                        data['error_rates'][current_task]['both_error_rate'].append(float(both_match.group(1)))
                    
                    # 总错误率
                    total_error_match = re.search(r'总错误率: ([\d.]+)', line)
                    if total_error_match:
                        data['error_rates'][current_task]['total_error_rate'].append(float(total_error_match.group(1)))
                
                # 提取性能指标
                perf_match = re.search(r'性能指标: (.+)', line)
                if perf_match and current_epoch is not None:
                    perf_str = perf_match.group(1)
                    perf_parts = perf_str.split(', ')
                    for part in perf_parts:
                        task, value = part.split(': ')
                        data['performance'][task].append(float(value))
        
        return data
    
    def plot_error_rates_over_time(self, save_plot: bool = True) -> None:
        """绘制错误率随时间变化的图表"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        tasks = list(self.data['error_rates'].keys())
        
        for i, task in enumerate(tasks):
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            epochs = self.data['tasks'][task]
            error_data = self.data['error_rates'][task]
            
            if epochs and error_data['total_error_rate']:
                ax.plot(epochs, error_data['total_error_rate'], 
                       label='Total Error Rate', marker='o')
                ax.plot(epochs, error_data['fixation_error_rate'], 
                       label='Fixation Error Rate', marker='s')
                ax.plot(epochs, error_data['direction_error_rate'], 
                       label='Direction Error Rate', marker='^')
                ax.plot(epochs, error_data['both_error_rate'], 
                       label='Saccade Still Fixating Error', marker='d')
                
                task_name = self.english_labels.get(task, task)
                ax.set_title(f'{task_name} Task')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Error Rate')
                ax.legend()
                ax.grid(True)
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig(os.path.join(self.output_dir, 'error_rates_over_time.png'))
        
        plt.show()
    
    def generate_summary_report(self, save_report: bool = True) -> Dict:
        """生成错误分析总结报告"""
        report = {
            'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'log_file': self.log_file_path,
            'total_epochs': len(self.data['epochs']),
            'tasks_analyzed': list(self.data['error_rates'].keys()),
            'final_performance': {},
            'final_error_rates': {},
            'training_summary': {}
        }
        
        # 最终性能
        for task, performance in self.data['performance'].items():
            if performance:
                report['final_performance'][task] = performance[-1]
        
        # 最终错误率
        for task in self.data['error_rates'].keys():
            error_data = self.data['error_rates'][task]
            if error_data['total_error_rate']:
                report['final_error_rates'][task] = {
                    'total_error_rate': error_data['total_error_rate'][-1],
                    'fixation_error_rate': error_data['fixation_error_rate'][-1],
                    'direction_error_rate': error_data['direction_error_rate'][-1],
                    'both_error_rate': error_data['both_error_rate'][-1],
                    'correct_rate': error_data['correct_rate'][-1]
                }
        
        # 训练总结
        if self.data['loss']:
            report['training_summary'] = {
                'initial_loss': self.data['loss'][0],
                'final_loss': self.data['loss'][-1],
                'loss_improvement': self.data['loss'][0] - self.data['loss'][-1],
                'best_performance_task': max(report['final_performance'].items(), key=lambda x: x[1])[0] if report['final_performance'] else None,
                'worst_performance_task': min(report['final_performance'].items(), key=lambda x: x[1])[0] if report['final_performance'] else None
            }
        
        if save_report:
            report_path = os.path.join(self.output_dir, 'error_analysis_report.json')
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
        
        return report
    
    def run_full_analysis(self) -> None:
        """运行完整的错误分析可视化"""
        print("开始错误分析可视化...")
        print(f"分析日志文件: {self.log_file_path}")
        print(f"输出目录: {self.output_dir}")
        

        
        # 生成图表
        self.plot_error_rates_over_time()
        
        # 生成报告
        report = self.generate_summary_report()
        
        print("错误分析可视化完成！")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='错误分析可视化工具')
    parser.add_argument('--log_file', default='../checkpoints/random_experiment_20250814_104210/logs/training_random_experiment_20250814_104210_20250814_104210.log', 
                       help='训练日志文件路径')
    parser.add_argument('--output_dir', default='../temp', help='输出目录')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.log_file):
        print(f"错误: 日志文件 {args.log_file} 不存在")
        print("请检查文件路径或使用 --log_file 参数指定正确的日志文件路径")
        return
    
    # 创建可视化器
    visualizer = ErrorAnalysisVisualizer(args.log_file, args.output_dir)
    
    # 运行分析
    visualizer.run_full_analysis()


if __name__ == "__main__":
    # 检查是否有命令行参数
    import sys
    if len(sys.argv) == 1:
        # 没有参数，直接运行默认分析
        print("使用默认设置运行错误分析可视化...")
        log_file = "../checkpoints/random_experiment_20250814_104210/logs/training_random_experiment_20250814_104210_20250814_104210.log"
        output_dir = "../temp"
        
        if os.path.exists(log_file):
            visualizer = ErrorAnalysisVisualizer(log_file, output_dir)
            visualizer.run_full_analysis()
        else:
            print(f"错误: 默认日志文件 {log_file} 不存在")
            print("请使用 --log_file 参数指定正确的日志文件路径")
    else:
        # 有参数，使用argparse
        main()
