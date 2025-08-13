#!/usr/bin/env python3
"""
工程化训练脚本 - 使用统一的日志、配置管理和命令行接口
"""

import sys
import os
import torch
import numpy as np
import random
from pathlib import Path
from datetime import datetime

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.logger import setup_logger, get_logger, TrainingLogger
from utils.config import ConfigManager, ExperimentConfig
from utils.cli import CLIInterface
from utils.visualization import VisualizationManager
from utils.metrics import MetricsTracker, PerformanceAnalyzer

from task import generate_trials
from model import CustomSaccadeModel, create_model_from_trial, prepare_trial_data, prepare_cost_mask
from dataset import load_stage_data, sample_params_for_stage, get_stage_info


class Trainer:
    """工程化训练器"""
    
    def __init__(self, config: ExperimentConfig, output_dir: str = "checkpoints"):
        """
        初始化训练器
        
        Args:
            config: 实验配置
            output_dir: 输出目录
        """
        self.config = config
        
        # 创建带时间戳的输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = config.experiment_name.replace(" ", "_").replace("/", "_")
        self.output_dir = Path(output_dir) / f"{experiment_name}_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化组件
        self.logger = TrainingLogger(str(self.output_dir / "logs"), config.experiment_name)
        self.visualizer = VisualizationManager(str(self.output_dir / "figures"))
        self.metrics_tracker = MetricsTracker(config.experiment_name, str(self.output_dir / "metrics"))
        
        # 记录配置
        self.logger.log_hyperparameters(config.get_hp())
        
        # 初始化模型
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """初始化模型"""
        hp = self.config.get_hp()
        
        # 创建初始trial
        trial = generate_trials('pro_saccade', hp, 'random', batch_size=1)
        # 为模型创建专门的子目录
        model_dir = str(self.output_dir / "model")
        self.model = create_model_from_trial(trial, model_dir=model_dir)
        
        self.logger.logger.info("模型初始化完成")
    
    def train_random_mode(self, detailed_analysis: bool = False) -> dict:
        """
        随机数据训练模式（支持持续学习）
        
        Args:
            detailed_analysis: 是否启用详细错误分析
            
        Returns:
            训练结果字典
        """
        self.logger.logger.info("开始随机数据训练模式")
        
        hp = self.config.get_hp()
        phase1_epochs = self.config.hyperparameters.phase1_epochs
        phase2_epochs = self.config.hyperparameters.phase2_epochs
        batch_size = self.config.hyperparameters.batch_size
        
        # 检查是否启用持续学习
        continual_learning_enabled = self.config.hyperparameters.continual_learning.get('enabled', False)
        if continual_learning_enabled:
            self.logger.logger.info("启用持续学习模式")
            self.logger.logger.info(f"  - 智能突触参数: c={self.config.hyperparameters.continual_learning['c_intsyn']}")
            self.logger.logger.info(f"  - 智能突触参数: ksi={self.config.hyperparameters.continual_learning['ksi_intsyn']}")
            self.logger.logger.info(f"  - EWC参数: lambda={self.config.hyperparameters.continual_learning['ewc_lambda']}")
        else:
            self.logger.logger.info("未启用持续学习模式")
        
        # 第一阶段训练
        phase1_tasks = ['pro_saccade', 'anti_saccade', 'delay_pro']
        self.logger.log_training_start("第一阶段", phase1_epochs, phase1_tasks)
        
        phase1_losses = []
        phase1_evaluations = []
        
        for epoch in range(phase1_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            # 随机打乱任务顺序
            shuffled_tasks = random.sample(phase1_tasks, len(phase1_tasks))
            
            for task_name in shuffled_tasks:
                try:
                    trial = generate_trials(task_name, hp, 'random', batch_size=batch_size)
                    x, y, y_loc = prepare_trial_data(trial)
                    c_mask = prepare_cost_mask(trial)
                    
                    loss = self.model.train_step(x, y, c_mask)
                    epoch_loss += loss
                    num_batches += 1
                    
                except Exception as e:
                    self.logger.log_error(e, f"第一阶段训练任务 {task_name}")
                    continue
            
            if num_batches > 0:
                avg_loss = epoch_loss / num_batches
                phase1_losses.append(avg_loss)
                
                self.metrics_tracker.log_training_loss("phase1", epoch, avg_loss)
                
                if epoch % self.config.hyperparameters.eval_interval == 0:
                    # 使用详细错误分析进行评估
                    eval_results = self.evaluate_tasks(phase1_tasks, batch_size, detailed_analysis=detailed_analysis)
                    phase1_evaluations.append(eval_results)
                    
                    # 过滤掉非数字字段，只保留任务性能指标用于日志
                    performance_metrics = {k: v for k, v in eval_results.items() 
                                         if k != 'error_analysis' and isinstance(v, (int, float))}
                    
                    self.logger.log_epoch("第一阶段", epoch, phase1_epochs, avg_loss, performance_metrics)
        
        # 第一阶段结束，保存模型
        self.model.save('model_phase1.pth')
        self.logger.logger.info("第一阶段训练完成，模型已保存")
        
        # 第二阶段：迁移学习
        phase2_tasks = ['delay_anti']
        self.logger.log_training_start("第二阶段", phase2_epochs, phase2_tasks)
        
        # 调整学习率（迁移学习）
        original_lr = self.model.optimizer.param_groups[0]['lr']
        transfer_lr = original_lr * self.config.hyperparameters.transfer_lr_factor
        self.model.optimizer.param_groups[0]['lr'] = transfer_lr
        self.logger.logger.info(f"迁移学习：学习率从 {original_lr} 调整为 {transfer_lr}")
        
        # 持续学习：开始新任务
        if continual_learning_enabled:
            self.model.start_new_task('delay_anti')
        
        phase2_losses = []
        phase2_evaluations = []
        current_indices = {task: 0 for task in phase2_tasks}
        
        for epoch in range(phase2_epochs):
            # 设置当前epoch和phase信息
            self.current_epoch = epoch
            self.current_phase = "phase2"
            
            epoch_loss = 0.0
            num_batches = 0
            
            # 迁移学习策略：主要训练新任务，偶尔温习老任务以防止遗忘
            if epoch % 5 == 0:  # 每5个epoch温习一次老任务
                # 温习老任务
                review_tasks = ['pro_saccade', 'anti_saccade', 'delay_pro']
                self.logger.logger.info(f"Epoch {epoch}: 温习老任务 {review_tasks}")
                for task_name in review_tasks:
                    try:
                        trial = generate_trials(task_name, hp, 'random', batch_size=batch_size)
                        x, y, y_loc = prepare_trial_data(trial)
                        c_mask = prepare_cost_mask(trial)
                        
                        loss = self.model.train_step(x, y, c_mask)
                        epoch_loss += loss
                        num_batches += 1
                        
                    except Exception as e:
                        self.logger.log_error(e, f"温习任务 {task_name}")
                        continue
            else:
                # 主要训练新任务
                self.logger.logger.info(f"Epoch {epoch}: 训练新任务 {phase2_tasks}")
                for task_name in phase2_tasks:
                    try:
                        trial = generate_trials(task_name, hp, 'random', batch_size=batch_size)
                        x, y, y_loc = prepare_trial_data(trial)
                        c_mask = prepare_cost_mask(trial)
                        
                        loss = self.model.train_step(x, y, c_mask)
                        epoch_loss += loss
                        num_batches += 1
                        
                    except Exception as e:
                        self.logger.log_error(e, f"第二阶段训练任务 {task_name}")
                        continue
            
            if num_batches > 0:
                avg_loss = epoch_loss / num_batches
                phase2_losses.append(avg_loss)
                
                self.metrics_tracker.log_training_loss("phase2", epoch, avg_loss)
                
                if epoch % self.config.hyperparameters.eval_interval == 0:
                    # 评估所有任务（包括已有任务和新任务），使用详细错误分析
                    all_tasks = ['pro_saccade', 'anti_saccade', 'delay_pro', 'delay_anti']
                    eval_results = self.evaluate_tasks(all_tasks, batch_size, detailed_analysis=detailed_analysis)
                    phase2_evaluations.append(eval_results)
                    
                    # 过滤掉非数字字段，只保留任务性能指标用于日志
                    performance_metrics = {k: v for k, v in eval_results.items() 
                                         if k != 'error_analysis' and isinstance(v, (int, float))}
                    
                    self.logger.log_epoch("第二阶段", epoch, phase2_epochs, avg_loss, performance_metrics)
        
        # 训练完成，保存最终模型
        self.model.save('model_final.pth')
        final_model_path = str(self.output_dir / "model" / "model_final.pth")
        self.logger.log_model_save(final_model_path)
        
        # 保存指标
        self.metrics_tracker.save_metrics()
        
        # 返回训练结果
        results = {
            'phase1_losses': phase1_losses,
            'phase1_evaluations': phase1_evaluations,
            'phase2_losses': phase2_losses,
            'phase2_evaluations': phase2_evaluations,
            'model_path': final_model_path,
            'continual_learning_enabled': continual_learning_enabled
        }
        
        return results
    
    def train_psychometric_mode(self, data_dir: str = "data", subject: str = "DD", detailed_analysis: bool = False) -> dict:
        """
        真实数据训练模式
        
        Args:
            data_dir: 数据目录
            subject: 被试名称
            detailed_analysis: 是否启用详细错误分析
        
        Returns:
            训练结果字典
        """
        self.logger.logger.info(f"开始真实数据训练模式，被试: {subject}")
        
        hp = self.config.get_hp()
        
        # 获取数据信息
        try:
            phase1_info = get_stage_info(data_dir, 'phase1', subject)
            phase2_info = get_stage_info(data_dir, 'phase2', subject)
            self.logger.logger.info(f"第一阶段可用任务: {list(phase1_info.keys())}")
            self.logger.logger.info(f"第二阶段可用任务: {list(phase2_info.keys())}")
        except Exception as e:
            self.logger.log_error(e, "获取数据信息")
            return {}
        
        # 第一阶段训练
        phase1_tasks = list(phase1_info.keys())
        if not phase1_tasks:
            self.logger.logger.error("第一阶段没有可用任务")
            return {}
        
        phase1_epochs = self.config.hyperparameters.phase1_epochs
        batch_size = self.config.hyperparameters.batch_size
        
        self.logger.log_training_start("第一阶段", phase1_epochs, phase1_tasks)
        
        phase1_losses = []
        phase1_evaluations = []
        current_indices = {task: 0 for task in phase1_tasks}
        
        for epoch in range(phase1_epochs):
            # 设置当前epoch和phase信息
            self.current_epoch = epoch
            self.current_phase = "phase1"
            
            epoch_loss = 0.0
            num_batches = 0
            
            for task_name in phase1_tasks:
                try:
                    params = sample_params_for_stage(
                        data_dir=data_dir,
                        stage='phase1',
                        rule_name=task_name,
                        max_samples=batch_size,
                        shuffle=False,
                        subject=subject,
                        start_idx=current_indices[task_name]
                    )
                    
                    if not params:
                        continue
                    
                    trial = generate_trials(task_name, hp, 'psychometric', params=params)
                    x, y, y_loc = prepare_trial_data(trial)
                    c_mask = prepare_cost_mask(trial)
                    
                    loss = self.model.train_step(x, y, c_mask)
                    epoch_loss += loss
                    num_batches += 1
                    
                    current_indices[task_name] += len(params)
                    
                except Exception as e:
                    self.logger.log_error(e, f"第一阶段训练任务 {task_name}")
                    continue
            
            if num_batches > 0:
                avg_loss = epoch_loss / num_batches
                phase1_losses.append(avg_loss)
                
                self.metrics_tracker.log_training_loss("phase1", epoch, avg_loss)
                
                if epoch % self.config.hyperparameters.eval_interval == 0:
                    eval_results = self.evaluate_tasks(phase1_tasks, batch_size, detailed_analysis=detailed_analysis)
                    phase1_evaluations.append(eval_results)
                    
                    # 过滤掉非数字字段，只保留任务性能指标用于日志
                    performance_metrics = {k: v for k, v in eval_results.items() 
                                         if k != 'error_analysis' and isinstance(v, (int, float))}
                    
                    self.logger.log_epoch("第一阶段", epoch, phase1_epochs, avg_loss, performance_metrics)
                    
                    for task_name, performance in eval_results.items():
                        if task_name != 'error_analysis' and isinstance(performance, (int, float)):
                            self.metrics_tracker.log_task_performance("phase1", epoch, task_name, performance)
        
        # 第一阶段完成
        final_phase1_eval = self.evaluate_tasks(phase1_tasks, batch_size, detailed_analysis=detailed_analysis)
        # 过滤掉非数字字段，只保留任务性能指标用于日志
        final_performance_metrics = {k: v for k, v in final_phase1_eval.items() 
                                   if k != 'error_analysis' and isinstance(v, (int, float))}
        self.logger.log_training_end("第一阶段", phase1_losses[-1], final_performance_metrics)
        
        self.model.save("model_phase1.pth")
        phase1_model_path = str(self.output_dir / "model" / "model_phase1.pth")
        self.logger.log_model_save(phase1_model_path)
        
        # 第二阶段：迁移学习
        phase2_tasks = list(phase2_info.keys())
        if not phase2_tasks:
            self.logger.logger.info("第二阶段没有可用任务，跳过迁移学习")
            return {
                'phase1_losses': phase1_losses,
                'phase1_evaluations': phase1_evaluations,
                'phase2_losses': [],
                'phase2_evaluations': [],
                'model_path': phase1_model_path
            }
        
        # 识别新增任务
        new_tasks = [task for task in phase2_tasks if task not in phase1_tasks]
        existing_tasks = [task for task in phase2_tasks if task in phase1_tasks]
        
        self.logger.logger.info(f"已有任务: {existing_tasks}")
        self.logger.logger.info(f"新增任务: {new_tasks}")
        
        if not new_tasks:
            self.logger.logger.info("第二阶段没有新增任务，跳过迁移学习")
            return {
                'phase1_losses': phase1_losses,
                'phase1_evaluations': phase1_evaluations,
                'phase2_losses': [],
                'phase2_evaluations': [],
                'model_path': phase1_model_path
            }
        
        # 迁移学习参数
        transfer_epochs = self.config.hyperparameters.phase2_epochs
        transfer_batch_size = batch_size // 2  # 较小的batch
        
        # 调整学习率
        original_lr = self.model.optimizer.param_groups[0]['lr']
        transfer_lr = original_lr * self.config.hyperparameters.transfer_lr_factor
        self.model.optimizer.param_groups[0]['lr'] = transfer_lr
        
        self.logger.logger.info(f"迁移学习学习率: {original_lr:.6f} -> {transfer_lr:.6f}")
        
        phase2_losses = []
        phase2_evaluations = []
        current_indices = {task: 0 for task in phase2_tasks}
        
        for epoch in range(transfer_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            # 迁移学习策略：主要训练新任务，偶尔温习老任务以防止遗忘
            if epoch % 5 == 0:  # 每5个epoch温习一次老任务
                # 温习老任务
                tasks_to_train = existing_tasks
                self.logger.logger.info(f"Epoch {epoch}: 温习老任务 {tasks_to_train}")
            else:
                # 主要训练新任务
                tasks_to_train = new_tasks
                self.logger.logger.info(f"Epoch {epoch}: 训练新任务 {tasks_to_train}")
            
            for task_name in tasks_to_train:
                try:
                    params = sample_params_for_stage(
                        data_dir=data_dir,
                        stage='phase2',
                        rule_name=task_name,
                        max_samples=transfer_batch_size,
                        shuffle=False,
                        subject=subject,
                        start_idx=current_indices[task_name]
                    )
                    
                    if not params:
                        continue
                    
                    trial = generate_trials(task_name, hp, 'psychometric', params=params)
                    x, y, y_loc = prepare_trial_data(trial)
                    c_mask = prepare_cost_mask(trial)
                    
                    loss = self.model.train_step(x, y, c_mask)
                    epoch_loss += loss
                    num_batches += 1
                    
                    current_indices[task_name] += len(params)
                    
                except Exception as e:
                    self.logger.log_error(e, f"第二阶段训练任务 {task_name}")
                    continue
            
            if num_batches > 0:
                avg_loss = epoch_loss / num_batches
                phase2_losses.append(avg_loss)
                
                self.metrics_tracker.log_training_loss("phase2", epoch, avg_loss)
                
                if epoch % 5 == 0:
                    eval_results = self.evaluate_tasks(existing_tasks + new_tasks, batch_size, detailed_analysis=detailed_analysis)
                    phase2_evaluations.append(eval_results)
                    
                    self.logger.log_epoch("第二阶段", epoch, transfer_epochs, avg_loss)
                    # 分离已有任务和新任务的性能用于日志记录
                    existing_perf = {task: eval_results.get(task, 0) for task in existing_tasks 
                                   if task in eval_results and isinstance(eval_results.get(task, 0), (int, float))}
                    new_perf = {task: eval_results.get(task, 0) for task in new_tasks 
                              if task in eval_results and isinstance(eval_results.get(task, 0), (int, float))}
                    self.logger.log_transfer_learning(existing_perf, new_perf)
                    
                    self.metrics_tracker.log_transfer_learning_metrics("phase2", epoch, existing_perf, new_perf)
        
        # 第二阶段完成
        final_eval = self.evaluate_tasks(existing_tasks + new_tasks, batch_size, detailed_analysis=detailed_analysis)
        self.logger.log_training_end("第二阶段", phase2_losses[-1])
        # 分离已有任务和新任务的性能用于日志记录
        existing_perf = {task: final_eval.get(task, 0) for task in existing_tasks 
                       if task in final_eval and isinstance(final_eval.get(task, 0), (int, float))}
        new_perf = {task: final_eval.get(task, 0) for task in new_tasks 
                   if task in final_eval and isinstance(final_eval.get(task, 0), (int, float))}
        self.logger.log_transfer_learning(existing_perf, new_perf)
        
        self.model.save("model_final.pth")
        final_model_path = str(self.output_dir / "model" / "model_final.pth")
        self.logger.log_model_save(final_model_path)
        
        # 保存指标
        self.metrics_tracker.save_metrics()
        
        # 返回训练结果
        results = {
            'phase1_losses': phase1_losses,
            'phase1_evaluations': phase1_evaluations,
            'phase2_losses': phase2_losses,
            'phase2_evaluations': phase2_evaluations,
            'model_path': final_model_path
        }
        
        return results
    
    def evaluate_tasks(self, task_names: list, batch_size: int = 32, num_trials: int = 100, 
                      detailed_analysis: bool = False) -> dict:
        """评估指定任务的性能
        
        Args:
            task_names: 要评估的任务名称列表
            batch_size: 批次大小
            num_trials: 评估试验数量
            detailed_analysis: 是否进行详细错误分析
            
        Returns:
            包含每个任务性能的字典
        """
        results = {}
        error_analysis_results = {}
        hp = self.config.get_hp()
        
        for task_name in task_names:
            try:
                performances = []
                error_analyses = []
                num_batches = max(1, num_trials // batch_size)
                
                for _ in range(num_batches):
                    trial = generate_trials(task_name, hp, 'random', batch_size=batch_size)
                    x, y, y_loc = prepare_trial_data(trial)
                    
                    self.model.eval()
                    with torch.no_grad():
                        y_hat = self.model(x)
                        
                        if detailed_analysis:
                            # 使用详细错误分析
                            analysis = self.model.evaluate(y_hat, y_loc, detailed_analysis=True)
                            performance = analysis['performance_metrics']['average_performance']
                            error_analyses.append(analysis['error_analysis'])
                        else:
                            # 使用标准评估
                            performance = self.model.evaluate(y_hat, y_loc, detailed_analysis=False)
                    
                    performances.append(performance)
                
                results[task_name] = np.mean(performances)
                
                # 如果进行了详细分析，合并错误分析结果
                if detailed_analysis and error_analyses:
                    # 合并所有batch的错误分析
                    combined_error_analysis = self._combine_error_analyses(error_analyses)
                    error_analysis_results[task_name] = combined_error_analysis
                    
                    # 记录错误分析到日志
                    self._log_error_analysis(task_name, combined_error_analysis)
                    
                    # 记录错误分析到metrics（如果有epoch信息）
                    if hasattr(self, 'current_epoch') and hasattr(self, 'current_phase'):
                        self.metrics_tracker.log_error_analysis(
                            self.current_phase, self.current_epoch, task_name, combined_error_analysis
                        )
                
            except Exception as e:
                self.logger.log_error(e, f"评估任务 {task_name}")
                results[task_name] = 0.0
        
        # 如果进行了详细分析，将错误分析结果添加到返回结果中
        if detailed_analysis:
            results['error_analysis'] = error_analysis_results
        
        return results
    
    def _combine_error_analyses(self, error_analyses: list) -> dict:
        """合并多个错误分析结果
        
        Args:
            error_analyses: 错误分析结果列表
            
        Returns:
            合并后的错误分析结果
        """
        if not error_analyses:
            return {}
        
        # 初始化合并结果
        combined = {
            'total_trials': 0,
            'fixation_errors': 0,
            'direction_errors': 0,
            'both_errors': 0,
            'correct_trials': 0
        }
        
        # 合并所有分析结果
        for analysis in error_analyses:
            combined['total_trials'] += analysis['total_trials']
            combined['fixation_errors'] += analysis['fixation_errors']
            combined['direction_errors'] += analysis['direction_errors']
            combined['both_errors'] += analysis['both_errors']
            combined['correct_trials'] += analysis['correct_trials']
        
        # 重新计算错误率
        total_trials = combined['total_trials']
        if total_trials > 0:
            combined['total_error_rate'] = (combined['fixation_errors'] + combined['direction_errors'] + combined['both_errors']) / total_trials
            combined['fixation_error_rate'] = combined['fixation_errors'] / total_trials
            combined['direction_error_rate'] = combined['direction_errors'] / total_trials
            combined['both_error_rate'] = combined['both_errors'] / total_trials
            combined['correct_rate'] = combined['correct_trials'] / total_trials
        else:
            combined['total_error_rate'] = 0.0
            combined['fixation_error_rate'] = 0.0
            combined['direction_error_rate'] = 0.0
            combined['both_error_rate'] = 0.0
            combined['correct_rate'] = 0.0
        
        return combined
    
    def _log_error_analysis(self, task_name: str, error_analysis: dict):
        """记录错误分析结果到日志
        
        Args:
            task_name: 任务名称
            error_analysis: 错误分析结果
        """
        self.logger.logger.info(f"任务 {task_name} 错误分析:")
        self.logger.logger.info(f"  - 总试验数: {error_analysis.get('total_trials', 0)}")
        
        # 安全地格式化错误率，确保它们是数字
        correct_rate = error_analysis.get('correct_rate', 0.0)
        fixation_error_rate = error_analysis.get('fixation_error_rate', 0.0)
        direction_error_rate = error_analysis.get('direction_error_rate', 0.0)
        both_error_rate = error_analysis.get('both_error_rate', 0.0)
        total_error_rate = error_analysis.get('total_error_rate', 0.0)
        
        # 确保所有值都是数字类型
        if isinstance(correct_rate, (int, float)):
            self.logger.logger.info(f"  - 正确率: {correct_rate:.3f}")
        else:
            self.logger.logger.info(f"  - 正确率: {correct_rate}")
            
        if isinstance(fixation_error_rate, (int, float)):
            self.logger.logger.info(f"  - 注视阶段没有注视错误率: {fixation_error_rate:.3f}")
        else:
            self.logger.logger.info(f"  - 注视阶段没有注视错误率: {fixation_error_rate}")
            
        if isinstance(direction_error_rate, (int, float)):
            self.logger.logger.info(f"  - 眼跳方向错误率: {direction_error_rate:.3f}")
        else:
            self.logger.logger.info(f"  - 眼跳方向错误率: {direction_error_rate}")
            
        if isinstance(both_error_rate, (int, float)):
            self.logger.logger.info(f"  - 眼跳阶段仍注视错误率: {both_error_rate:.3f}")
        else:
            self.logger.logger.info(f"  - 眼跳阶段仍注视错误率: {both_error_rate}")
            
        if isinstance(total_error_rate, (int, float)):
            self.logger.logger.info(f"  - 总错误率: {total_error_rate:.3f}")
        else:
            self.logger.logger.info(f"  - 总错误率: {total_error_rate}")
    
    def _generate_visualizations(self, results: dict):
        """生成可视化结果"""
        try:
            # 生成综合结果图表
            self.visualizer.plot_comprehensive_results(results)
            
            # 保存结果摘要
            self.visualizer.save_results_summary(results)
            
            # 生成性能分析报告
            analyzer = PerformanceAnalyzer(self.metrics_tracker)
            analyzer.generate_report()
            
            self.logger.logger.info("所有可视化结果生成完成")
            
        except Exception as e:
            self.logger.log_error(e, "生成可视化结果")


def main():
    """主函数 - 直接使用CLI接口"""
    from utils.cli import CLIInterface
    
    cli = CLIInterface()
    cli.run()


if __name__ == "__main__":
    main()