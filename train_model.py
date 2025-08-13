"""训练自定义眼跳任务的PyTorch LSTM模型"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
from typing import Dict, List, Optional
import matplotlib.pyplot as plt

from task import generate_trials
from model import CustomSaccadeModel, create_model_from_trial, prepare_trial_data, prepare_cost_mask
from dataset import load_stage_data, sample_params_for_stage, get_stage_info

def train_with_random_data(save_dir: str = 'checkpoints') -> Dict:
    """
    使用随机生成的数据进行两阶段训练：
    第一阶段：训练前三个任务（pro_saccade, anti_saccade, delay_pro）
    第二阶段：迁移学习训练delay_anti任务
    """
    print("开始random模式两阶段训练...")
    
    # 默认超参数
    hp = {
        'dt': 20,                  # 时间步长 (20ms)
        'tau': 100,                # 时间常数 (100ms)
        'alpha': 0.2,              # alpha = dt/tau
        'sigma_x': 0.1,            # 输入噪声
        'n_eachring': 32,          # 环单元数
        'n_input': 37,             # 1个注视 + 32个环 + 4个任务信号
        'n_output': 33,            # 1个注视 + 32个环
        'loss_type': 'lsq',        # 损失函数类型
        'rng': np.random.RandomState(0)
    }
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建模型
    trial = generate_trials('pro_saccade', hp, 'random', batch_size=1)
    model = create_model_from_trial(trial, model_dir=save_dir)
    
    # 训练参数
    phase1_epochs = 400  # 第一阶段训练前三个任务
    phase2_epochs = 40  # 第二阶段迁移学习
    batch_size = 32
    
    # 第一阶段：训练前三个任务
    print("\n=== 第一阶段：训练前三个任务 ===")
    phase1_tasks = ['pro_saccade', 'anti_saccade', 'delay_pro']
    phase1_losses = []
    phase1_evaluations = []  # 记录评估结果
    
    for epoch in range(phase1_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for task_name in phase1_tasks:
            try:
                trial = generate_trials(task_name, hp, 'random', batch_size=batch_size)
                x, y, y_loc = prepare_trial_data(trial)
                c_mask = prepare_cost_mask(trial)
                
                loss = model.train_step(x, y, c_mask)
                epoch_loss += loss
                num_batches += 1
            except Exception as e:
                print(f"第一阶段训练任务 {task_name} 时出错: {e}")
                continue
        
        if num_batches > 0:
            avg_loss = epoch_loss / num_batches
            phase1_losses.append(avg_loss)
            
            # 每10个epoch评估一次
            if epoch % 10 == 0:
                print(f"第一阶段 Epoch {epoch}/{phase1_epochs}, 平均损失: {avg_loss:.6f}")
                
                # 评估第一阶段任务性能
                eval_results = evaluate_random_tasks(model, hp, phase1_tasks, batch_size)
                phase1_evaluations.append(eval_results)
                print("  评估结果:", {k: f"{v:.3f}" for k, v in eval_results.items()})
    
    print(f"第一阶段训练完成，最终损失: {phase1_losses[-1]:.6f}")
    
    # 第一阶段最终评估
    final_phase1_eval = evaluate_random_tasks(model, hp, phase1_tasks, batch_size)
    print("第一阶段最终评估:", {k: f"{v:.3f}" for k, v in final_phase1_eval.items()})
    
    model.save("model_phase1.pth")
    
    # 第二阶段：迁移学习任务4
    print("\n=== 第二阶段：迁移学习delay_anti任务 ===")
    phase2_tasks = ['delay_anti']  # 只训练新任务
    phase2_losses = []
    phase2_evaluations = []  # 记录评估结果
    
    # 降低学习率进行迁移学习
    original_lr = model.optimizer.param_groups[0]['lr']
    transfer_lr = original_lr * 0.1  # 降低到原来的1/10
    model.optimizer.param_groups[0]['lr'] = transfer_lr
    print(f"迁移学习学习率: {original_lr} -> {transfer_lr}")
    
    for epoch in range(phase2_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for task_name in phase2_tasks:
            try:
                trial = generate_trials(task_name, hp, 'random', batch_size=batch_size)
                x, y, y_loc = prepare_trial_data(trial)
                c_mask = prepare_cost_mask(trial)
                
                loss = model.train_step(x, y, c_mask)
                epoch_loss += loss
                num_batches += 1
            except Exception as e:
                print(f"第二阶段训练任务 {task_name} 时出错: {e}")
                continue
        
        if num_batches > 0:
            avg_loss = epoch_loss / num_batches
            phase2_losses.append(avg_loss)
            
            # 每5个epoch评估一次
            if epoch % 5 == 0:
                print(f"第二阶段 Epoch {epoch}/{phase2_epochs}, 平均损失: {avg_loss:.6f}")
                
                # 评估迁移学习效果
                eval_results = evaluate_random_transfer_learning(model, hp, phase1_tasks, phase2_tasks, batch_size)
                phase2_evaluations.append(eval_results)
                print("  迁移学习评估:", {k: {sk: f"{sv:.3f}" for sk, sv in v.items()} if isinstance(v, dict) else f"{v:.3f}" for k, v in eval_results.items()})
    
    print(f"第二阶段完成，最终损失: {phase2_losses[-1]:.6f}")
    
    # 最终评估
    final_eval = evaluate_random_transfer_learning(model, hp, phase1_tasks, phase2_tasks, batch_size)
    print("最终迁移学习评估:", {k: {sk: f"{sv:.3f}" for sk, sv in v.items()} if isinstance(v, dict) else f"{v:.3f}" for k, v in final_eval.items()})
    
    model.save("model_final.pth")
    
    return {
        'phase1_losses': phase1_losses,
        'phase1_evaluations': phase1_evaluations,
        'phase2_losses': phase2_losses,
        'phase2_evaluations': phase2_evaluations,
        'model_path': f"{save_dir}/model_final.pth"
    }

def evaluate_random_tasks(model: CustomSaccadeModel, hp: Dict, task_names: List[str], 
                         batch_size: int, num_trials: int = 100) -> Dict[str, float]:
    """
    评估random模式任务的性能
    """
    results = {}
    
    for task_name in task_names:
        try:
            # 生成多个随机评估数据并取平均
            performances = []
            num_batches = max(1, num_trials // batch_size)
            
            for _ in range(num_batches):
                trial = generate_trials(task_name, hp, 'random', batch_size=batch_size)
                x, y, y_loc = prepare_trial_data(trial)
                
                model.eval()
                with torch.no_grad():
                    y_hat = model(x)
                    performance = model.evaluate(y_hat, y_loc)
                
                performances.append(performance)
            
            # 取平均性能
            results[task_name] = np.mean(performances)
            
        except Exception as e:
            print(f"评估任务 {task_name} 时出错: {e}")
            results[task_name] = 0.0
    
    return results

def evaluate_random_transfer_learning(model: CustomSaccadeModel, hp: Dict, 
                                    existing_tasks: List[str], new_tasks: List[str], 
                                    batch_size: int, num_trials: int = 100) -> Dict[str, Dict[str, float]]:
    """
    评估random模式迁移学习效果：比较已有任务和新任务的性能
    """
    results = {
        'existing_tasks': {},
        'new_tasks': {},
        'summary': {}
    }
    
    # 评估已有任务（防止遗忘）
    for task_name in existing_tasks:
        try:
            # 生成多个随机评估数据并取平均
            performances = []
            num_batches = max(1, num_trials // batch_size)
            
            for _ in range(num_batches):
                trial = generate_trials(task_name, hp, 'random', batch_size=batch_size)
                x, y, y_loc = prepare_trial_data(trial)
                
                model.eval()
                with torch.no_grad():
                    y_hat = model(x)
                    performance = model.evaluate(y_hat, y_loc)
                
                performances.append(performance)
            
            results['existing_tasks'][task_name] = np.mean(performances)
        except Exception as e:
            print(f"评估已有任务 {task_name} 时出错: {e}")
            results['existing_tasks'][task_name] = 0.0
    
    # 评估新任务（学习效果）
    for task_name in new_tasks:
        try:
            # 生成多个随机评估数据并取平均
            performances = []
            num_batches = max(1, num_trials // batch_size)
            
            for _ in range(num_batches):
                trial = generate_trials(task_name, hp, 'random', batch_size=batch_size)
                x, y, y_loc = prepare_trial_data(trial)
                
                model.eval()
                with torch.no_grad():
                    y_hat = model(x)
                    performance = model.evaluate(y_hat, y_loc)
                
                performances.append(performance)
            
            results['new_tasks'][task_name] = np.mean(performances)
        except Exception as e:
            print(f"评估新任务 {task_name} 时出错: {e}")
            results['new_tasks'][task_name] = 0.0
    
    # 计算汇总统计
    if results['existing_tasks']:
        results['summary']['avg_existing'] = sum(results['existing_tasks'].values()) / len(results['existing_tasks'])
    if results['new_tasks']:
        results['summary']['avg_new'] = sum(results['new_tasks'].values()) / len(results['new_tasks'])
    
    return results

def train_with_psychometric_data(data_dir: str = 'data', subject: str = 'DD', 
                                save_dir: str = 'checkpoints', mode: str = 'psychometric') -> Dict:
    """
    使用psychometric数据进行两阶段训练：
    第一阶段：使用Task3数据训练所有可用任务
    第二阶段：使用Task4数据进行迁移学习，重点掌握新增的delay_anti任务
    """
    print(f"开始psychometric训练，被试: {subject}")
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 默认超参数
    hp = {
        'dt': 20,                  # 时间步长 (20ms)
        'tau': 100,                # 时间常数 (100ms)
        'alpha': 0.2,              # alpha = dt/tau
        'sigma_x': 0.1,            # 输入噪声
        'n_eachring': 32,          # 环单元数
        'n_input': 37,             # 1个注视 + 32个环 + 4个任务信号
        'n_output': 33,            # 1个注视 + 32个环
        'loss_type': 'lsq',        # 损失函数类型
        'rng': np.random.RandomState(0)
    }
    
    # 获取阶段数据信息
    try:
        phase1_info = get_stage_info(data_dir, 'phase1', subject)
        phase2_info = get_stage_info(data_dir, 'phase2', subject)
        print(f"第一阶段可用任务: {list(phase1_info.keys())}")
        print(f"第二阶段可用任务: {list(phase2_info.keys())}")
    except Exception as e:
        print(f"获取数据信息失败: {e}")
        return {}
    
    # 创建模型
    trial = generate_trials('pro_saccade', hp, 'random', batch_size=1)
    model = create_model_from_trial(trial, hp, model_dir=save_dir)
    model.hp['rng'] = hp['rng']
    model.rng = hp['rng']
    
    # 第一阶段：训练Task3的所有任务
    print("\n=== 第一阶段：训练Task3的所有任务 ===")
    phase1_tasks = list(phase1_info.keys())
    if not phase1_tasks:
        print("第一阶段没有可用任务")
        return {}
    
    # 第一阶段训练参数
    phase1_epochs = 50
    phase1_batch_size = 32
    
    # 第一阶段训练循环
    phase1_losses = []
    phase1_evaluations = []  # 记录评估结果
    current_indices = {task: 0 for task in phase1_tasks}
    
    for epoch in range(phase1_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for task_name in phase1_tasks:
            try:
                # 从Task3数据中采样
                params = sample_params_for_stage(
                    data_dir=data_dir,
                    stage='phase1',
                    rule_name=task_name,
                    max_samples=phase1_batch_size,
                    shuffle=False,
                    subject=subject,
                    start_idx=current_indices[task_name]
                )
                
                if not params:
                    print(f"任务 {task_name} 没有更多数据，跳过")
                    continue
                
                trial = generate_trials(task_name, hp, 'psychometric', params=params)
                x, y, y_loc = prepare_trial_data(trial)
                c_mask = prepare_cost_mask(trial)
                
                loss = model.train_step(x, y, c_mask)
                epoch_loss += loss
                num_batches += 1
                
                # 更新索引
                current_indices[task_name] += len(params)
                
            except Exception as e:
                print(f"第一阶段训练任务 {task_name} 时出错: {e}")
                continue
        
        if num_batches > 0:
            avg_loss = epoch_loss / num_batches
            phase1_losses.append(avg_loss)
            
            # 每10个epoch评估一次
            if epoch % 10 == 0:
                print(f"第一阶段 Epoch {epoch}/{phase1_epochs}, 平均损失: {avg_loss:.6f}")
                
                # 评估第一阶段任务性能
                eval_results = evaluate_phase_tasks(model, hp, data_dir, phase1_tasks, subject, 'phase1')
                phase1_evaluations.append(eval_results)
                print("  评估结果:", {k: f"{v:.3f}" for k, v in eval_results.items()})
    
    print(f"第一阶段训练完成，最终损失: {phase1_losses[-1]:.6f}")
    
    # 第一阶段最终评估
    final_phase1_eval = evaluate_phase_tasks(model, hp, data_dir, phase1_tasks, subject, 'phase1')
    print("第一阶段最终评估:", {k: f"{v:.3f}" for k, v in final_phase1_eval.items()})
    
    model.save("model_phase1.pth")
    
    # 第二阶段：迁移学习，重点掌握新增的delay_anti任务
    print("\n=== 第二阶段：迁移学习，掌握新增的delay_anti任务 ===")
    phase2_tasks = list(phase2_info.keys())
    if not phase2_tasks:
        print("第二阶段没有可用任务，跳过")
        return {
            'phase1_losses': phase1_losses,
            'phase1_evaluations': phase1_evaluations,
            'phase2_losses': [],
            'phase2_evaluations': [],
            'model_path': f"{save_dir}/model_phase1.pth"
        }
    
    # 识别新增任务
    new_tasks = [task for task in phase2_tasks if task not in phase1_tasks]
    existing_tasks = [task for task in phase2_tasks if task in phase1_tasks]
    
    print(f"已有任务: {existing_tasks}")
    print(f"新增任务: {new_tasks}")
    
    if not new_tasks:
        print("第二阶段没有新增任务，跳过迁移学习")
        return {
            'phase1_losses': phase1_losses,
            'phase1_evaluations': phase1_evaluations,
            'phase2_losses': [],
            'phase2_evaluations': [],
            'model_path': f"{save_dir}/model_phase1.pth"
        }
    
    # 迁移学习参数
    transfer_epochs = 20  # 较少的epoch
    transfer_batch_size = 16  # 较小的batch
    
    # 降低学习率进行迁移学习
    original_lr = model.optimizer.param_groups[0]['lr']
    transfer_lr = original_lr * 0.1  # 降低到原来的1/10
    model.optimizer.param_groups[0]['lr'] = transfer_lr
    print(f"迁移学习学习率: {original_lr} -> {transfer_lr}")
    
    # 第二阶段训练循环
    phase2_losses = []
    phase2_evaluations = []  # 记录评估结果
    current_indices = {task: 0 for task in phase2_tasks}
    
    for epoch in range(transfer_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        # 优先训练新增任务，偶尔训练已有任务以防止遗忘
        if epoch % 3 == 0:  # 每3个epoch训练一次已有任务
            tasks_to_train = phase2_tasks
        else:  # 其他epoch主要训练新增任务
            tasks_to_train = new_tasks
        
        for task_name in tasks_to_train:
            try:
                # 从Task4数据中采样
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
                    print(f"任务 {task_name} 没有更多数据，跳过")
                    continue
                
                trial = generate_trials(task_name, hp, 'psychometric', params=params)
                x, y, y_loc = prepare_trial_data(trial)
                c_mask = prepare_cost_mask(trial)
                
                loss = model.train_step(x, y, c_mask)
                epoch_loss += loss
                num_batches += 1
                
                # 更新索引
                current_indices[task_name] += len(params)
                
            except Exception as e:
                print(f"第二阶段训练任务 {task_name} 时出错: {e}")
                continue
        
        if num_batches > 0:
            avg_loss = epoch_loss / num_batches
            phase2_losses.append(avg_loss)
            
            # 每5个epoch评估一次
            if epoch % 5 == 0:
                print(f"迁移学习 Epoch {epoch}/{transfer_epochs}, 平均损失: {avg_loss:.6f}")
                if epoch % 3 == 0:
                    print(f"  训练任务: {tasks_to_train}")
                
                # 评估迁移学习效果
                eval_results = evaluate_transfer_learning_effect(model, hp, data_dir, 
                                                              existing_tasks, new_tasks, subject)
                phase2_evaluations.append(eval_results)
                print("  迁移学习评估:", {k: {sk: f"{sv:.3f}" for sk, sv in v.items()} if isinstance(v, dict) else f"{v:.3f}" for k, v in eval_results.items()})
    
    print(f"迁移学习完成，最终损失: {phase2_losses[-1]:.6f}")
    
    # 最终评估
    final_eval = evaluate_transfer_learning_effect(model, hp, data_dir, 
                                                 existing_tasks, new_tasks, subject)
    print("最终迁移学习评估:", {k: {sk: f"{sv:.3f}" for sk, sv in v.items()} if isinstance(v, dict) else f"{v:.3f}" for k, v in final_eval.items()})
    
    model.save("model_final.pth")
    
    return {
        'phase1_losses': phase1_losses,
        'phase1_evaluations': phase1_evaluations,
        'phase2_losses': phase2_losses,
        'phase2_evaluations': phase2_evaluations,
        'model_path': f"{save_dir}/model_final.pth"
    }

def evaluate_phase_tasks(model: CustomSaccadeModel, hp: Dict, data_dir: str, 
                        task_names: List[str], subject: str, stage: str, 
                        num_trials: int = 100) -> Dict[str, float]:
    """
    评估指定阶段任务的性能
    """
    results = {}
    
    for task_name in task_names:
        try:
            # 从指定阶段采样评估数据
            params = sample_params_for_stage(
                data_dir=data_dir,
                stage=stage,
                rule_name=task_name,
                max_samples=num_trials,
                shuffle=False,
                subject=subject
            )
            
            if not params:
                continue
            
            # 生成评估trial
            trial = generate_trials(task_name, hp, 'psychometric', params=params)
            x, y, y_loc = prepare_trial_data(trial)
            c_mask = prepare_cost_mask(trial)
            
            # 评估
            model.eval()
            with torch.no_grad():
                y_hat = model(x)
                perf = model.evaluate(y_hat, y_loc)
            
            results[task_name] = perf
            
        except Exception as e:
            print(f"评估任务 {task_name} 时出错: {e}")
            continue
    
    return results

def evaluate_transfer_learning_effect(model: CustomSaccadeModel, hp: Dict, data_dir: str,
                                    existing_tasks: List[str], new_tasks: List[str], 
                                    subject: str, num_trials: int = 100) -> Dict[str, Dict[str, float]]:
    """
    评估迁移学习效果：比较已有任务和新任务的性能
    """
    results = {
        'existing_tasks': {},
        'new_tasks': {},
        'summary': {}
    }
    
    # 评估已有任务（防止遗忘）
    for task_name in existing_tasks:
        try:
            # 从Task4数据评估已有任务
            params = sample_params_for_stage(
                data_dir=data_dir,
                stage='phase2',
                rule_name=task_name,
                max_samples=num_trials,
                shuffle=False,
                subject=subject
            )
            
            if params:
                trial = generate_trials(task_name, hp, 'psychometric', params=params)
                x, y, y_loc = prepare_trial_data(trial)
                c_mask = prepare_cost_mask(trial)
                
                model.eval()
                with torch.no_grad():
                    y_hat = model(x)
                    perf = model.evaluate(y_hat, y_loc)
                
                results['existing_tasks'][task_name] = perf
        except Exception as e:
            print(f"评估已有任务 {task_name} 时出错: {e}")
    
    # 评估新任务（学习效果）
    for task_name in new_tasks:
        try:
            # 从Task4数据评估新任务
            params = sample_params_for_stage(
                data_dir=data_dir,
                stage='phase2',
                rule_name=task_name,
                max_samples=num_trials,
                shuffle=False,
                subject=subject
            )
            
            if params:
                trial = generate_trials(task_name, hp, 'psychometric', params=params)
                x, y, y_loc = prepare_trial_data(trial)
                c_mask = prepare_cost_mask(trial)
                
                model.eval()
                with torch.no_grad():
                    y_hat = model(x)
                    perf = model.evaluate(y_hat, y_loc)
                
                results['new_tasks'][task_name] = perf
        except Exception as e:
            print(f"评估新任务 {task_name} 时出错: {e}")
    
    # 计算汇总统计
    if results['existing_tasks']:
        results['summary']['avg_existing'] = sum(results['existing_tasks'].values()) / len(results['existing_tasks'])
    if results['new_tasks']:
        results['summary']['avg_new'] = sum(results['new_tasks'].values()) / len(results['new_tasks'])
    
    return results

def visualize_training_results(results: Dict):
    """
    可视化训练结果和迁移学习效果
    """
    
    # 1. 训练损失曲线
    plt.figure(figsize=(15, 10))
    
    # 第一阶段损失
    plt.subplot(2, 3, 1)
    if results.get('phase1_losses'):
        plt.plot(results['phase1_losses'], label='Phase 1', color='blue')
        plt.title('Phase 1 Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
    
    # 第二阶段损失
    plt.subplot(2, 3, 2)
    if results.get('phase2_losses'):
        plt.plot(results['phase2_losses'], label='Phase 2', color='red')
        plt.title('Phase 2 Transfer Learning Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
    
    # 2. 第一阶段任务性能
    plt.subplot(2, 3, 3)
    if results.get('phase1_evaluations'):
        phase1_evals = results['phase1_evaluations']
        if phase1_evals:
            # 获取任务名称
            task_names = list(phase1_evals[0].keys())
            epochs = list(range(0, len(phase1_evals) * 10, 10))
            
            for task in task_names:
                performances = [eval_result.get(task, 0) for eval_result in phase1_evals]
                plt.plot(epochs, performances, label=task, marker='o')
            
            plt.title('Phase 1 Task Performance')
            plt.xlabel('Epoch')
            plt.ylabel('Performance')
            plt.legend()
            plt.grid(True)
    
    # 3. 迁移学习效果 - 已有任务性能
    plt.subplot(2, 3, 4)
    if results.get('phase2_evaluations'):
        phase2_evals = results['phase2_evaluations']
        if phase2_evals:
            epochs = list(range(0, len(phase2_evals) * 5, 5))
            
            # 已有任务性能
            existing_tasks = list(phase2_evals[0].get('existing_tasks', {}).keys())
            for task in existing_tasks:
                performances = [eval_result.get('existing_tasks', {}).get(task, 0) for eval_result in phase2_evals]
                plt.plot(epochs, performances, label=f'{task} (existing)', marker='s')
            
            plt.title('Existing Tasks Performance (Transfer)')
            plt.xlabel('Epoch')
            plt.ylabel('Performance')
            plt.legend()
            plt.grid(True)
    
    # 4. 迁移学习效果 - 新任务性能
    plt.subplot(2, 3, 5)
    if results.get('phase2_evaluations'):
        phase2_evals = results['phase2_evaluations']
        if phase2_evals:
            epochs = list(range(0, len(phase2_evals) * 5, 5))
            
            # 新任务性能
            new_tasks = list(phase2_evals[0].get('new_tasks', {}).keys())
            for task in new_tasks:
                performances = [eval_result.get('new_tasks', {}).get(task, 0) for eval_result in phase2_evals]
                plt.plot(epochs, performances, label=f'{task} (new)', marker='^', color='red')
            
            plt.title('New Tasks Performance (Transfer)')
            plt.xlabel('Epoch')
            plt.ylabel('Performance')
            plt.legend()
            plt.grid(True)
    
    # 5. 迁移学习汇总
    plt.subplot(2, 3, 6)
    if results.get('phase2_evaluations'):
        phase2_evals = results['phase2_evaluations']
        if phase2_evals:
            epochs = list(range(0, len(phase2_evals) * 5, 5))
            
            # 平均性能
            avg_existing = [eval_result.get('summary', {}).get('avg_existing', 0) for eval_result in phase2_evals]
            avg_new = [eval_result.get('summary', {}).get('avg_new', 0) for eval_result in phase2_evals]
            
            plt.plot(epochs, avg_existing, label='Avg Existing Tasks', marker='o', color='blue')
            plt.plot(epochs, avg_new, label='Avg New Tasks', marker='s', color='red')
            
            plt.title('Transfer Learning Summary')
            plt.xlabel('Epoch')
            plt.ylabel('Average Performance')
            plt.legend()
            plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 选择训练模式
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'random':
        # 运行random模式训练
        results = train_with_random_data(save_dir='checkpoints')
        
        print("random模式训练完成！")
        print(f"第一阶段损失数量: {len(results.get('phase1_losses', []))}")
        print(f"第二阶段损失数量: {len(results.get('phase2_losses', []))}")
        print(f"模型保存路径: {results.get('model_path', 'N/A')}")
        
        # 可视化结果
        print("\n生成可视化结果...")
        visualize_training_results(results)
        
    else:
        # 运行psychometric训练（默认）
        print("开始psychometric训练...")
        results = train_with_psychometric_data(
            data_dir='data',
            subject='DD',  # 可以改为 'Evender' 或 None
            save_dir='checkpoints',
            mode='psychometric'
        )
        
        print("训练完成！")
        print(f"第一阶段损失数量: {len(results.get('phase1_losses', []))}")
        print(f"第二阶段损失数量: {len(results.get('phase2_losses', []))}")
        print(f"模型保存路径: {results.get('model_path', 'N/A')}")
        
        # 可视化结果
        print("\n生成可视化结果...")
        visualize_training_results(results)
