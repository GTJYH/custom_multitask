"""基于PyTorch的LSTM网络模型，用于处理自定义眼跳任务"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json
import pickle
from typing import Dict, Optional, Tuple, List, Union


def popvec(y):
    """群体向量读取 - 从环状网络的激活模式中解码出角度位置
    
    原理：通过计算所有单元激活的加权平均方向来得到最终的角度位置
    每个单元都有一个首选方向，通过三角函数计算加权平均
    
    Args:
        y: 环状网络的群体输出. Numpy数组 (Batch, Units)
    
    Returns:
        解码出的角度位置: Numpy数组 (Batch,)
    """
    # 计算每个单元的首选方向（0到2π均匀分布）
    pref = np.arange(0, 2*np.pi, 2*np.pi/y.shape[-1])
    
    # 计算总激活强度
    temp_sum = y.sum(axis=-1)
    
    # 计算加权平均的cos和sin分量
    temp_cos = np.sum(y*np.cos(pref), axis=-1)/temp_sum
    temp_sin = np.sum(y*np.sin(pref), axis=-1)/temp_sum
    
    # 转换为角度
    loc = np.arctan2(temp_sin, temp_cos)
    return np.mod(loc, 2*np.pi)








def analyze_error_types(y_hat, y_loc):
    """详细分析错误类型
    
    错误类型分类：
    - fixation_error: 注视阶段没有注视
    - direction_error: 眼跳方向错误
    - both_errors: 眼跳阶段仍注视
    - correct: 完全正确
    
    Args:
        y_hat: 网络实际输出. Numpy数组 (Time, Batch, Unit)
        y_loc: 目标输出位置（-1表示注视）. Numpy数组 (Time, Batch)
    
    Returns:
        error_analysis: 包含详细错误分析的字典
    """
    if len(y_hat.shape) != 3:
        raise ValueError('y_hat must have shape (Time, Batch, Unit)')
    
    time_steps, batch_size, n_units = y_hat.shape
    
    # 初始化错误统计
    error_stats = {
        'total_trials': time_steps * batch_size,
        'fixation_errors': 0,      # 注视阶段没有注视
        'direction_errors': 0,     # 眼跳方向错误
        'both_errors': 0,          # 眼跳阶段仍注视
        'correct_trials': 0,       # 完全正确
        'error_details': []        # 详细错误记录
    }
    
    # 对每个时间点和batch进行分析
    for t in range(time_steps):
        for b in range(batch_size):
            y_loc_tb = y_loc[t, b]
            y_hat_tb = y_hat[t, b]
            
            # 解析网络输出
            y_hat_fix = y_hat_tb[0]        # 注视单元输出
            y_hat_loc = popvec(y_hat_tb[1:])  # 环单元输出解码为角度位置
            
            # 判断当前状态
            fixating = y_hat_fix > 0.5       # 是否在注视状态
            
            # 计算位置误差
            original_dist = y_loc_tb - y_hat_loc
            dist = np.minimum(np.abs(original_dist), 2*np.pi-np.abs(original_dist))
            corr_loc = dist < 0.2*np.pi      # 位置是否正确
            
            # 判断应该做什么
            should_fix = y_loc_tb < 0         # 目标位置<0表示应该注视
            
            # 分析错误类型
            # fixation_correct: 注视状态是否正确（应该注视时在注视，应该眼跳时不在注视）
            fixation_correct = (should_fix and fixating) or (not should_fix and not fixating)
            # direction_correct: 方向是否正确（眼跳阶段方向正确，或注视阶段不检查方向）
            direction_correct = (not should_fix and corr_loc) or should_fix
            
            # 记录错误详情
            error_detail = {
                'time_step': t,
                'batch_idx': b,
                'should_fix': should_fix,
                'is_fixating': fixating,
                'target_location': y_loc_tb,
                'predicted_location': y_hat_loc,
                'location_error': dist,
                'fixation_correct': fixation_correct,
                'direction_correct': direction_correct,
                'error_type': None
            }
            
            # 分类错误类型
            if fixation_correct and direction_correct:
                error_stats['correct_trials'] += 1
                error_detail['error_type'] = 'correct'
            elif not fixation_correct and not direction_correct:
                error_stats['both_errors'] += 1
                error_detail['error_type'] = 'both_errors'  # 眼跳阶段仍注视
            elif not fixation_correct:
                error_stats['fixation_errors'] += 1
                error_detail['error_type'] = 'fixation_error'  # 注视阶段没有注视
            else:  # not direction_correct
                error_stats['direction_errors'] += 1
                error_detail['error_type'] = 'direction_error'  # 眼跳方向错误
            
            error_stats['error_details'].append(error_detail)
    
    # 计算错误率
    total_errors = error_stats['fixation_errors'] + error_stats['direction_errors'] + error_stats['both_errors']
    error_stats['total_error_rate'] = total_errors / error_stats['total_trials']
    error_stats['fixation_error_rate'] = error_stats['fixation_errors'] / error_stats['total_trials']  # 注视阶段没有注视的错误率
    error_stats['direction_error_rate'] = error_stats['direction_errors'] / error_stats['total_trials']  # 眼跳方向错误率
    error_stats['both_error_rate'] = error_stats['both_errors'] / error_stats['total_trials']  # 眼跳阶段仍注视的错误率
    error_stats['correct_rate'] = error_stats['correct_trials'] / error_stats['total_trials']
    
    return error_stats


def get_perf(y_hat, y_loc):
    """计算眼跳任务的性能（全时间评估的汇总指标）
    
    评估逻辑：
    - 网络输出：1个注视单元 + 32个环单元
    - 注视单元：控制是否注视（>0.5表示注视）
    - 环单元：通过popvec解码出眼跳方向
    - 性能判断：应该注视时在注视，应该眼跳时方向正确
    
    Args:
        y_hat: 网络实际输出. Numpy数组 (Time, Batch, Unit)
        y_loc: 目标输出位置（-1表示注视）. Numpy数组 (Time, Batch)
    
    Returns:
        metrics: 包含多个性能指标的字典
    """
    if len(y_hat.shape) != 3:
        raise ValueError('y_hat must have shape (Time, Batch, Unit)')
    
    time_steps, batch_size, n_units = y_hat.shape
    perf_series = np.zeros((time_steps, batch_size))
    
    # 对每个时间点计算性能
    for t in range(time_steps):
        y_loc_t = y_loc[t]
        y_hat_t = y_hat[t]
        
        # 解析网络输出
        y_hat_fix = y_hat_t[..., 0]        # 注视单元输出
        y_hat_loc = popvec(y_hat_t[..., 1:])  # 环单元输出解码为角度位置
        
        # 判断当前状态
        fixating = y_hat_fix > 0.5       # 是否在注视状态
        
        # 计算位置误差（考虑角度周期性）
        original_dist = y_loc_t - y_hat_loc
        dist = np.minimum(np.abs(original_dist), 2*np.pi-np.abs(original_dist))
        corr_loc = dist < 0.2*np.pi      # 位置是否正确（容差约36度）
        
        # 判断应该做什么
        should_fix = y_loc_t < 0         # 目标位置<0表示应该注视
        
        # 计算性能
        # 应该注视时：注视=1分，不注视=0分
        # 应该眼跳时：不注视且方向正确=1分，其他=0分
        perf_t = should_fix * fixating + (1-should_fix) * corr_loc * (1-fixating)
        perf_series[t] = perf_t
    
    # 计算各种性能指标
    final_perf = np.mean(perf_series[-1])      # 最终成功率
    avg_perf = np.mean(perf_series)            # 平均表现
    stability = 1 - np.std(perf_series)        # 稳定性（1-标准差）
    max_perf = np.max(perf_series)             # 最佳表现
    
    # 计算学习曲线（性能随时间的变化趋势）
    time_trend = np.polyfit(range(len(perf_series)), np.mean(perf_series, axis=1), 1)[0]
    
    return {
        'final_performance': final_perf,       # 最终成功率
        'average_performance': avg_perf,       # 平均表现
        'stability': max(0, stability),        # 稳定性（0-1）
        'max_performance': max_perf,           # 最佳表现
        'learning_trend': time_trend,          # 学习趋势（斜率）
        'performance_series': perf_series      # 完整性能时间序列
    }


class LSTMNetwork(nn.Module):
    """LSTM网络模型"""
    
    def __init__(self, n_input: int, n_rnn: int, n_output: int, 
                 activation: str = 'relu', dropout: float = 0.0):
        """
        Args:
            n_input: 输入维度 (37: 1注视 + 32环 + 4任务信号)
            n_rnn: LSTM隐藏层维度
            n_output: 输出维度 (33: 1注视 + 32环)
            activation: 激活函数 ('relu', 'tanh', 'softplus')
            dropout: dropout率
        """
        super(LSTMNetwork, self).__init__()
        
        self.n_input = n_input
        self.n_rnn = n_rnn
        self.n_output = n_output
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=n_input,
            hidden_size=n_rnn,
            num_layers=1,
            batch_first=False,  # (seq_len, batch, input_size)
            dropout=dropout
        )
        
        # 输出层
        self.output_layer = nn.Linear(n_rnn, n_output)
        
        # 激活函数
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'softplus':
            self.activation = nn.Softplus()
        else:
            raise ValueError(f'Unsupported activation: {activation}')
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    # LSTM权重初始化
                    nn.init.xavier_uniform_(param)
                else:
                    # 输出层权重初始化
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x):
        """
        Args:
            x: 输入张量 (seq_len, batch_size, n_input)
        
        Returns:
            output: 输出张量 (seq_len, batch_size, n_output)
        """
        # LSTM前向传播
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # 输出层
        output = self.output_layer(lstm_out)
        
        # 应用激活函数
        output = self.activation(output)
        
        return output


class CustomSaccadeModel(nn.Module):
    """自定义眼跳任务模型"""
    
    def __init__(self, model_dir: str, hp: Optional[Dict] = None, 
                 device: str = 'cpu'):
        super(CustomSaccadeModel, self).__init__()
        """
        Args:
            model_dir: 模型保存目录
            hp: 超参数字典
            device: 计算设备 ('cpu', 'cuda')
        """
        self.model_dir = model_dir
        self.device = torch.device(device)
        
        # 加载或设置超参数
        if hp is None:
            hp = self._load_hp()
        
        self.hp = hp
        self.rng = np.random.RandomState(hp.get('seed', 0))
        
        # 构建网络
        self.network = LSTMNetwork(
            n_input=hp['n_input'],
            n_rnn=hp['n_rnn'],
            n_output=hp['n_output'],
            activation=hp.get('activation', 'relu'),
            dropout=hp.get('dropout', 0.0)
        ).to(self.device)
        
        # 设置优化器
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=hp.get('learning_rate', 0.001)
        )
        
        # 损失函数
        self.loss_type = hp.get('loss_type', 'lsq')
        if self.loss_type == 'lsq':
            self.criterion = nn.MSELoss(reduction='none')
        else:
            raise ValueError(f'Unsupported loss type: {self.loss_type}')
        
        # 持续学习相关
        self.continual_learning_config = hp.get('continual_learning', {
            'enabled': False,
            'c_intsyn': 1.0,
            'ksi_intsyn': 0.01,
            'ewc_lambda': 100.0
        })
        
        # 智能突触相关变量
        self.v_anc0 = None  # 锚点权重
        self.Omega0 = None  # 重要性矩阵
        self.omega0 = None  # 累积梯度
        self.v_delta = None  # 权重变化
        
        # EWC相关变量
        self.ewc_data = {}  # 存储Fisher信息矩阵和锚点权重
        
        # 创建模型目录
        os.makedirs(model_dir, exist_ok=True)
        
        # 保存超参数
        self._save_hp()
    
    def _load_hp(self) -> Dict:
        """加载超参数"""
        hp_path = os.path.join(self.model_dir, 'hp.json')
        if os.path.exists(hp_path):
            with open(hp_path, 'r') as f:
                return json.load(f)
        else:
            # 返回默认超参数
            return {
                'n_input': 37,
                'n_rnn': 256,
                'n_output': 33,
                'learning_rate': 0.001,
                'loss_type': 'lsq',
                'activation': 'relu',
                'dropout': 0.0,
                'seed': 42
            }
    
    def _save_hp(self):
        """保存超参数"""
        hp_path = os.path.join(self.model_dir, 'hp.json')
        with open(hp_path, 'w') as f:
            json.dump(self.hp, f, indent=2)
    
    def _get_current_weights(self) -> List[torch.Tensor]:
        """获取当前权重"""
        return [param.clone().detach() for param in self.network.parameters()]
    
    def _compute_importance_weights(self, v_current: List[torch.Tensor], 
                                  v_prev: List[torch.Tensor], 
                                  gradients: List[torch.Tensor]) -> List[torch.Tensor]:
        """计算智能突触的重要性权重
        
        Args:
            v_current: 当前权重
            v_prev: 前一步权重
            gradients: 梯度
            
        Returns:
            omega0: 重要性权重
        """
        c = self.continual_learning_config['c_intsyn']
        ksi = self.continual_learning_config['ksi_intsyn']
        
        omega0 = []
        for v_c, v_p, v_g in zip(v_current, v_prev, gradients):
            # 计算权重变化
            v_d = v_c - v_p
            # 更新重要性权重
            o = -(v_d * v_g)
            omega0.append(o)
        
        return omega0
    
    def _update_importance_matrix(self, v_current: List[torch.Tensor]):
        """更新重要性矩阵（智能突触）
        
        Args:
            v_current: 当前权重
        """
        if self.v_anc0 is None:
            # 第一次训练，初始化
            self.v_anc0 = v_current
            self.Omega0 = [torch.zeros_like(v) for v in v_current]
            self.omega0 = [torch.zeros_like(v) for v in v_current]
            self.v_delta = [torch.zeros_like(v) for v in v_current]
        else:
            # 计算权重变化
            v_delta = [v - v_prev for v, v_prev in zip(v_current, self.v_anc0)]
            self.v_delta = v_delta
            
            # 更新重要性矩阵
            c = self.continual_learning_config['c_intsyn']
            ksi = self.continual_learning_config['ksi_intsyn']
            
            new_Omega0 = []
            for O, o, v_d in zip(self.Omega0, self.omega0, v_delta):
                # 使用ReLU确保非负性
                O_new = torch.relu(O + o / (v_d ** 2 + ksi))
                new_Omega0.append(O_new)
            
            self.Omega0 = new_Omega0
            self.v_anc0 = v_current
            self.omega0 = [torch.zeros_like(v) for v in v_current]
    
    def _compute_intelligent_synapses_loss(self, v_current: List[torch.Tensor]) -> torch.Tensor:
        """计算智能突触正则化损失
        
        Args:
            v_current: 当前权重
            
        Returns:
            reg_loss: 正则化损失
        """
        if self.Omega0 is None:
            return torch.tensor(0.0, device=self.device)
        
        c = self.continual_learning_config['c_intsyn']
        reg_loss = torch.tensor(0.0, device=self.device)
        
        for param, omega in zip(self.network.parameters(), self.Omega0):
            if param.requires_grad:
                # 计算与锚点权重的差异
                v_anchor = self.v_anc0[len(reg_loss.shape):] if len(reg_loss.shape) > 0 else self.v_anc0[0]
                diff = param - v_anchor
                # 加权平方差
                reg_loss += c * torch.sum(omega * diff ** 2)
        
        return reg_loss
    
    def _compute_ewc_loss(self) -> torch.Tensor:
        """计算EWC正则化损失
        
        Returns:
            reg_loss: EWC正则化损失
        """
        if not self.ewc_data:
            return torch.tensor(0.0, device=self.device)
        
        lambda_ewc = self.continual_learning_config['ewc_lambda']
        reg_loss = torch.tensor(0.0, device=self.device)
        
        for name, param in self.network.named_parameters():
            if name in self.ewc_data and param.requires_grad:
                fisher_info = self.ewc_data[name]['fisher']
                anchor_weight = self.ewc_data[name]['anchor']
                diff = param - anchor_weight
                reg_loss += lambda_ewc * torch.sum(fisher_info * diff ** 2)
        
        return reg_loss
    
    def update_ewc_data(self, task_name: str):
        """更新EWC数据（计算Fisher信息矩阵）
        
        Args:
            task_name: 任务名称
        """
        # 这里简化实现，实际应该基于验证数据计算Fisher信息
        # 暂时使用单位矩阵作为Fisher信息的近似
        for name, param in self.network.named_parameters():
            if param.requires_grad:
                self.ewc_data[name] = {
                    'fisher': torch.ones_like(param),
                    'anchor': param.clone().detach()
                }
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.network(x)
    
    def compute_loss(self, y_hat: torch.Tensor, y: torch.Tensor, 
                    c_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """计算损失（包括正则化项）
        
        Args:
            y_hat: 预测输出 (seq_len, batch_size, n_output)
            y: 目标输出 (seq_len, batch_size, n_output)
            c_mask: 成本掩码 (seq_len, batch_size, n_output)
        
        Returns:
            total_loss: 总损失
        """
        # 基础损失
        if c_mask is not None:
            # 使用成本掩码
            loss = self.criterion(y_hat, y) * c_mask
            loss = loss.sum() / c_mask.sum()
        else:
            # 不使用成本掩码
            loss = self.criterion(y_hat, y).mean()
        
        # 持续学习正则化
        if self.continual_learning_config['enabled']:
            # 智能突触正则化
            v_current = self._get_current_weights()
            intsyn_loss = self._compute_intelligent_synapses_loss(v_current)
            
            # EWC正则化
            ewc_loss = self._compute_ewc_loss()
            
            # 总损失
            total_loss = loss + intsyn_loss + ewc_loss
        else:
            total_loss = loss
        
        return total_loss
    
    def train_step(self, x: torch.Tensor, y: torch.Tensor, 
                  c_mask: Optional[torch.Tensor] = None) -> float:
        """训练一步（支持持续学习）
        
        Args:
            x: 输入 (seq_len, batch_size, n_input)
            y: 目标 (seq_len, batch_size, n_output)
            c_mask: 成本掩码 (seq_len, batch_size, n_output)
        
        Returns:
            loss: 损失值
        """
        self.train()
        self.optimizer.zero_grad()
        
        # 获取训练前的权重（用于智能突触）
        if self.continual_learning_config['enabled']:
            v_prev = self._get_current_weights()
        
        # 前向传播
        y_hat = self.forward(x)
        
        # 计算损失
        loss = self.compute_loss(y_hat, y, c_mask)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        
        # 更新参数
        self.optimizer.step()
        
        # 持续学习：更新智能突触
        if self.continual_learning_config['enabled']:
            v_current = self._get_current_weights()
            
            # 获取梯度（在参数更新后）
            gradients = []
            for param in self.network.parameters():
                if param.grad is not None:
                    gradients.append(param.grad.clone().detach())
                else:
                    gradients.append(torch.zeros_like(param))
            
            # 更新重要性权重
            if self.omega0 is not None:
                self.omega0 = self._compute_importance_weights(v_current, v_prev, gradients)
        
        return loss.item()
    
    def evaluate(self, y_hat: torch.Tensor, y_loc: torch.Tensor, detailed_analysis: bool = False) -> Union[float, Dict]:
        """评估模型性能
        
        Args:
            y_hat: 网络输出 (seq_len, batch_size, n_output)
            y_loc: 目标位置 (seq_len, batch_size)
            detailed_analysis: 是否进行详细错误分析
        
        Returns:
            如果detailed_analysis=False: 返回平均性能值 (float)
            如果detailed_analysis=True: 返回包含性能指标和错误分析的字典
        """
        with torch.no_grad():
            # 转换为numpy进行性能计算
            y_hat_np = y_hat.cpu().numpy()
            y_loc_np = y_loc.cpu().numpy()
            
            # 计算综合性能指标
            metrics = get_perf(y_hat_np, y_loc_np)
            
            if not detailed_analysis:
                # 返回平均性能值
                return metrics['average_performance']
            else:
                # 计算详细错误分析
                error_analysis = analyze_error_types(y_hat_np, y_loc_np)
                
                # 合并结果
                analysis = {
                    'performance_metrics': metrics,
                    'error_analysis': error_analysis
                }
                
                return analysis
    

    

    
    def save(self, filename: str = 'model.pth'):
        """保存模型"""
        model_path = os.path.join(self.model_dir, filename)
        torch.save({
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'hp': self.hp
        }, model_path)
        # 移除print语句，只使用logger
        # print(f'Model saved to {model_path}')
    
    def load(self, filename: str = 'model.pth'):
        """加载模型"""
        model_path = os.path.join(self.model_dir, filename)
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.network.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f'Model loaded from {model_path}')
        else:
            print(f'No model found at {model_path}')
    
    def get_trainable_parameters(self) -> int:
        """获取可训练参数数量"""
        return sum(p.numel() for p in self.network.parameters() if p.requires_grad)

    def start_new_task(self, task_name: str):
        """开始新任务（用于持续学习）
        
        Args:
            task_name: 任务名称
        """
        if self.continual_learning_config['enabled']:
            # 更新重要性矩阵
            v_current = self._get_current_weights()
            self._update_importance_matrix(v_current)
            
            # 更新EWC数据
            self.update_ewc_data(task_name)
            
            print(f"持续学习：开始新任务 {task_name}")
            print(f"  - 智能突触参数: c={self.continual_learning_config['c_intsyn']}, ksi={self.continual_learning_config['ksi_intsyn']}")
            print(f"  - EWC参数: lambda={self.continual_learning_config['ewc_lambda']}")


def create_model_from_trial(trial, hp: Optional[Dict] = None, 
                          model_dir: str = 'models/custom_saccade',
                          device: str = 'cpu') -> CustomSaccadeModel:
    """从trial创建模型
    
    Args:
        trial: Trial对象
        hp: 超参数字典
        model_dir: 模型保存目录
        device: 计算设备
    
    Returns:
        model: 自定义眼跳模型
    """
    if hp is None:
        hp = {}
    
    # 从trial获取维度信息
    hp['n_input'] = trial.n_input
    hp['n_output'] = trial.n_output
    
    # 设置默认值
    hp.setdefault('n_rnn', 256)
    hp.setdefault('activation', 'relu')
    hp.setdefault('learning_rate', 0.001)
    hp.setdefault('loss_type', 'lsq')
    hp.setdefault('dropout', 0.0)
    hp.setdefault('seed', 0)
    
    return CustomSaccadeModel(model_dir, hp, device)


def prepare_trial_data(trial, device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """准备trial数据用于训练
    
    Args:
        trial: Trial对象
        device: 计算设备
    
    Returns:
        x: 输入张量 (seq_len, batch_size, n_input)
        y: 目标张量 (seq_len, batch_size, n_output)
        y_loc: 位置张量 (seq_len, batch_size)
    """
    # 转换为PyTorch张量
    x = torch.FloatTensor(trial.x).to(device)
    y = torch.FloatTensor(trial.y).to(device)
    y_loc = torch.FloatTensor(trial.y_loc).to(device)
    
    return x, y, y_loc


def prepare_cost_mask(trial, device: str = 'cpu') -> torch.Tensor:
    """准备成本掩码
    
    Args:
        trial: Trial对象
        device: 计算设备
    
    Returns:
        c_mask: 成本掩码张量 (seq_len, batch_size, n_output)
    """
    if hasattr(trial, 'c_mask'):
        # 重塑掩码以匹配输出维度
        c_mask = trial.c_mask.reshape(trial.tdim, trial.batch_size, trial.n_output)
        return torch.FloatTensor(c_mask).to(device)
    else:
        # 如果没有掩码，返回全1
        return torch.ones(trial.tdim, trial.batch_size, trial.n_output).to(device)
