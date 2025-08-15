"""
眼跳任务定点分析模块
分析LSTM网络中的定点（attractor）和稳态行为，用于理解网络的动态特性
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
import json
from pathlib import Path
import sys

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from task import generate_trials
from model import CustomSaccadeModel
from utils.logger import AnalysisLogger
from utils.config import HyperParameters


class FixedPointAnalysis:
    """LSTM网络定点分析类，用于分析网络的动态特性和定点行为"""

    def __init__(self, model_path, hp_path=None):
        """初始化定点分析
        
        Args:
            model_path: str, 模型文件路径
            hp_path: str, 超参数文件路径，如果为None则从model_path同目录查找
        """
        # 设置日志
        self.logger = AnalysisLogger("fixed_point_analysis")
        
        self.logger.log_info("开始初始化定点分析")
        
        # 加载模型和超参数
        self.model_path = model_path
        self.model_dir = str(Path(model_path).parent)
        
        if hp_path is None:
            hp_path = os.path.join(self.model_dir, 'hp.json')
        
        self.logger.log_info(f"加载超参数文件: {hp_path}")
        with open(hp_path, 'r') as f:
            model_hp = json.load(f)
        
        # 从config.py加载默认的任务参数
        default_hp = HyperParameters()
        default_hp_dict = default_hp.to_dict()
        
        # 合并参数：模型参数优先，缺失的用默认值补充
        self.hp = default_hp_dict.copy()
        self.hp.update(model_hp)  # 模型参数覆盖默认值
        
        self.logger.log_info(f"模型参数: {model_hp}")
        self.logger.log_info(f"完整超参数: {self.hp}")
        
        # 加载模型
        self.logger.log_info("加载PyTorch模型")
        
        # 添加随机数生成器到超参数（用于trial生成）
        self.hp['rng'] = np.random.RandomState(self.hp.get('random_seed', 42))
        
        # 为模型初始化创建可序列化的hp副本（移除rng）
        model_hp_serializable = self.hp.copy()
        del model_hp_serializable['rng']
        
        # 初始化模型
        self.logger.log_info("开始创建CustomSaccadeModel实例...")
        try:
            self.model = CustomSaccadeModel(
                model_dir=self.model_dir,
                hp=model_hp_serializable
            )
            self.logger.log_info("CustomSaccadeModel实例创建完成")
        except Exception as e:
            self.logger.log_error(e, "创建CustomSaccadeModel实例")
            raise e
        
        # 加载训练好的权重
        self.logger.log_info("开始加载模型权重...")
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
            self.logger.log_info("模型权重加载完成")
        except Exception as e:
            self.logger.log_error(e, "加载模型权重")
            raise e
        
        # 检查是否是完整的检查点还是单独的state_dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # 完整检查点格式
            state_dict = checkpoint['model_state_dict']
            self.logger.log_info("检测到完整检查点格式，提取model_state_dict")
        else:
            # 单独的state_dict格式
            state_dict = checkpoint
            self.logger.log_info("检测到单独的state_dict格式")
        
        # 处理键名映射问题
        self.logger.log_info("开始处理状态字典键名映射...")
        new_state_dict = {}
        for key, value in state_dict.items():
            # 如果键名不包含'network.'前缀，添加它
            if not key.startswith('network.'):
                new_key = f'network.{key}'
                new_state_dict[new_key] = value
                self.logger.log_info(f"映射键名: {key} -> {new_key}")
            else:
                new_state_dict[key] = value
        
        self.logger.log_info("开始加载状态字典到模型...")
        self.model.load_state_dict(new_state_dict)
        self.logger.log_info("状态字典加载完成")
        self.model.eval()
        self.logger.log_info("模型加载完成")
        
        # 定义四个眼跳任务
        self.tasks = ['pro_saccade', 'anti_saccade', 'delay_pro', 'delay_anti']
        
        # 任务颜色映射
        self.task_colors = {
            'pro_saccade': '#1f77b4',      # 蓝色
            'anti_saccade': '#ff7f0e',     # 橙色
            'delay_pro': '#2ca02c',        # 绿色
            'delay_anti': '#d62728'        # 红色
        }
        
        # 任务名称映射
        self.task_names = {
            'pro_saccade': 'Pro-saccade',
            'anti_saccade': 'Anti-saccade', 
            'delay_pro': 'Delay Pro',
            'delay_anti': 'Delay Anti'
        }
        
        # 存储分析结果
        self.fixed_points = OrderedDict()
        self.attractor_basins = OrderedDict()
        self.stability_analysis = OrderedDict()
        self.trajectory_analysis = OrderedDict()
        
        # 网络参数
        self.n_hidden = self.hp.get('n_rnn', 256)
        self.n_input = self.hp.get('n_input', 37)
        
        self.logger.log_info(f"网络参数: 隐藏层维度={self.n_hidden}, 输入维度={self.n_input}")
    
    def find_fixed_points(self, task_name, num_trials=100, tolerance=1e-6, max_iter=1000):
        """寻找特定任务的定点
        
        Args:
            task_name: str, 任务名称
            num_trials: int, 寻找定点的试验次数
            tolerance: float, 收敛容差
            max_iter: int, 最大迭代次数
            
        Returns:
            fixed_points: list, 找到的定点列表
        """
        self.logger.log_info(f"开始寻找任务 {task_name} 的定点")
        
        fixed_points = []
        
        with torch.no_grad():
            for trial_idx in range(num_trials):
                if trial_idx % 5 == 0:  # 每5次试验输出一次进度
                    self.logger.log_info(f"试验进度: {trial_idx}/{num_trials}")
                
                # 生成随机输入
                trial = generate_trials(task_name, self.hp, 'random', batch_size=1)
                x = torch.tensor(trial.x, dtype=torch.float32)
                
                # 使用不同的初始隐藏状态
                h0 = torch.randn(1, 1, self.n_hidden) * 0.1  # 小随机初始值
                
                # 寻找定点
                fixed_point = self._find_single_fixed_point(x, h0, tolerance, max_iter)
                
                if fixed_point is not None:
                    fixed_points.append(fixed_point)
                    if trial_idx % 5 == 0:  # 减少日志输出频率
                        self.logger.log_info(f"试验 {trial_idx}: 找到定点，收敛误差: {fixed_point['error']:.2e}")
                else:
                    if trial_idx % 20 == 0:  # 减少日志输出频率
                        self.logger.log_info(f"试验 {trial_idx}: 未找到定点")
        
        self.fixed_points[task_name] = fixed_points
        self.logger.log_info(f"任务 {task_name} 找到 {len(fixed_points)} 个定点")
        
        return fixed_points
    
    def _find_single_fixed_point(self, x, h0, tolerance, max_iter):
        """寻找单个定点
        
        Args:
            x: torch.Tensor, 输入序列
            h0: torch.Tensor, 初始隐藏状态
            tolerance: float, 收敛容差
            max_iter: int, 最大迭代次数
            
        Returns:
            fixed_point: dict, 定点信息
        """
        # 获取LSTM参数
        lstm = self.model.network.lstm
        
        # 提取权重
        W_ih = lstm.weight_ih_l0.data.clone()
        W_hh = lstm.weight_hh_l0.data.clone()
        b_ih = lstm.bias_ih_l0.data.clone()
        b_hh = lstm.bias_hh_l0.data.clone()
        
        # 分离输入门、遗忘门、候选门、输出门
        n_hidden = self.n_hidden
        W_ii, W_if, W_ig, W_io = W_ih.chunk(4, 0)
        W_hi, W_hf, W_hg, W_ho = W_hh.chunk(4, 0)
        b_ii, b_if, b_ig, b_io = b_ih.chunk(4, 0)
        b_hi, b_hf, b_hg, b_ho = b_hh.chunk(4, 0)
        
        # 使用平均输入作为固定输入
        x_avg = x.mean(dim=0, keepdim=True)  # (1, 1, n_input)
        x_avg_2d = x_avg.squeeze(0)  # (1, n_input) - 保持2D形状用于矩阵乘法
        
        # 迭代寻找定点
        h = h0.clone()
        c = torch.zeros_like(h)
        
        for iter_idx in range(max_iter):
            h_prev = h.clone()
            c_prev = c.clone()
            
            # LSTM前向传播（使用分离的门控权重）
            # 输入门
            i_input = torch.mm(x_avg_2d, W_ii.t()) + b_ii
            i_hidden = torch.mm(h.squeeze(0), W_hi.t()) + b_hi
            i = torch.sigmoid(i_input + i_hidden)
            
            # 遗忘门
            f_input = torch.mm(x_avg_2d, W_if.t()) + b_if
            f_hidden = torch.mm(h.squeeze(0), W_hf.t()) + b_hf
            f = torch.sigmoid(f_input + f_hidden)
            
            # 候选门
            g_input = torch.mm(x_avg_2d, W_ig.t()) + b_ig
            g_hidden = torch.mm(h.squeeze(0), W_hg.t()) + b_hg
            g = torch.tanh(g_input + g_hidden)
            
            # 输出门
            o_input = torch.mm(x_avg_2d, W_io.t()) + b_io
            o_hidden = torch.mm(h.squeeze(0), W_ho.t()) + b_ho
            o = torch.sigmoid(o_input + o_hidden)
            
            c = f * c + i * g
            h = o * torch.tanh(c)
            
            # 检查收敛
            h_error = torch.norm(h - h_prev)
            c_error = torch.norm(c - c_prev)
            
            if h_error < tolerance and c_error < tolerance:
                return {
                    'h': h.detach().numpy(),
                    'c': c.detach().numpy(),
                    'x': x_avg.detach().numpy(),
                    'error': float(h_error),
                    'iterations': iter_idx + 1
                }
        
        return None
    
    def analyze_stability(self, task_name, fixed_points, num_perturbations=100):
        """分析定点的稳定性
        
        Args:
            task_name: str, 任务名称
            fixed_points: list, 定点列表
            num_perturbations: int, 扰动试验次数
            
        Returns:
            stability_results: dict, 稳定性分析结果
        """
        self.logger.log_info(f"开始分析任务 {task_name} 的定点稳定性")
        
        stability_results = {
            'stable_points': [],
            'unstable_points': [],
            'stability_scores': [],
            'perturbation_analysis': []
        }
        
        for i, fp in enumerate(fixed_points):
            self.logger.log_info(f"分析定点 {i+1}/{len(fixed_points)} 的稳定性...")
            h_fp = torch.tensor(fp['h'], dtype=torch.float32)
            c_fp = torch.tensor(fp['c'], dtype=torch.float32)
            
            # 计算雅可比矩阵
            self.logger.log_info(f"  计算雅可比矩阵...")
            J = self._compute_jacobian(h_fp, c_fp, fp['x'])
            self.logger.log_info(f"  雅可比矩阵计算完成")
            
            # 计算特征值
            eigenvals = np.linalg.eigvals(J)
            
            # 判断稳定性（最大特征值的实部小于0）
            max_real_part = np.max(np.real(eigenvals))
            # 使用更宽松的稳定性判断标准
            # 考虑到数值误差和LSTM网络的特性
            is_stable = max_real_part < 0.01  # 允许小的正特征值
            
            # 输出调试信息
            self.logger.log_info(f"    定点 {i+1} 最大特征值实部: {max_real_part:.6f}, 稳定: {is_stable}")
            self.logger.log_info(f"    特征值范围: [{np.min(np.real(eigenvals)):.6f}, {np.max(np.real(eigenvals)):.6f}]")
            
            # 扰动试验
            perturbation_results = self._perturbation_test(h_fp, c_fp, fp['x'], num_perturbations)
            
            stability_score = {
                'point_index': i,
                'is_stable': is_stable,
                'max_eigenvalue_real': max_real_part,
                'eigenvalues': eigenvals,
                'perturbation_results': perturbation_results
            }
            
            stability_results['stability_scores'].append(stability_score)
            
            if is_stable:
                stability_results['stable_points'].append(fp)
            else:
                stability_results['unstable_points'].append(fp)
        
        self.stability_analysis[task_name] = stability_results
        self.logger.log_info(f"任务 {task_name}: {len(stability_results['stable_points'])} 个稳定定点, "
                        f"{len(stability_results['unstable_points'])} 个不稳定定点")
        
        return stability_results
    
    def _compute_jacobian(self, h, c, x):
        """计算雅可比矩阵
        
        Args:
            h: torch.Tensor, 隐藏状态
            c: torch.Tensor, 细胞状态
            x: torch.Tensor, 输入
            
        Returns:
            J: numpy.ndarray, 雅可比矩阵
        """
        # 获取LSTM参数
        lstm = self.model.network.lstm
        W_ih = lstm.weight_ih_l0.data
        W_hh = lstm.weight_hh_l0.data
        b_ih = lstm.bias_ih_l0.data
        b_hh = lstm.bias_hh_l0.data
        
        # 分离门控权重
        n_hidden = self.n_hidden
        W_ii, W_if, W_ig, W_io = W_ih.chunk(4, 0)
        W_hi, W_hf, W_hg, W_ho = W_hh.chunk(4, 0)
        b_ii, b_if, b_ig, b_io = b_ih.chunk(4, 0)
        b_hi, b_hf, b_hg, b_ho = b_hh.chunk(4, 0)
        
        # 确保输入和隐藏状态的维度正确
        # 将numpy数组转换为torch张量
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        if isinstance(h, np.ndarray):
            h = torch.tensor(h, dtype=torch.float32)
        
        x_2d = x.squeeze(0) if x.dim() == 3 else x  # (1, n_input)
        h_2d = h.squeeze(0) if h.dim() == 3 else h  # (1, n_hidden)
        
        # 使用分离的门控权重计算门控值
        # 输入门
        i_input = torch.mm(x_2d, W_ii.t()) + b_ii
        i_hidden = torch.mm(h_2d, W_hi.t()) + b_hi
        i = torch.sigmoid(i_input + i_hidden)
        
        # 遗忘门
        f_input = torch.mm(x_2d, W_if.t()) + b_if
        f_hidden = torch.mm(h_2d, W_hf.t()) + b_hf
        f = torch.sigmoid(f_input + f_hidden)
        
        # 候选门
        g_input = torch.mm(x_2d, W_ig.t()) + b_ig
        g_hidden = torch.mm(h_2d, W_hg.t()) + b_hg
        g = torch.tanh(g_input + g_hidden)
        
        # 输出门
        o_input = torch.mm(x_2d, W_io.t()) + b_io
        o_hidden = torch.mm(h_2d, W_ho.t()) + b_ho
        o = torch.sigmoid(o_input + o_hidden)
        
        # 计算雅可比矩阵（更准确的版本）
        # 这里我们计算∂h/∂h的近似
        J = torch.zeros(n_hidden, n_hidden)
        
        # 使用更准确的数值微分近似
        eps = 1e-5  # 增大扰动大小
        self.logger.log_info(f"    开始数值微分计算，共 {n_hidden} 个维度...")
        
        # 计算未扰动时的状态
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        h_new_1d = h_new.squeeze()
        
        for j in range(n_hidden):
            if j % 100 == 0:  # 每50个维度输出一次进度
                self.logger.log_info(f"    数值微分进度: {j}/{n_hidden}")
            
            # 正向扰动
            h_plus = h.clone()
            h_plus[0, 0, j] += eps
            h_plus_2d = h_plus.squeeze(0) if h_plus.dim() == 3 else h_plus
            
            # 计算正向扰动后的门控值
            i_input_plus = torch.mm(x_2d, W_ii.t()) + b_ii
            i_hidden_plus = torch.mm(h_plus_2d, W_hi.t()) + b_hi
            i_plus = torch.sigmoid(i_input_plus + i_hidden_plus)
            
            f_input_plus = torch.mm(x_2d, W_if.t()) + b_if
            f_hidden_plus = torch.mm(h_plus_2d, W_hf.t()) + b_hf
            f_plus = torch.sigmoid(f_input_plus + f_hidden_plus)
            
            g_input_plus = torch.mm(x_2d, W_ig.t()) + b_ig
            g_hidden_plus = torch.mm(h_plus_2d, W_hg.t()) + b_hg
            g_plus = torch.tanh(g_input_plus + g_hidden_plus)
            
            o_input_plus = torch.mm(x_2d, W_io.t()) + b_io
            o_hidden_plus = torch.mm(h_plus_2d, W_ho.t()) + b_ho
            o_plus = torch.sigmoid(o_input_plus + o_hidden_plus)
            
            c_plus = f_plus * c + i_plus * g_plus
            h_new_plus = o_plus * torch.tanh(c_plus)
            h_new_plus_1d = h_new_plus.squeeze()
            
            # 负向扰动
            h_minus = h.clone()
            h_minus[0, 0, j] -= eps
            h_minus_2d = h_minus.squeeze(0) if h_minus.dim() == 3 else h_minus
            
            # 计算负向扰动后的门控值
            i_input_minus = torch.mm(x_2d, W_ii.t()) + b_ii
            i_hidden_minus = torch.mm(h_minus_2d, W_hi.t()) + b_hi
            i_minus = torch.sigmoid(i_input_minus + i_hidden_minus)
            
            f_input_minus = torch.mm(x_2d, W_if.t()) + b_if
            f_hidden_minus = torch.mm(h_minus_2d, W_hf.t()) + b_hf
            f_minus = torch.sigmoid(f_input_minus + f_hidden_minus)
            
            g_input_minus = torch.mm(x_2d, W_ig.t()) + b_ig
            g_hidden_minus = torch.mm(h_minus_2d, W_hg.t()) + b_hg
            g_minus = torch.tanh(g_input_minus + g_hidden_minus)
            
            o_input_minus = torch.mm(x_2d, W_io.t()) + b_io
            o_hidden_minus = torch.mm(h_minus_2d, W_ho.t()) + b_ho
            o_minus = torch.sigmoid(o_input_minus + o_hidden_minus)
            
            c_minus = f_minus * c + i_minus * g_minus
            h_new_minus = o_minus * torch.tanh(c_minus)
            h_new_minus_1d = h_new_minus.squeeze()
            
            # 使用中心差分计算偏导数
            J[:, j] = (h_new_plus_1d - h_new_minus_1d) / (2 * eps)
        
        return J.detach().numpy()
    
    def _perturbation_test(self, h_fp, c_fp, x, num_perturbations):
        """扰动试验
        
        Args:
            h_fp: torch.Tensor, 定点隐藏状态
            c_fp: torch.Tensor, 定点细胞状态
            x: torch.Tensor, 输入
            num_perturbations: int, 扰动次数
            
        Returns:
            results: dict, 扰动试验结果
        """
        results = {
            'returned_to_fixed_point': 0,
            'escaped': 0,
            'convergence_times': []
        }
        
        for _ in range(num_perturbations):
            # 添加随机扰动
            perturbation = torch.randn_like(h_fp) * 0.1
            h_perturbed = h_fp + perturbation
            
            # 模拟演化
            h_current = h_perturbed.clone()
            c_current = c_fp.clone()
            
            converged = False
            for step in range(100):  # 最多100步
                # LSTM前向传播（使用分离的门控权重）
                lstm = self.model.network.lstm
                W_ih = lstm.weight_ih_l0.data
                W_hh = lstm.weight_hh_l0.data
                b_ih = lstm.bias_ih_l0.data
                b_hh = lstm.bias_hh_l0.data
                
                # 分离门控权重
                W_ii, W_if, W_ig, W_io = W_ih.chunk(4, 0)
                W_hi, W_hf, W_hg, W_ho = W_hh.chunk(4, 0)
                b_ii, b_if, b_ig, b_io = b_ih.chunk(4, 0)
                b_hi, b_hf, b_hg, b_ho = b_hh.chunk(4, 0)
                
                # 确保输入和隐藏状态的维度正确
                # 将numpy数组转换为torch张量
                if isinstance(x, np.ndarray):
                    x = torch.tensor(x, dtype=torch.float32)
                
                x_2d = x.squeeze(0) if x.dim() == 3 else x
                h_current_2d = h_current.squeeze(0) if h_current.dim() == 3 else h_current
                
                # 计算门控值
                # 输入门
                i_input = torch.mm(x_2d, W_ii.t()) + b_ii
                i_hidden = torch.mm(h_current_2d, W_hi.t()) + b_hi
                i = torch.sigmoid(i_input + i_hidden)
                
                # 遗忘门
                f_input = torch.mm(x_2d, W_if.t()) + b_if
                f_hidden = torch.mm(h_current_2d, W_hf.t()) + b_hf
                f = torch.sigmoid(f_input + f_hidden)
                
                # 候选门
                g_input = torch.mm(x_2d, W_ig.t()) + b_ig
                g_hidden = torch.mm(h_current_2d, W_hg.t()) + b_hg
                g = torch.tanh(g_input + g_hidden)
                
                # 输出门
                o_input = torch.mm(x_2d, W_io.t()) + b_io
                o_hidden = torch.mm(h_current_2d, W_ho.t()) + b_ho
                o = torch.sigmoid(o_input + o_hidden)
                
                c_current = f * c_current + i * g
                h_new = o * torch.tanh(c_current)
                
                # 检查是否回到定点附近
                if torch.norm(h_new - h_fp) < 0.01:
                    results['returned_to_fixed_point'] += 1
                    results['convergence_times'].append(step + 1)
                    converged = True
                    break
                
                h_current = h_new
            
            if not converged:
                results['escaped'] += 1
        
        return results
    
    def analyze_attractor_basins(self, task_name, fixed_points, num_trajectories=20):
        """分析吸引子盆地
        
        Args:
            task_name: str, 任务名称
            fixed_points: list, 定点列表
            num_trajectories: int, 轨迹数量
            
        Returns:
            basin_results: dict, 盆地分析结果
        """
        self.logger.log_info(f"开始分析任务 {task_name} 的吸引子盆地")
        
        basin_results = {
            'basin_sizes': [],
            'basin_boundaries': [],
            'trajectory_analysis': []
        }
        
        # 为每个定点计算盆地大小
        for i, fp in enumerate(fixed_points):
            self.logger.log_info(f"  分析定点 {i+1}/{len(fixed_points)} 的盆地...")
            basin_size = 0
            trajectories_to_fp = []
            
            for traj_idx in range(num_trajectories):
                if traj_idx % 5 == 0:  # 每5次轨迹输出一次进度
                    self.logger.log_info(f"    轨迹进度: {traj_idx}/{num_trajectories}")
                
                # 随机初始状态
                h0 = torch.randn(1, 1, self.n_hidden) * 2.0  # 更大的随机范围
                
                # 模拟轨迹
                trajectory = self._simulate_trajectory(h0, fp['x'])
                
                # 检查是否收敛到当前定点
                final_state = trajectory[-1]
                distances = [np.linalg.norm(final_state - fp['h'].squeeze()) for fp in fixed_points]
                closest_fp_idx = np.argmin(distances)
                
                if closest_fp_idx == i and distances[i] < 0.1:
                    basin_size += 1
                    trajectories_to_fp.append(trajectory)
            
            basin_size_ratio = basin_size / num_trajectories
            self.logger.log_info(f"  定点 {i+1} 盆地大小: {basin_size_ratio:.3f} ({basin_size}/{num_trajectories})")
            basin_results['basin_sizes'].append(basin_size_ratio)
            basin_results['trajectory_analysis'].append(trajectories_to_fp)
        
        self.attractor_basins[task_name] = basin_results
        self.logger.log_info(f"任务 {task_name} 盆地分析完成")
        
        return basin_results
    
    def _simulate_trajectory(self, h0, x, max_steps=20):
        """模拟轨迹
        
        Args:
            h0: torch.Tensor, 初始隐藏状态
            x: torch.Tensor, 固定输入
            max_steps: int, 最大步数
            
        Returns:
            trajectory: list, 轨迹
        """
        trajectory = []
        h = h0.clone()
        c = torch.zeros_like(h)
        
        for step in range(max_steps):
            trajectory.append(h.detach().numpy().squeeze())
            
            # LSTM前向传播（使用分离的门控权重）
            lstm = self.model.network.lstm
            W_ih = lstm.weight_ih_l0.data
            W_hh = lstm.weight_hh_l0.data
            b_ih = lstm.bias_ih_l0.data
            b_hh = lstm.bias_hh_l0.data
            
            # 分离门控权重
            W_ii, W_if, W_ig, W_io = W_ih.chunk(4, 0)
            W_hi, W_hf, W_hg, W_ho = W_hh.chunk(4, 0)
            b_ii, b_if, b_ig, b_io = b_ih.chunk(4, 0)
            b_hi, b_hf, b_hg, b_ho = b_hh.chunk(4, 0)
            
            # 确保输入和隐藏状态的维度正确
            # 将numpy数组转换为torch张量
            if isinstance(x, np.ndarray):
                x = torch.tensor(x, dtype=torch.float32)
            
            x_2d = x.squeeze(0) if x.dim() == 3 else x
            h_2d = h.squeeze(0) if h.dim() == 3 else h
            
            # 计算门控值
            # 输入门
            i_input = torch.mm(x_2d, W_ii.t()) + b_ii
            i_hidden = torch.mm(h_2d, W_hi.t()) + b_hi
            i = torch.sigmoid(i_input + i_hidden)
            
            # 遗忘门
            f_input = torch.mm(x_2d, W_if.t()) + b_if
            f_hidden = torch.mm(h_2d, W_hf.t()) + b_hf
            f = torch.sigmoid(f_input + f_hidden)
            
            # 候选门
            g_input = torch.mm(x_2d, W_ig.t()) + b_ig
            g_hidden = torch.mm(h_2d, W_hg.t()) + b_hg
            g = torch.tanh(g_input + g_hidden)
            
            # 输出门
            o_input = torch.mm(x_2d, W_io.t()) + b_io
            o_hidden = torch.mm(h_2d, W_ho.t()) + b_ho
            o = torch.sigmoid(o_input + o_hidden)
            
            c = f * c + i * g
            h = o * torch.tanh(c)
        
        return trajectory
    
    def visualize_fixed_points(self, task_name, fixed_points, stability_results=None, 
                              save_path=None, figsize=(12, 8)):
        """可视化定点分析结果
        
        Args:
            task_name: str, 任务名称
            fixed_points: list, 定点列表
            stability_results: dict, 稳定性分析结果
            save_path: str, 保存路径
            figsize: tuple, 图形大小
        """
        self.logger.log_info(f"开始可视化任务 {task_name} 的定点分析结果")
        
        if not fixed_points:
            self.logger.warning(f"任务 {task_name} 没有找到定点")
            return
        
        # 提取隐藏状态
        h_states = np.array([fp['h'].squeeze() for fp in fixed_points])
        
        # 降维到2D
        if h_states.shape[1] > 2:
            pca = PCA(n_components=2)
            h_2d = pca.fit_transform(h_states)
            self.logger.log_info(f"PCA解释方差比: {pca.explained_variance_ratio_}")
        else:
            h_2d = h_states
        
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. 定点分布
        ax1 = axes[0, 0]
        colors = []
        sizes = []
        
        for i, fp in enumerate(fixed_points):
            if stability_results and i < len(stability_results['stability_scores']):
                is_stable = stability_results['stability_scores'][i]['is_stable']
                colors.append('green' if is_stable else 'red')
                sizes.append(100 if is_stable else 50)
            else:
                colors.append('blue')
                sizes.append(75)
        
        scatter = ax1.scatter(h_2d[:, 0], h_2d[:, 1], c=colors, s=sizes, alpha=0.7)
        ax1.set_title(f'{self.task_names[task_name]} Fixed Points')
        ax1.set_xlabel('PC1')
        ax1.set_ylabel('PC2')
        
        # 添加图例
        if stability_results:
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='green', alpha=0.7, label='Stable'),
                Patch(facecolor='red', alpha=0.7, label='Unstable')
            ]
            ax1.legend(handles=legend_elements)
        
        # 2. 收敛误差分布
        ax2 = axes[0, 1]
        errors = [fp['error'] for fp in fixed_points]
        ax2.hist(errors, bins=20, alpha=0.7, color='skyblue')
        ax2.set_title('Convergence Errors')
        ax2.set_xlabel('Error')
        ax2.set_ylabel('Count')
        ax2.set_yscale('log')
        
        # 3. 迭代次数分布
        ax3 = axes[1, 0]
        iterations = [fp['iterations'] for fp in fixed_points]
        ax3.hist(iterations, bins=20, alpha=0.7, color='lightgreen')
        ax3.set_title('Convergence Iterations')
        ax3.set_xlabel('Iterations')
        ax3.set_ylabel('Count')
        
        # 4. 稳定性分析（如果有）
        ax4 = axes[1, 1]
        if stability_results:
            stable_count = len(stability_results['stable_points'])
            unstable_count = len(stability_results['unstable_points'])
            total_count = stable_count + unstable_count
            
            labels = ['Stable', 'Unstable']
            sizes = [stable_count, unstable_count]
            colors = ['green', 'red']
            
            ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax4.set_title('Fixed Point Stability')
        else:
            ax4.text(0.5, 0.5, 'No stability analysis\navailable', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Stability Analysis')
        
        plt.tight_layout()
        
        # 保存图片
        if save_path is None:
            save_path = os.path.join(self.model_dir, f'fixed_points_{task_name}.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self.logger.log_info(f"定点分析图已保存到: {save_path}")
        plt.show()
    
    def analyze_trajectories(self, task_name, fixed_points, num_trajectories=100):
        """分析轨迹特性
        
        Args:
            task_name: str, 任务名称
            fixed_points: list, 定点列表
            num_trajectories: int, 轨迹数量
            
        Returns:
            trajectory_results: dict, 轨迹分析结果
        """
        self.logger.log_info(f"开始分析任务 {task_name} 的轨迹特性")
        
        trajectory_results = {
            'convergence_times': [],
            'trajectory_lengths': [],
            'final_states': [],
            'attraction_strength': []
        }
        
        for fp in fixed_points:
            fp_convergence_times = []
            fp_trajectory_lengths = []
            fp_final_states = []
            
            for _ in range(num_trajectories):
                # 随机初始状态
                h0 = torch.randn(1, 1, self.n_hidden) * 1.0
                
                # 模拟轨迹
                trajectory = self._simulate_trajectory(h0, fp['x'])
                
                # 分析轨迹
                final_state = trajectory[-1]
                fp_final_states.append(final_state)
                
                # 计算收敛时间
                convergence_time = self._compute_convergence_time(trajectory, fp['h'].squeeze())
                fp_convergence_times.append(convergence_time)
                
                # 计算轨迹长度
                trajectory_length = self._compute_trajectory_length(trajectory)
                fp_trajectory_lengths.append(trajectory_length)
            
            trajectory_results['convergence_times'].append(fp_convergence_times)
            trajectory_results['trajectory_lengths'].append(fp_trajectory_lengths)
            trajectory_results['final_states'].append(fp_final_states)
            
            # 计算吸引力强度（收敛到该定点的比例）
            attraction_strength = np.mean([t < 50 for t in fp_convergence_times])  # 50步内收敛
            trajectory_results['attraction_strength'].append(attraction_strength)
        
        self.trajectory_analysis[task_name] = trajectory_results
        self.logger.log_info(f"任务 {task_name} 轨迹分析完成")
        
        return trajectory_results
    
    def _compute_convergence_time(self, trajectory, target_state, threshold=0.01):
        """计算收敛时间
        
        Args:
            trajectory: list, 轨迹
            target_state: numpy.ndarray, 目标状态
            threshold: float, 收敛阈值
            
        Returns:
            convergence_time: int, 收敛时间步数
        """
        for i, state in enumerate(trajectory):
            distance = np.linalg.norm(state - target_state)
            if distance < threshold:
                return i + 1
        return len(trajectory)  # 未收敛
    
    def _compute_trajectory_length(self, trajectory):
        """计算轨迹长度
        
        Args:
            trajectory: list, 轨迹
            
        Returns:
            length: float, 轨迹长度
        """
        length = 0.0
        for i in range(1, len(trajectory)):
            length += np.linalg.norm(trajectory[i] - trajectory[i-1])
        return length
    
    def compare_tasks(self, tasks=None):
        """比较不同任务的定点特性
        
        Args:
            tasks: list, 要比较的任务列表
            
        Returns:
            comparison_results: dict, 比较结果
        """
        if tasks is None:
            tasks = self.tasks
        
        self.logger.log_info(f"开始比较任务: {tasks}")
        
        comparison_results = {
            'num_fixed_points': [],
            'stability_ratios': [],
            'avg_convergence_times': [],
            'avg_attraction_strengths': []
        }
        
        for task in tasks:
            # 使用已经计算的结果
            if task in self.fixed_points:
                fixed_points = self.fixed_points[task]
                
                if fixed_points:
                    # 记录结果
                    comparison_results['num_fixed_points'].append(len(fixed_points))
                    
                    # 计算稳定性比例
                    if task in self.stability_analysis:
                        stability_results = self.stability_analysis[task]
                        stable_ratio = len(stability_results['stable_points']) / len(fixed_points)
                        comparison_results['stability_ratios'].append(stable_ratio)
                    else:
                        comparison_results['stability_ratios'].append(0.0)
                    
                    # 计算平均收敛时间和吸引力强度
                    if task in self.trajectory_analysis:
                        trajectory_results = self.trajectory_analysis[task]
                        avg_convergence = np.mean([np.mean(times) for times in trajectory_results['convergence_times']])
                        comparison_results['avg_convergence_times'].append(avg_convergence)
                        
                        avg_attraction = np.mean(trajectory_results['attraction_strength'])
                        comparison_results['avg_attraction_strengths'].append(avg_attraction)
                    else:
                        comparison_results['avg_convergence_times'].append(0.0)
                        comparison_results['avg_attraction_strengths'].append(0.0)
                else:
                    comparison_results['num_fixed_points'].append(0)
                    comparison_results['stability_ratios'].append(0.0)
                    comparison_results['avg_convergence_times'].append(0.0)
                    comparison_results['avg_attraction_strengths'].append(0.0)
            else:
                comparison_results['num_fixed_points'].append(0)
                comparison_results['stability_ratios'].append(0.0)
                comparison_results['avg_convergence_times'].append(0.0)
                comparison_results['avg_attraction_strengths'].append(0.0)
        
        return comparison_results
    
    def plot_task_comparison(self, comparison_results, tasks=None, save_path=None):
        """绘制任务比较图
        
        Args:
            comparison_results: dict, 比较结果
            tasks: list, 任务列表
            save_path: str, 保存路径
        """
        if tasks is None:
            tasks = self.tasks
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 定点数量比较
        ax1 = axes[0, 0]
        task_names = [self.task_names[task] for task in tasks]
        ax1.bar(task_names, comparison_results['num_fixed_points'], 
               color=[self.task_colors[task] for task in tasks], alpha=0.7)
        ax1.set_title('Number of Fixed Points')
        ax1.set_ylabel('Count')
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # 2. 稳定性比例比较
        ax2 = axes[0, 1]
        ax2.bar(task_names, comparison_results['stability_ratios'], 
               color=[self.task_colors[task] for task in tasks], alpha=0.7)
        ax2.set_title('Stability Ratio')
        ax2.set_ylabel('Ratio')
        ax2.set_ylim(0, 1)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # 3. 平均收敛时间比较
        ax3 = axes[1, 0]
        ax3.bar(task_names, comparison_results['avg_convergence_times'], 
               color=[self.task_colors[task] for task in tasks], alpha=0.7)
        ax3.set_title('Average Convergence Time')
        ax3.set_ylabel('Steps')
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
        
        # 4. 平均吸引力强度比较
        ax4 = axes[1, 1]
        ax4.bar(task_names, comparison_results['avg_attraction_strengths'], 
               color=[self.task_colors[task] for task in tasks], alpha=0.7)
        ax4.set_title('Average Attraction Strength')
        ax4.set_ylabel('Strength')
        ax4.set_ylim(0, 1)
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        # 保存图片
        if save_path is None:
            save_path = os.path.join(self.model_dir, 'task_comparison_fixed_points.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self.logger.log_info(f"任务比较图已保存到: {save_path}")
        plt.show()
    
    def save_analysis_results(self, save_path=None):
        """保存分析结果
        
        Args:
            save_path: str, 保存路径
        """
        if save_path is None:
            save_path = os.path.join(self.model_dir, 'fixed_point_analysis_results.json')
        
        # 准备保存的数据
        save_data = {
            'fixed_points': {},
            'stability_analysis': {},
            'attractor_basins': {},
            'trajectory_analysis': {}
        }
        
        # 转换numpy数组为列表以便JSON序列化
        for task in self.tasks:
            if task in self.fixed_points:
                save_data['fixed_points'][task] = []
                for fp in self.fixed_points[task]:
                    fp_save = {
                        'h': fp['h'].tolist(),
                        'c': fp['c'].tolist(),
                        'x': fp['x'].tolist(),
                        'error': fp['error'],
                        'iterations': fp['iterations']
                    }
                    save_data['fixed_points'][task].append(fp_save)
            
            if task in self.stability_analysis:
                # 转换numpy数组为列表以便JSON序列化
                stability_data = self.stability_analysis[task].copy()
                for score in stability_data['stability_scores']:
                    if 'eigenvalues' in score:
                        score['eigenvalues'] = score['eigenvalues'].tolist()
                save_data['stability_analysis'][task] = stability_data
            
            if task in self.attractor_basins:
                save_data['attractor_basins'][task] = self.attractor_basins[task]
            
            if task in self.trajectory_analysis:
                save_data['trajectory_analysis'][task] = self.trajectory_analysis[task]
        
        # 保存到文件
        with open(save_path, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        self.logger.log_info(f"分析结果已保存到: {save_path}")


def analyze_fixed_points(model_path, hp_path=None, tasks=None, save_results=True):
    """分析眼跳任务定点的便捷函数
    
    Args:
        model_path: str, 模型文件路径
        hp_path: str, 超参数文件路径
        tasks: list, 要分析的任务列表
        save_results: bool, 是否保存结果
        
    Returns:
        fpa: FixedPointAnalysis对象
        comparison_results: dict, 比较结果
    """
    logger = AnalysisLogger("fixed_point_analysis_main")
    
    logger.log_info(f"开始分析眼跳任务定点")
    logger.log_info(f"模型路径: {model_path}")
    
    if tasks is None:
        tasks = ['pro_saccade', 'anti_saccade', 'delay_pro', 'delay_anti']
    
    # 创建分析对象
    fpa = FixedPointAnalysis(model_path, hp_path)
    
    # 为每个任务生成详细分析
    for task in tasks:
        logger.log_info(f"分析任务: {task}")
        
        # 寻找定点
        logger.log_info(f"开始寻找任务 {task} 的定点...")
        fixed_points = fpa.find_fixed_points(task, num_trials=20)
        logger.log_info(f"任务 {task} 定点寻找完成，找到 {len(fixed_points)} 个定点")
        
        if fixed_points:
            # 分析稳定性
            stability_results = fpa.analyze_stability(task, fixed_points)
            
            # 分析吸引子盆地
            basin_results = fpa.analyze_attractor_basins(task, fixed_points)
            
            # 可视化结果
            fpa.visualize_fixed_points(task, fixed_points, stability_results)
        else:
            logger.log_warning(f"任务 {task} 没有找到定点")
    
    # 比较任务（使用已经计算的结果）
    comparison_results = fpa.compare_tasks(tasks)
    
    # 绘制比较图
    fpa.plot_task_comparison(comparison_results, tasks)
    
    # 保存结果
    if save_results:
        fpa.save_analysis_results()
    
    logger.log_info("眼跳任务定点分析完成")
    return fpa, comparison_results


if __name__ == "__main__":
    # 示例使用
    model_path = "../checkpoints/random_experiment_20250814_104210/model/model_final.pth"
    
    if os.path.exists(model_path):
        try:
            # 分析所有任务的定点
            fpa, comparison_results = analyze_fixed_points(
                model_path=model_path,
                tasks=['pro_saccade', 'anti_saccade', 'delay_pro', 'delay_anti']
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
    else:
        print(f"模型文件不存在: {model_path}")
        print("请先训练模型或检查模型路径")