"""
眼跳任务集分析模块
分析刺激平均活动的状态空间，用于理解不同眼跳任务的神经表示
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE, Isomap
import json
from pathlib import Path
import sys

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from task import generate_trials
from model import CustomSaccadeModel
from utils.logger import get_logger
from utils.config import HyperParameters


class TaskSetAnalysis:
    """眼跳任务集分析类，用于分析PyTorch模型中的任务表示"""

    def __init__(self, model_path, hp_path=None):
        """初始化任务集分析
        
        Args:
            model_path: str, 模型文件路径
            hp_path: str, 超参数文件路径，如果为None则从model_path同目录查找
        """
        # 设置日志
        self.logger = get_logger(__name__)
        self.logger.info("开始初始化任务集分析")
        
        # 加载模型和超参数
        self.model_path = model_path
        self.model_dir = str(Path(model_path).parent)
        
        if hp_path is None:
            hp_path = os.path.join(self.model_dir, 'hp.json')
        
        self.logger.info(f"加载超参数文件: {hp_path}")
        with open(hp_path, 'r') as f:
            model_hp = json.load(f)
        
        # 从config.py加载默认的任务参数
        default_hp = HyperParameters()
        default_hp_dict = default_hp.to_dict()
        
        # 合并参数：模型参数优先，缺失的用默认值补充
        self.hp = default_hp_dict.copy()
        self.hp.update(model_hp)  # 模型参数覆盖默认值
        
        self.logger.info(f"模型参数: {model_hp}")
        self.logger.info(f"完整超参数: {self.hp}")
        
        # 加载模型
        self.logger.info("加载PyTorch模型")
        
        # 添加随机数生成器到超参数（用于trial生成）
        self.hp['rng'] = np.random.RandomState(self.hp.get('random_seed', 42))
        
        # 为模型初始化创建可序列化的hp副本（移除rng）
        model_hp_serializable = self.hp.copy()
        del model_hp_serializable['rng']
        
        # 初始化模型
        self.model = CustomSaccadeModel(
            model_dir=self.model_dir,
            hp=model_hp_serializable
        )
        
        # 加载训练好的权重
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        
        # 检查是否是完整的检查点还是单独的state_dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # 完整检查点格式
            state_dict = checkpoint['model_state_dict']
            self.logger.info("检测到完整检查点格式，提取model_state_dict")
        else:
            # 单独的state_dict格式
            state_dict = checkpoint
            self.logger.info("检测到单独的state_dict格式")
        
        # 处理键名映射问题
        new_state_dict = {}
        for key, value in state_dict.items():
            # 如果键名不包含'network.'前缀，添加它
            if not key.startswith('network.'):
                new_key = f'network.{key}'
                new_state_dict[new_key] = value
                self.logger.info(f"映射键名: {key} -> {new_key}")
            else:
                new_state_dict[key] = value
        
        self.model.load_state_dict(new_state_dict)
        self.model.eval()
        self.logger.info("模型加载完成")
        
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
        self.h_stimavg_bytask = OrderedDict()
        self.h_stimavg_byepoch = OrderedDict()
        self.h_lastt_byepoch = OrderedDict()
        
        # 计算刺激平均活动
        self._compute_stimulus_averaged_activity()
        
        # 调试信息：显示所有可用的epochs
        available_epochs = set()
        for key in self.h_stimavg_byepoch.keys():
            available_epochs.add(key[1])
        self.logger.info(f"所有可用的epochs: {sorted(available_epochs)}")
    
    def _compute_stimulus_averaged_activity(self):
        """计算刺激平均活动"""
        self.logger.info("开始计算刺激平均活动")
        
        with torch.no_grad():
            for task in self.tasks:
                self.logger.info(f"处理任务: {task}")
                
                # 生成测试trial
                trial = generate_trials(task, self.hp, 'random', batch_size=32)
                
                # 获取隐藏层活动
                h_activities = self._get_hidden_activities(trial.x)
                
                # 平均跨刺激条件
                h_stimavg = h_activities.mean(axis=1)  # (time, hidden_dim)
                
                # 忽略初始过渡期，从100ms开始
                t_start = int(100 / self.hp.get('dt', 20))
                self.h_stimavg_bytask[task] = h_stimavg[t_start:, :]
                
                # 按epoch分组
                self._process_epochs(task, h_activities, trial)
        
        self.logger.info("刺激平均活动计算完成")
    
    def _get_hidden_activities(self, x):
        """获取隐藏层活动
        
        Args:
            x: 输入数据 (time, batch, input_dim)
            
        Returns:
            h_activities: 隐藏层活动 (time, batch, hidden_dim)
        """
        # 确保输入是torch tensor
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        
        # 前向传播获取隐藏层活动
        h_activities = self.model.get_hidden_activities(x)
        return h_activities.numpy()
    
    def _process_epochs(self, task, h_activities, trial):
        """处理不同epoch的活动
        
        Args:
            task: 任务名称
            h_activities: 隐藏层活动
            trial: trial对象
        """
        # 使用trial对象中已定义的epochs
        epochs = trial.epochs
        self.logger.info(f"任务 {task} 的epochs: {epochs}")
        
        # 计算每个epoch的活动
        h_stimavg = h_activities.mean(axis=1)  # (time, hidden_dim)
        
        for epoch_name, (start_t, end_t) in epochs.items():
            # 处理None值（表示开始或结束）
            if start_t is None:
                start_t = 0
            if end_t is None:
                end_t = h_stimavg.shape[0]
            
            if start_t < h_stimavg.shape[0] and end_t <= h_stimavg.shape[0]:
                # 存储整个epoch的活动
                self.h_stimavg_byepoch[(task, epoch_name)] = h_stimavg[start_t:end_t, :]
                # 存储epoch最后一个时间点的活动
                self.h_lastt_byepoch[(task, epoch_name)] = h_activities[end_t-1, :, :]
                self.logger.info(f"存储 {task} {epoch_name}: {start_t}-{end_t}")
            else:
                self.logger.warning(f"跳过 {task} {epoch_name}: 时间范围 {start_t}-{end_t} 超出数据范围 {h_stimavg.shape[0]}")
    
    def filter(self, h, tasks=None, epochs=None, non_tasks=None, non_epochs=None,
               get_lasttimepoint=True, get_timeaverage=False, **kwargs):
        """过滤数据
        
        Args:
            h: 数据字典
            tasks: 要包含的任务列表
            epochs: 要包含的epoch列表
            non_tasks: 要排除的任务列表
            non_epochs: 要排除的epoch列表
            get_lasttimepoint: 是否只取最后一个时间点
            get_timeaverage: 是否取时间平均
            
        Returns:
            h_new: 过滤后的数据字典
        """
        if get_lasttimepoint:
            self.logger.info('分析epoch的最后一个时间点')
        if get_timeaverage:
            self.logger.info('分析epoch的时间平均活动')
        
        h_new = OrderedDict()
        for key in h:
            task, epoch = key
            
            include_key = True
            if tasks is not None:
                include_key = include_key and (task in tasks)
            
            if epochs is not None:
                include_key = include_key and (epoch in epochs)
            
            if non_tasks is not None:
                include_key = include_key and (task not in non_tasks)
            
            if non_epochs is not None:
                include_key = include_key and (epoch not in non_epochs)
            
            if include_key:
                if get_lasttimepoint:
                    h_new[key] = h[key][np.newaxis, -1, :]
                elif get_timeaverage:
                    h_new[key] = np.mean(h[key], axis=0, keepdims=True)
                else:
                    h_new[key] = h[key]
        
        return h_new
    
    def compute_taskspace(self, tasks=None, epochs=None, dim_reduction_type='PCA', **kwargs):
        """计算任务空间
        
        Args:
            tasks: 要分析的任务列表
            epochs: 要分析的epoch列表
            dim_reduction_type: 降维方法 ('PCA', 'MDS', 'TSNE', 'IsoMap')
            
        Returns:
            h_trans: 降维后的数据字典
        """
        self.logger.info(f"开始计算任务空间，降维方法: {dim_reduction_type}")
        
        # 只取每个epoch的最后一个时间点
        h = self.filter(self.h_stimavg_byepoch, epochs=epochs, tasks=tasks, 
                       get_lasttimepoint=True, **kwargs)
        
        # 调试信息：显示过滤后的数据
        self.logger.info(f"过滤后的数据键: {list(h.keys())}")
        self.logger.info(f"请求的epochs: {epochs}")
        self.logger.info(f"请求的tasks: {tasks}")
        
        # 连接所有数据
        data = np.concatenate(list(h.values()), axis=0)
        data = data.astype(dtype='float64')
        
        self.logger.info(f"数据形状: {data.shape}")
        
        # 选择降维方法
        if dim_reduction_type == 'PCA':
            model = PCA(n_components=2)
            self.logger.info("使用PCA降维")
        elif dim_reduction_type == 'MDS':
            model = MDS(n_components=2, metric=True, random_state=0)
            self.logger.info("使用MDS降维")
        elif dim_reduction_type == 'TSNE':
            model = TSNE(n_components=2, init='pca', 
                        verbose=1, method='exact', learning_rate=100, perplexity=5)
            self.logger.info("使用t-SNE降维")
        elif dim_reduction_type == 'IsoMap':
            model = Isomap(n_components=2)
            self.logger.info("使用IsoMap降维")
        else:
            raise ValueError(f'未知的降维方法: {dim_reduction_type}')
        
        # 降维变换
        data_trans = model.fit_transform(data)
        self.logger.info("降维完成")
        
        # 打包回字典
        h_trans = OrderedDict()
        i_start = 0
        for key, val in h.items():
            i_end = i_start + val.shape[0]
            h_trans[key] = data_trans[i_start:i_end, :]
            i_start = i_end
        
        return h_trans
    
    def plot_taskspace(self, h_trans, epochs=None, dim_reduction_type='PCA',
                       plot_text=True, figsize=(10, 8), markersize=10, plot_label=True,
                       save_path=None):
        """绘制任务空间
        
        Args:
            h_trans: 降维后的数据
            epochs: epoch列表
            dim_reduction_type: 降维方法
            plot_text: 是否显示文本标签
            figsize: 图形大小
            markersize: 标记大小
            plot_label: 是否显示轴标签
            save_path: 保存路径
        """
        self.logger.info("开始绘制任务空间图")
        
        # 形状映射
        shape_mapping = {
            'task_signal': 'o',
            'stimulus': 's', 
            'delay': '^',
            'response': 'd'
        }
        
        fs = 12  # 字体大小
        dim0, dim1 = (0, 1)  # 绘制的维度
        
        texts = []
        
        fig, ax = plt.subplots(figsize=figsize)
        
        for key, val in h_trans.items():
            task, epoch = key
            
            # 获取颜色
            color = self.task_colors[task]
            
            # 绘制最后一个时间点
            ax.plot(val[-1, dim0], val[-1, dim1], shape_mapping.get(epoch, 'o'),
                    color=color, mec=color, mew=2.0, ms=markersize, 
                    label=f"{self.task_names[task]} ({epoch})")
                        
            # 绘制轨迹（如果不是task_signal epoch）
            if epoch != 'task_signal' and val.shape[0] > 1:
                ax.plot(val[:, dim0], val[:, dim1], color=color, alpha=0.3, linewidth=1)
        
        # 设置轴标签
        if plot_label:
            if dim_reduction_type == 'PCA':
                xlabel = f'PC {dim0+1}'
                ylabel = f'PC {dim1+1}'
            else:
                xlabel = f'{dim_reduction_type} dim. {dim0+1}'
                ylabel = f'{dim_reduction_type} dim. {dim1+1}'
            ax.set_xlabel(xlabel, fontsize=fs)
            ax.set_ylabel(ylabel, fontsize=fs)
        
        ax.tick_params(axis='both', which='major', labelsize=fs)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        
        # 添加图例
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), 
                 loc='upper right', fontsize=10, frameon=False)
        
        plt.tight_layout()
        
        # 保存图片
        if save_path is None:
            save_name = f'taskspace_{dim_reduction_type}'
            if epochs is not None:
                save_name += '_' + '_'.join(epochs)
            save_path = os.path.join(self.model_dir, f'{save_name}.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"任务空间图已保存到: {save_path}")
        plt.show()
    
    def compute_and_plot_taskspace(self, tasks=None, epochs=None, **kwargs):
        """计算并绘制任务空间
        
        Args:
            tasks: 任务列表
            epochs: epoch列表
            **kwargs: 其他参数
        """
        h_trans = self.compute_taskspace(tasks=tasks, epochs=epochs, **kwargs)
        self.plot_taskspace(h_trans, epochs=epochs, **kwargs)
        return h_trans


def analyze_saccade_taskspace(model_path, hp_path=None, epochs=None, 
                             dim_reduction_type='PCA', save_path=None):
    """分析眼跳任务空间的便捷函数
    
    Args:
        model_path: 模型文件路径
        hp_path: 超参数文件路径
        epochs: 要分析的epoch列表，如果为None则分析所有epoch
        dim_reduction_type: 降维方法
        save_path: 保存路径
        
    Returns:
        tsa: TaskSetAnalysis对象
        h_trans: 降维后的数据
    """
    logger = get_logger(__name__)
    logger.info(f"开始分析眼跳任务空间")
    logger.info(f"模型路径: {model_path}")
    logger.info(f"降维方法: {dim_reduction_type}")
    
    # 如果没有指定epochs，使用默认的
    if epochs is None:
        epochs = ['task_signal', 'stimulus', 'response']
        # 如果有延迟任务，替换为延迟任务的epochs
        tasks = ['pro_saccade', 'anti_saccade', 'delay_pro', 'delay_anti']
        if any('delay' in task for task in tasks):
            epochs = ['task_signal', 'delay', 'response']  # 延迟任务只有这三个epoch
    
    # 创建分析对象
    tsa = TaskSetAnalysis(model_path, hp_path)
    
    logger.info(f"分析的epochs: {epochs}")
    
    # 计算并绘制任务空间
    h_trans = tsa.compute_and_plot_taskspace(
        tasks=tsa.tasks, 
        epochs=epochs, 
        dim_reduction_type=dim_reduction_type,
        save_path=save_path
    )
    
    logger.info("眼跳任务空间分析完成")
    return tsa, h_trans


if __name__ == "__main__":
    # 示例使用
    model_path = "../checkpoints/random_experiment_20250814_104210/model/model_final.pth"
    
    if os.path.exists(model_path):
        # 分析任务空间（自动包含delay epoch）
        tsa, h_trans = analyze_saccade_taskspace(
            model_path=model_path,
            epochs=None,  # 使用自动检测，会包含delay epoch
            dim_reduction_type='PCA'
        )
        
        # 也可以尝试其他降维方法
        # tsa, h_trans = analyze_saccade_taskspace(
        #     model_path=model_path,
        #     epochs=['task_signal', 'stimulus', 'response'],
        #     dim_reduction_type='TSNE'
        # )
    else:
        print(f"模型文件不存在: {model_path}")
        print("请先训练模型或检查模型路径")
