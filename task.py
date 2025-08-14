"""眼跳实验的任务定义."""

import numpy as np


# 定义所有自定义任务
TASK_NAMES = ['pro_saccade', 'anti_saccade', 'delay_pro', 'delay_anti']

# 统一定义任务时间片
TASK_TIMES = {
    # 非延迟任务（pro_saccade, anti_saccade）
    'non_delay': {
        'task_signal_dur': 500,  # 任务信号持续时间（ms）
        'stim_dur': 500,         # 点刺激持续时间（ms）
        'response_dur': 500      # 反应持续时间（ms）
    },
    # 延迟任务（delay_pro, delay_anti）
    'delay': {
        'task_signal_dur': 500,  # 任务信号持续时间（ms）
        'response_dur': 500      # 反应持续时间（ms）
        # 延迟持续时间是随机的（1000-2000ms），不在这里定义
    }
}


def get_dist(original_dist):
    """获取周期性边界条件下的距离."""
    return np.minimum(np.abs(original_dist), 2*np.pi - np.abs(original_dist))


def get_task_times(task_type):
    """根据任务类型获取对应的时间片配置
    
    Args:
        task_type: str, 任务类型名称
        
    Returns:
        dict: 对应的时间片配置
    """
    if task_type in ['delay_pro', 'delay_anti']:
        return TASK_TIMES['delay']
    else:
        return TASK_TIMES['non_delay']


def _direction_to_angle(direction):
    """将方向标签(0=左, 1=右)映射为角度值(弧度)。

    约定：右=0, 左=pi。
    支持标量或numpy数组。
    """
    if hasattr(direction, '__iter__'):
        direction = np.asarray(direction)
        return np.where(direction.astype(int) == 1, 0.0, np.pi).astype(np.float32)
    else:
        return 0.0 if int(direction) == 1 else np.pi


def _compute_response_angles(rule_name, point_locs, error_types, rng=None):
    """根据任务类型与实验错误类型，返回每个trial目标扫视角度。

    返回值为float数组，若该trial无明确扫视方向标签，则用np.nan表示。

    规则说明：
    - 正确(0)：输出为正确响应方向
    - 反向扫视(6)：输出为与正确方向相反
    - 无注视但方向相反(7)：方向取与正确方向相反
    - 无注视但方向相同(8)：方向取正确方向
    - 其他方向(9)：方向取与刺激正交方向 (stim + pi/2)
    - 2,3,4：视为无注视，且无明确扫视方向 -> 返回nan
    """
    point_locs = np.asarray(point_locs, dtype=np.float32)
    error_types = np.asarray(error_types, dtype=int)

    # 正确方向：pro/delay_pro 与刺激同向；anti/delay_anti 与刺激反向
    if rule_name in ['pro_saccade', 'delay_pro']:
        correct = point_locs.copy()
    elif rule_name in ['anti_saccade', 'delay_anti']:
        correct = (point_locs + np.pi) % (2 * np.pi)
    else:
        raise ValueError('未知任务类型: ' + str(rule_name))

    resp = np.full_like(correct, np.nan, dtype=np.float32)

    # 0: 正确
    mask0 = (error_types == 0)
    resp[mask0] = correct[mask0]

    # 6: 反向扫视 (相对正确方向取反)
    mask6 = (error_types == 6)
    resp[mask6] = (correct[mask6] + np.pi) % (2 * np.pi)

    # 7: 无注视且方向相反 -> 与正确方向相反
    mask7 = (error_types == 7)
    resp[mask7] = (correct[mask7] + np.pi) % (2 * np.pi)

    # 8: 无注视但方向相同 -> 取正确方向
    mask8 = (error_types == 8)
    resp[mask8] = correct[mask8]

    # 9: 扫视其他方向 -> 与刺激正交，随机选择 +pi/2 或 -pi/2
    mask9 = (error_types == 9)
    if np.any(mask9):
        if rng is None:
            rng = np.random.RandomState(0)
        signs = rng.choice([-1.0, 1.0], size=np.sum(mask9)).astype(np.float32)
        resp[mask9] = (point_locs[mask9] + signs * (np.pi/2.0)) % (2 * np.pi)

    # 2,3,4: 无注视，且无明确扫视方向 -> 保持为nan
    return resp


class Trial(object):
    """表示一批试验的类."""

    def __init__(self, config, tdim, batch_size):
        """一批试验.

        Args:
            config: 配置字典
            tdim: int, 时间步数
            batch_size: int, 批次大小
        """
        self.float_type = 'float32'
        self.config = config
        self.dt = self.config['dt']

        self.n_eachring = self.config['n_eachring']
        self.n_input = self.config['n_input']
        self.n_output = self.config['n_output']
        self.pref = np.arange(0, 2*np.pi, 2*np.pi/self.n_eachring)  # 偏好方向

        self.batch_size = batch_size
        self.tdim = tdim
        self.x = np.zeros((tdim, batch_size, self.n_input), dtype=self.float_type)
        self.y = np.zeros((tdim, batch_size, self.n_output), dtype=self.float_type)
        if self.config['loss_type'] == 'lsq':
            self.y[:, :, :] = 0.05
        # y_loc 是输出的刺激位置，-1表示注视，(0,2 pi)表示反应
        self.y_loc = -np.ones((tdim, batch_size), dtype=self.float_type)

        self._sigma_x = config['sigma_x'] * np.sqrt(2 / config['alpha'])

    def expand(self, var):
        """将int/float扩展为列表."""
        if not hasattr(var, '__iter__'):
            var = [var] * self.batch_size
        return var

    def add(self, loc_type, locs=None, ons=None, offs=None, strengths=1, mods=None):
        """添加输入或刺激输出.

        Args:
            loc_type: str (fix_in, stim, fix_out, out), 要添加的信息类型
            locs: array of list of float (batch_size,), 要添加的位置，仅用于loc_type=stim或out
            ons: int or list, 开始时间的索引
            offs: int or list, 结束时间的索引
            strengths: float or list, 输入或目标输出的强度
            mods: int or list, 输入或目标输出的模态
        """

        ons = self.expand(ons)
        offs = self.expand(offs)
        strengths = self.expand(strengths)
        mods = self.expand(mods)

        for i in range(self.batch_size):
            if loc_type == 'fix_in':
                self.x[ons[i]: offs[i], i, 0] = 1
            elif loc_type == 'stim':
                # 刺激输入：使用第一个环 (位置1-32)
                stim_range = slice(1, 1+self.n_eachring)
                x_loc = self.add_x_loc(locs[i]) * strengths[i]
                # 修复广播问题：确保x_loc是正确形状
                if x_loc.ndim == 1:
                    self.x[ons[i]: offs[i], i, stim_range] += x_loc
                else:
                    self.x[ons[i]: offs[i], i, stim_range] += x_loc.flatten()
            elif loc_type == 'fix_out':
                # 注意这里不应该设置为1，因为输出是逻辑函数且在1处饱和
                if self.config['loss_type'] == 'lsq':
                    self.y[ons[i]: offs[i], i, 0] = 0.8
                else:
                    self.y[ons[i]: offs[i], i, 0] = 1.0
            elif loc_type == 'out':
                if self.config['loss_type'] == 'lsq':
                    y_loc = self.add_y_loc(locs[i]) * strengths[i]
                    # 修复广播问题：确保y_loc是正确形状
                    if y_loc.ndim == 1:
                        self.y[ons[i]: offs[i], i, 1:] += y_loc
                    else:
                        self.y[ons[i]: offs[i], i, 1:] += y_loc.flatten()
                else:
                    y_tmp = self.add_y_loc(locs[i])
                    y_tmp /= np.sum(y_tmp)
                    # 修复广播问题：确保y_tmp是正确形状
                    if y_tmp.ndim == 1:
                        self.y[ons[i]: offs[i], i, 1:] += y_tmp
                    else:
                        self.y[ons[i]: offs[i], i, 1:] += y_tmp.flatten()
                self.y_loc[ons[i]: offs[i], i] = locs[i]
            else:
                raise ValueError('未知的loc_type')

    def add_x_noise(self):
        """添加输入噪声."""
        self.x += self.config['rng'].randn(*self.x.shape) * self._sigma_x

    def add_c_mask(self, pre_offs, post_ons, task_type=None):
        """基于时间段的精细成本掩码控制
        
        通常有两个时期，反应前和反应后
        对反应后时期的比例进行缩放，使其总重要性与反应前时期相当
        
        Args:
            pre_offs: 反应前时期结束时间
            post_ons: 反应后时期开始时间
            task_type: 任务类型，用于精细控制
        """

        pre_on = int(100/self.dt)  # 从不检查前100ms
        pre_offs = self.expand(pre_offs)
        post_ons = self.expand(post_ons)

        if self.config['loss_type'] == 'lsq':
            c_mask = np.zeros((self.tdim, self.batch_size, self.n_output), dtype=self.float_type)
            for i in range(self.batch_size):
                # 反应后时期通常在各任务中长度相同
                c_mask[post_ons[i]:, i, :] = 5.
                
                # 基于任务类型的精细控制
                if task_type in ['delay_pro', 'delay_anti']:
                    # Delay任务的精细控制
                    task_times = get_task_times(task_type)
                    task_signal_end = int(task_times['task_signal_dur']/self.dt)
                    
                    # 任务信号期（0-500ms）：权重1.0
                    c_mask[pre_on:task_signal_end, i, :] = 1.0
                    
                    # 延迟期（500ms到反应开始）：权重3.0（关键改进）
                    # 这个阶段任务信号和点刺激同时存在
                    delay_start = task_signal_end
                    delay_end = pre_offs[i]
                    if delay_end > delay_start:
                        c_mask[delay_start:delay_end, i, :] = 3.0
                else:
                    # Pro/Anti任务的标准控制
                    # 反应前时期统一权重1.0
                    c_mask[pre_on:pre_offs[i], i, :] = 1.0

            # 注视很重要，保持权重2.0
            c_mask[:, :, 0] *= 2.
            
            # 归一化以保持loss一致性
            # 计算每个batch的平均权重，然后归一化
            for i in range(self.batch_size):
                avg_weight = np.mean(c_mask[:, i, :])
                if avg_weight > 0:
                    c_mask[:, i, :] /= avg_weight

            self.c_mask = c_mask.reshape((self.tdim*self.batch_size, self.n_output))
        else:
            c_mask = np.zeros((self.tdim, self.batch_size), dtype=self.float_type)
            for i in range(self.batch_size):
                # 反应后时期通常在各任务中长度相同
                # 大于1的值鼓励网络实现更高性能
                c_mask[post_ons[i]:, i] = 5.
                
                # 基于任务类型的精细控制
                if task_type in ['delay_pro', 'delay_anti']:
                    # Delay任务的精细控制
                    task_times = get_task_times(task_type)
                    task_signal_end = int(task_times['task_signal_dur']/self.dt)
                    
                    # 任务信号期（0-500ms）：权重1.0
                    c_mask[pre_on:task_signal_end, i] = 1.0
                    
                    # 延迟期（500ms到反应开始）：权重3.0（关键改进）
                    # 这个阶段任务信号和点刺激同时存在
                    delay_start = task_signal_end
                    delay_end = pre_offs[i]
                    if delay_end > delay_start:
                        c_mask[delay_start:delay_end, i] = 3.0
                else:
                    # Pro/Anti任务的标准控制
                    c_mask[pre_on:pre_offs[i], i] = 1.0

            self.c_mask = c_mask.reshape((self.tdim*self.batch_size,))
            self.c_mask /= self.c_mask.mean()

    def add_task_signal(self, task_id, on=None, off=None, strength=1.):
        """添加任务信号（one-hot编码）.
        
        Args:
            task_id: int, 任务ID (0: pro_saccade, 1: anti_saccade, 2: delay_pro, 3: delay_anti)
            on: int, 开始时间
            off: int, 结束时间
            strength: float, 信号强度
        """
        # 任务信号编码在位置33-36（原来的规则输入位置）
        task_signal_pos = 33 + task_id
        self.x[on:off, :, task_signal_pos] = strength



    def add_x_loc(self, x_loc):
        """给定位置的输入活动."""
        dist = get_dist(x_loc-self.pref)  # 周期性边界
        dist /= np.pi/8
        return 0.8*np.exp(-dist**2/2)

    def add_y_loc(self, y_loc):
        """给定位置的目标响应."""
        dist = get_dist(y_loc-self.pref)  # 周期性边界
        if self.config['loss_type'] == 'lsq':
            dist /= np.pi/8
            y = 0.8*np.exp(-dist**2/2)
        else:
            # One-hot输出
            y = np.zeros_like(dist)
            ind = np.argmin(dist)
            y[ind] = 1.
        return y


def pro_saccade(config, mode, **kwargs):
    """
    Pro-saccade任务.
    
    第一阶段：注视中心出现任务信号（500ms）
    第二阶段：任务信号消失，出现点刺激（500ms）
    第三阶段：根据任务信号和点刺激位置执行眼跳
    
    Args:
        config: 配置字典
        mode: 'random' 或 'psychometric'
    """
    dt = config['dt']
    rng = config['rng']
    
    # 获取任务时间片配置
    task_times = get_task_times('pro_saccade')
    
    if mode == 'random':
        batch_size = kwargs['batch_size']
        
        # 点位置：左侧(pi)或右侧(0)
        point_locs = rng.choice([0, np.pi], batch_size)
        
        # 时间安排
        task_signal_dur = int(task_times['task_signal_dur']/dt)
        task_signal_off = task_signal_dur
        stim_dur = int(task_times['stim_dur']/dt)
        go_cue_on = task_signal_off + stim_dur
        response_dur = int(task_times['response_dur']/dt)
        tdim = go_cue_on + response_dur
        
    elif mode == 'psychometric':
        p = kwargs['params']
        # 将List[Tuple[int, int]]转换为字典格式
        if isinstance(p, list):
            # 如果是List[Tuple[int, int]]格式，转换为字典
            directions = [item[0] for item in p]
            error_types = [item[1] for item in p]
            p = {'directions': directions, 'error_types': error_types}
        
        # 输入方向标签 -> 角度 (1=右->0, 0=左->pi)
        point_locs = _direction_to_angle(p['directions'])
        error_types = np.asarray(p['error_types']).astype(int)

        task_signal_dur = int(task_times['task_signal_dur']/dt)
        task_signal_off = task_signal_dur
        stim_dur = int(task_times['stim_dur']/dt)
        go_cue_on = task_signal_off + stim_dur
        response_dur = int(task_times['response_dur']/dt)
        tdim = go_cue_on + response_dur

        batch_size = len(point_locs)
    else:
        raise ValueError('未知模式: ' + str(mode))
    
    check_ons = go_cue_on + int(100/dt)
    
    # 反应位置
    if mode == 'psychometric':
        response_locs = _compute_response_angles('pro_saccade', point_locs, error_types, rng)
    else:
        response_locs = point_locs
    
    trial = Trial(config, tdim, batch_size)
    
    # 第一阶段：任务信号（one-hot编码在位置33）
    trial.add('fix_in', offs=task_signal_off)  # 注视输入直到任务信号消失
    trial.add_task_signal(0, on=0, off=task_signal_off)  # Pro-saccade任务信号
    
    # 第二阶段：点刺激
    trial.add('stim', point_locs, ons=task_signal_off, offs=tdim)  # 点刺激持续到试验结束
    
    # 输出
    trial.add('fix_out', offs=go_cue_on)  # 注视输出直到go cue
    if mode == 'psychometric':
        # 先添加默认的 out（将 nan 映射到 0 角度，稍后会覆盖无明确扫视的trial）
        trial.add('out', np.nan_to_num(response_locs, nan=0.0), ons=go_cue_on)
        # 对无注视错误（7/8/2/3/4）在延迟阶段降低注视通道标签
        nofix_mask = np.isin(error_types, [7, 8, 2, 3, 4])
        if np.any(nofix_mask):
            for i, is_nofix in enumerate(nofix_mask):
                if is_nofix:
                    trial.y[go_cue_on:, i, 0] = 0.01  # 延迟阶段注视降低
        # 对无明确扫视（2/3/4）覆盖输出期标签：注视极小，环为中心（统一低基线）
        unknown_mask = np.isin(error_types, [2, 3, 4])
        if np.any(unknown_mask):
            for i, is_unknown in enumerate(unknown_mask):
                if is_unknown:
                    trial.y[go_cue_on:, i, 0] = 0.01
                    trial.y[go_cue_on:, i, 1:] = 0.05
                    trial.y_loc[go_cue_on:, i] = -1
    else:
        trial.add('out', response_locs, ons=go_cue_on)
    trial.add_c_mask(pre_offs=go_cue_on, post_ons=check_ons, task_type='pro_saccade')
    
    trial.epochs = {
        'task_signal': (None, task_signal_off),
        'stimulus': (task_signal_off, go_cue_on),
        'response': (go_cue_on, None)
    }
    
    return trial


def anti_saccade(config, mode, **kwargs):
    """
    Anti-saccade任务.
    
    第一阶段：注视中心出现任务信号（500ms）
    第二阶段：任务信号消失，出现点刺激（500ms）
    第三阶段：根据任务信号和点刺激位置执行眼跳（反方向）
    
    Args:
        config: 配置字典
        mode: 'random' 或 'psychometric'
        kwargs: 其他参数
            - batch_size: 批次大小 (mode='random'时需要)
    
    Returns:
        Trial实例
    """
    dt = config['dt']
    rng = config['rng']
    
    # 获取任务时间片配置
    task_times = get_task_times('anti_saccade')
    
    if mode == 'random':
        batch_size = kwargs['batch_size']
        
        # 点位置：左侧(pi)或右侧(0)
        point_locs = rng.choice([0, np.pi], batch_size)
        
        # 时间安排
        task_signal_dur = int(task_times['task_signal_dur']/dt)
        task_signal_off = task_signal_dur
        stim_dur = int(task_times['stim_dur']/dt)
        go_cue_on = task_signal_off + stim_dur
        response_dur = int(task_times['response_dur']/dt)
        tdim = go_cue_on + response_dur
        
    elif mode == 'psychometric':
        p = kwargs['params']
        # 将List[Tuple[int, int]]转换为字典格式
        if isinstance(p, list):
            # 如果是List[Tuple[int, int]]格式，转换为字典
            directions = [item[0] for item in p]
            error_types = [item[1] for item in p]
            p = {'directions': directions, 'error_types': error_types}
        
        point_locs = _direction_to_angle(p['directions'])
        error_types = np.asarray(p['error_types']).astype(int)

        task_signal_dur = int(task_times['task_signal_dur']/dt)
        task_signal_off = task_signal_dur
        stim_dur = int(task_times['stim_dur']/dt)
        go_cue_on = task_signal_off + stim_dur
        response_dur = int(task_times['response_dur']/dt)
        tdim = go_cue_on + response_dur

        batch_size = len(point_locs)
    else:
        raise ValueError('未知模式: ' + str(mode))
    
    check_ons = go_cue_on + int(100/dt)
    
    # 反应位置
    if mode == 'psychometric':
        response_locs = _compute_response_angles('anti_saccade', point_locs, error_types)
    else:
        response_locs = (point_locs + np.pi) % (2 * np.pi)
    
    trial = Trial(config, tdim, batch_size)
    
    # 第一阶段：任务信号（one-hot编码在位置34）
    trial.add('fix_in', offs=task_signal_off)  # 注视输入直到任务信号消失
    trial.add_task_signal(1, on=0, off=task_signal_off)  # Anti-saccade任务信号
    
    # 第二阶段：点刺激
    trial.add('stim', point_locs, ons=task_signal_off, offs=tdim)  # 点刺激持续到试验结束
    
    # 输出
    trial.add('fix_out', offs=go_cue_on)  # 注视输出直到go cue
    if mode == 'psychometric':
        trial.add('out', np.nan_to_num(response_locs, nan=0.0), ons=go_cue_on)
        unknown_mask = np.isin(error_types, [2, 3, 4])
        if np.any(unknown_mask):
            for i, is_unknown in enumerate(unknown_mask):
                if is_unknown:
                    trial.y[go_cue_on:, i, 0] = 0.01
                    trial.y[go_cue_on:, i, 1:] = 0.05
                    trial.y_loc[go_cue_on:, i] = -1
    else:
        trial.add('out', response_locs, ons=go_cue_on)  # go cue后的扫视输出(反方向)
    trial.add_c_mask(pre_offs=go_cue_on, post_ons=check_ons, task_type='anti_saccade')
    
    trial.epochs = {
        'task_signal': (None, task_signal_off),
        'stimulus': (task_signal_off, go_cue_on),
        'response': (go_cue_on, None)
    }
    
    return trial


def delay_pro(config, mode, **kwargs):
    """
    Delay pro-saccade任务.
    
    第一阶段：任务信号呈现期（0-500ms）
    第二阶段：任务信号+点刺激同时存在（500ms-延迟结束）
    第三阶段：只有点刺激，开始扫视（延迟结束-试验结束）
    
    Args:
        config: 配置字典
        mode: 'random' 或 'psychometric'
    """
    dt = config['dt']
    rng = config['rng']
    
    # 获取任务时间片配置
    task_times = get_task_times('delay_pro')
    
    if mode == 'random':
        batch_size = kwargs['batch_size']
        
        # 点位置：左侧(pi)或右侧(0)
        point_locs = rng.choice([0, np.pi], batch_size)
        
        # 时间安排
        task_signal_dur = int(task_times['task_signal_dur']/dt)
        delay_on = task_signal_dur  # 延迟开始时间点
        delay_dur = int(rng.choice([1000, 1500, 2000])/dt)  # 可变延迟
        go_cue_on = delay_on + delay_dur  # 任务信号消失，开始扫视
        tdim = go_cue_on + int(task_times['response_dur']/dt)
        
    elif mode == 'psychometric':
        p = kwargs['params']
        # 将List[Tuple[int, int]]转换为字典格式
        if isinstance(p, list):
            # 如果是List[Tuple[int, int]]格式，转换为字典
            directions = [item[0] for item in p]
            error_types = [item[1] for item in p]
            p = {'directions': directions, 'error_types': error_types}
        
        point_locs = _direction_to_angle(p['directions'])
        error_types = np.asarray(p['error_types']).astype(int)

        task_signal_dur = int(task_times['task_signal_dur']/dt)
        delay_on = task_signal_dur
        delay_dur = int(1500/dt)
        go_cue_on = delay_on + delay_dur
        tdim = go_cue_on + int(task_times['response_dur']/dt)

        batch_size = len(point_locs)
    else:
        raise ValueError('未知模式: ' + str(mode))
    
    check_ons = go_cue_on + int(100/dt)
    
    # 反应位置
    if mode == 'psychometric':
        response_locs = _compute_response_angles('delay_pro', point_locs, error_types, rng)
    else:
        response_locs = point_locs
    
    trial = Trial(config, tdim, batch_size)
    
    # 第一阶段：任务信号（one-hot编码在位置2）
    trial.add('fix_in', offs=go_cue_on)  # 注视输入直到延迟结束
    trial.add_task_signal(2, on=0, off=go_cue_on)  # Delay Pro-saccade任务信号持续到反应开始
    
    # 第二阶段：点刺激（从delay_on开始）
    trial.add('stim', point_locs, ons=delay_on, offs=tdim)  # 点刺激从延迟开始持续到试验结束
    
    # 输出
    trial.add('fix_out', offs=go_cue_on)  # 注视输出直到延迟结束
    if mode == 'psychometric':
        trial.add('out', np.nan_to_num(response_locs, nan=0.0), ons=go_cue_on)
        unknown_mask = np.isin(error_types, [2, 3, 4])
        if np.any(unknown_mask):
            for i, is_unknown in enumerate(unknown_mask):
                if is_unknown:
                    trial.y[go_cue_on:, i, 0] = 0.01
                    trial.y[go_cue_on:, i, 1:] = 0.05
                    trial.y_loc[go_cue_on:, i] = -1
    else:
        trial.add('out', response_locs, ons=go_cue_on)  # 开始提示后的扫视输出
    trial.add_c_mask(pre_offs=go_cue_on, post_ons=check_ons, task_type='delay_pro')
    
    trial.epochs = {
        'task_signal': (None, delay_on),
        'delay': (delay_on, go_cue_on),
        'response': (go_cue_on, None)
    }
    
    return trial


def delay_anti(config, mode, **kwargs):
    """
    Delay anti-saccade任务.
    
    第一阶段：任务信号呈现期（0-500ms）
    第二阶段：任务信号+点刺激同时存在（500ms-延迟结束）
    第三阶段：只有点刺激，开始扫视（延迟结束-试验结束）
    
    Args:
        config: 配置字典
        mode: 'random' 或 'psychometric'
    """
    dt = config['dt']
    rng = config['rng']
    
    # 获取任务时间片配置
    task_times = get_task_times('delay_anti')
    
    if mode == 'random':
        batch_size = kwargs['batch_size']
        
        # 点位置：左侧(pi)或右侧(0)
        point_locs = rng.choice([0, np.pi], batch_size)
        
        # 时间安排
        task_signal_dur = int(task_times['task_signal_dur']/dt)
        delay_on = task_signal_dur  # 延迟开始时间点
        delay_dur = int(rng.choice([1000, 1500, 2000])/dt)  # 可变延迟
        go_cue_on = delay_on + delay_dur  # 任务信号消失，开始扫视
        tdim = go_cue_on + int(task_times['response_dur']/dt)
        
    elif mode == 'psychometric':
        p = kwargs['params']
        # 将List[Tuple[int, int]]格式，转换为字典
        if isinstance(p, list):
            # 如果是List[Tuple[int, int]]格式，转换为字典
            directions = [item[0] for item in p]
            error_types = [item[1] for item in p]
            p = {'directions': directions, 'error_types': error_types}
        
        point_locs = _direction_to_angle(p['directions'])
        error_types = np.asarray(p['error_types']).astype(int)

        task_signal_dur = int(task_times['task_signal_dur']/dt)
        delay_on = task_signal_dur
        delay_dur = int(1500/dt)
        go_cue_on = delay_on + delay_dur
        tdim = go_cue_on + int(task_times['response_dur']/dt)

        batch_size = len(point_locs)
    else:
        raise ValueError('未知模式: ' + str(mode))
    
    check_ons = go_cue_on + int(100/dt)
    
    # 反应位置
    if mode == 'psychometric':
        response_locs = _compute_response_angles('delay_anti', point_locs, error_types)
    else:
        response_locs = (point_locs + np.pi) % (2 * np.pi)
    
    trial = Trial(config, tdim, batch_size)
    
    # 第一阶段：任务信号（one-hot编码在位置3）
    trial.add('fix_in', offs=go_cue_on)  # 注视输入直到延迟结束
    trial.add_task_signal(3, on=0, off=go_cue_on)  # Delay Anti-saccade任务信号持续到反应开始
    
    # 第二阶段：点刺激（从delay_on开始）
    trial.add('stim', point_locs, ons=delay_on, offs=tdim)  # 点刺激从延迟开始持续到试验结束
    
    # 输出
    trial.add('fix_out', offs=go_cue_on)  # 注视输出直到延迟结束
    if mode == 'psychometric':
        trial.add('out', np.nan_to_num(response_locs, nan=0.0), ons=go_cue_on)
        unknown_mask = np.isin(error_types, [2, 3, 4])
        if np.any(unknown_mask):
            for i, is_unknown in enumerate(unknown_mask):
                if is_unknown:
                    trial.y[go_cue_on:, i, 0] = 0.01
                    trial.y[go_cue_on:, i, 1:] = 0.05
                    trial.y_loc[go_cue_on:, i] = -1
    else:
        trial.add('out', response_locs, ons=go_cue_on)  # 开始提示后的扫视输出(反方向)
    trial.add_c_mask(pre_offs=go_cue_on, post_ons=check_ons, task_type='delay_anti')
    
    trial.epochs = {
        'task_signal': (None, delay_on),
        'delay': (delay_on, go_cue_on),
        'response': (go_cue_on, None)
    }
    
    return trial


# 将规则名称映射到函数
rule_mapping = {
    'pro_saccade': pro_saccade,
    'anti_saccade': anti_saccade,
    'delay_pro': delay_pro,
    'delay_anti': delay_anti
}

rule_name = {
    'pro_saccade': 'Pro Saccade',
    'anti_saccade': 'Anti Saccade',
    'delay_pro': 'Delay Pro',
    'delay_anti': 'Delay Anti'
}


def generate_trials(rule, hp, mode, noise_on=True, **kwargs):
    """生成一批数据.

    Args:
        rule: str, 这批数据的任务类型
        hp: 超参数字典
        mode: str, 生成模式。选项: random, psychometric
        noise_on: bool, 是否添加输入噪声

    Return:
        trial: Trial类实例，包含输入和目标输出
    """
    config = hp
    trial = rule_mapping[rule](config, mode, **kwargs)

    if noise_on:
        trial.add_x_noise()

    return trial