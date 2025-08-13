"""命令行接口模块，提供统一的命令行控制"""

import argparse
import sys
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import json

from .logger import setup_logger, get_logger
from .config import ConfigManager, ExperimentConfig


class CLIInterface:
    """命令行接口类"""
    
    def __init__(self):
        """初始化命令行接口"""
        self.parser = self._create_parser()
        self.config_manager = ConfigManager()
        self.logger = get_logger("cli")
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """创建命令行参数解析器"""
        parser = argparse.ArgumentParser(
            description="自定义眼跳任务多任务学习系统",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
  参数优先级（从高到低）:
    1. 命令行参数（最高优先级）
    2. 配置文件参数
    3. 默认配置参数（最低优先级）

  使用示例:
    # 使用随机数据进行训练
    python train.py train --mode random --phase1-epochs 400 --phase2-epochs 100 --batch-size 32
    
    # 使用真实数据进行训练
    python train.py train --mode psychometric --subject DD --phase1-epochs 50 --phase2-epochs 30
    
    # 使用配置文件训练（命令行参数会覆盖配置文件）
    python train.py train --mode random --config random_training --phase1-epochs 100 --phase2-epochs 50
    
    # 查看可用配置
    python train.py config list
    
    # 创建新配置
    python train.py config create my_experiment --base-config random_training
    
         # 评估模型
     python train.py evaluate --model-path checkpoints/model_final.pth
              """
        )
        
        # 主要命令组
        subparsers = parser.add_subparsers(dest='command', help='可用命令')
        
        # 训练命令
        train_parser = subparsers.add_parser('train', help='训练模型')
        self._add_training_args(train_parser)
        
        # 评估命令
        eval_parser = subparsers.add_parser('evaluate', help='评估模型')
        self._add_evaluation_args(eval_parser)
        
        # 配置管理命令
        config_parser = subparsers.add_parser('config', help='配置管理')
        self._add_config_args(config_parser)
        

        
        return parser
    
    def _add_training_args(self, parser: argparse.ArgumentParser):
        """添加训练相关参数"""
        # 训练模式
        parser.add_argument(
            '--mode', '-m',
            choices=['random', 'psychometric'],
            default='random',
            help='训练模式 (默认: random)'
        )
        
        # 数据相关
        parser.add_argument(
            '--data-dir', '-d',
            default='data',
            help='数据目录 (默认: data)'
        )
        
        parser.add_argument(
            '--subject', '-s',
            choices=['DD', 'Evender', 'both'],
            default='DD',
            help='被试选择 (默认: DD)'
        )
        
        # 训练参数
        parser.add_argument(
            '--phase1-epochs','-e1',
            type=int,
            help='第一阶段训练轮数'
        )
        
        parser.add_argument(
            '--phase2-epochs','-e2',
            type=int,
            help='第二阶段训练轮数'
        )
        
        parser.add_argument(
            '--batch-size',
            type=int,
            help='批次大小'
        )
        
        parser.add_argument(
            '--learning-rate',
            type=float,
            help='学习率'
        )
        
        # 持续学习参数
        parser.add_argument(
            '--continual-learning',
            action='store_true',
            help='启用持续学习'
        )
        
        parser.add_argument(
            '--c-intsyn',
            type=float,
            help='智能突触参数c (默认: 1.0)'
        )
        
        parser.add_argument(
            '--ksi-intsyn',
            type=float,
            help='智能突触参数ksi (默认: 0.01)'
        )
        
        parser.add_argument(
            '--ewc-lambda',
            type=float,
            help='EWC正则化参数lambda (默认: 100.0)'
        )
        
        # 其他参数
        parser.add_argument(
            '--config',
            help='使用配置文件'
        )
        
        parser.add_argument(
            '--debug',
            action='store_true',
            help='启用调试模式'
        )
        
        parser.add_argument(
            '--detail',
            action='store_true',
            help='启用详细错误分析（会增加评估时间但提供更详细的结果）'
        )
    
    def _add_evaluation_args(self, parser: argparse.ArgumentParser):
        """添加评估相关参数"""
        parser.add_argument(
            '--model-path', '-m',
            required=True,
            help='模型文件路径'
        )
        
        parser.add_argument(
            '--tasks',
            nargs='+',
            default=['pro_saccade', 'anti_saccade', 'delay_pro', 'delay_anti'],
            help='要评估的任务列表'
        )
        
        parser.add_argument(
            '--data-dir', '-d',
            default='data',
            help='数据目录'
        )
        
        parser.add_argument(
            '--output-file',
            help='评估结果输出文件'
        )
        
        parser.add_argument(
            '--detail',
            action='store_true',
            help='启用详细错误分析（会增加评估时间但提供更详细的结果）'
        )
    
    def _add_config_args(self, parser: argparse.ArgumentParser):
        """添加配置管理相关参数"""
        config_subparsers = parser.add_subparsers(dest='config_command', help='配置管理子命令')
        
        # 列出配置
        list_parser = config_subparsers.add_parser('list', help='列出所有配置')
        
        # 创建配置
        create_parser = config_subparsers.add_parser('create', help='创建新配置')
        create_parser.add_argument('name', help='配置名称')
        create_parser.add_argument('--base-config', default='random_training', help='基础配置')
        create_parser.add_argument('--phase1-epochs', type=int, help='第一阶段训练轮数')
        create_parser.add_argument('--phase2-epochs', type=int, help='第二阶段训练轮数')
        create_parser.add_argument('--batch-size', type=int, help='批次大小')
        create_parser.add_argument('--learning-rate', type=float, help='学习率')
        
        # 查看配置
        view_parser = config_subparsers.add_parser('view', help='查看配置内容')
        view_parser.add_argument('name', help='配置名称')
    

    
    def parse_args(self, args: Optional[list] = None) -> Dict[str, Any]:
        """解析命令行参数"""
        if args is None:
            args = sys.argv[1:]
        
        parsed_args = self.parser.parse_args(args)
        
        # 转换为字典
        args_dict = vars(parsed_args)
        
        # 处理特殊情况
        if args_dict.get('command') == 'config' and args_dict.get('config_command') == 'list':
            self._list_configs()
            sys.exit(0)
        
        return args_dict
    
    def _list_configs(self):
        """列出所有配置"""
        configs = self.config_manager.list_configs()
        print("可用配置文件:")
        for config in configs:
            print(f"  - {config}")
    
    def get_experiment_config(self, args: Dict[str, Any]) -> ExperimentConfig:
        """根据命令行参数获取实验配置
        
        参数优先级（从高到低）：
        1. 命令行参数（最高优先级）
        2. 配置文件参数
        3. 默认配置参数（最低优先级）
        """
        mode = args.get('mode', 'random')
        config_name = args.get('config')
        
        # 1. 首先获取基础配置（配置文件或默认配置）
        if config_name:
            # 使用指定的配置文件
            try:
                config_dict = self.config_manager.load_config(config_name)
                self.logger.info(f"加载配置文件: {config_name}")
            except FileNotFoundError:
                self.logger.warning(f"配置文件 {config_name} 不存在，使用默认配置")
                config_dict = self.config_manager.get_default_config(f"{mode}_training")
        else:
            # 使用默认配置
            config_dict = self.config_manager.get_default_config(f"{mode}_training")
            self.logger.info(f"使用默认配置: {mode}_training")
        
        # 2. 应用命令行覆盖参数（最高优先级）
        if args.get('phase1_epochs'):
            config_dict['hyperparameters']['phase1_epochs'] = args['phase1_epochs']
            self.logger.info(f"命令行覆盖: phase1_epochs = {args['phase1_epochs']}")
        if args.get('phase2_epochs'):
            config_dict['hyperparameters']['phase2_epochs'] = args['phase2_epochs']
            self.logger.info(f"命令行覆盖: phase2_epochs = {args['phase2_epochs']}")
        if args.get('batch_size'):
            config_dict['hyperparameters']['batch_size'] = args['batch_size']
            self.logger.info(f"命令行覆盖: batch_size = {args['batch_size']}")
        if args.get('learning_rate'):
            config_dict['hyperparameters']['learning_rate'] = args['learning_rate']
            self.logger.info(f"命令行覆盖: learning_rate = {args['learning_rate']}")
        
        # 3. 处理持续学习参数
        if 'continual_learning' not in config_dict['hyperparameters']:
            config_dict['hyperparameters']['continual_learning'] = {
                'enabled': False,
                'c_intsyn': 1.0,
                'ksi_intsyn': 0.01,
                'ewc_lambda': 100.0
            }
        
        # 命令行持续学习参数覆盖
        if args.get('continual_learning'):
            config_dict['hyperparameters']['continual_learning']['enabled'] = True
            self.logger.info("命令行启用持续学习")
        
        if args.get('c_intsyn'):
            config_dict['hyperparameters']['continual_learning']['c_intsyn'] = args['c_intsyn']
            self.logger.info(f"命令行覆盖: c_intsyn = {args['c_intsyn']}")
        
        if args.get('ksi_intsyn'):
            config_dict['hyperparameters']['continual_learning']['ksi_intsyn'] = args['ksi_intsyn']
            self.logger.info(f"命令行覆盖: ksi_intsyn = {args['ksi_intsyn']}")
        
        if args.get('ewc_lambda'):
            config_dict['hyperparameters']['continual_learning']['ewc_lambda'] = args['ewc_lambda']
            self.logger.info(f"命令行覆盖: ewc_lambda = {args['ewc_lambda']}")
        
        # 处理详细分析参数
        if args.get('detail'):
            if 'evaluation_config' not in config_dict:
                config_dict['evaluation_config'] = {}
            config_dict['evaluation_config']['detailed_analysis'] = True
            self.logger.info("命令行启用详细错误分析")
        
        # 4. 设置实验名称
        experiment_name = args.get('experiment_name') or f"{mode}_experiment"
        config_dict['experiment_name'] = experiment_name
        
        return ExperimentConfig(config_dict)
    
    def run(self, args: Optional[list] = None):
        """运行命令行接口"""
        parsed_args = self.parse_args(args)
        command = parsed_args.get('command')
        
        if not command:
            self.parser.print_help()
            return
        
        try:
            if command == 'train':
                self._handle_train_command(parsed_args)
            elif command == 'evaluate':
                self._handle_evaluate_command(parsed_args)
            elif command == 'config':
                self._handle_config_command(parsed_args)

            else:
                self.logger.error(f"未知命令: {command}")
                sys.exit(1)
        
        except Exception as e:
            self.logger.error(f"执行命令时出错: {e}")
            if parsed_args.get('debug'):
                raise
            sys.exit(1)
    
    def _handle_train_command(self, args: Dict[str, Any]):
        """处理训练命令"""
        from train import Trainer
        
        mode = args.get('mode', 'random')
        experiment_config = self.get_experiment_config(args)
        
        # 创建训练器
        output_dir = args.get('output_dir', 'checkpoints')
        trainer = Trainer(experiment_config, output_dir)
        
        # 获取详细分析配置
        detailed_analysis = args.get('detail', False)
        
        # 执行训练
        if mode == 'random':
            results = trainer.train_random_mode(detailed_analysis=detailed_analysis)
        elif mode == 'psychometric':
            results = trainer.train_psychometric_mode(
                data_dir=args.get('data_dir', 'data'),
                subject=args.get('subject', 'DD'),
                detailed_analysis=detailed_analysis
            )
        else:
            raise ValueError(f"不支持的训练模式: {mode}")
        
        # 生成可视化结果
        if not args.get('no_visualization'):
            trainer._generate_visualizations(results)
    
    def _handle_evaluate_command(self, args: Dict[str, Any]):
        """处理评估命令"""
        from train import EngineeredTrainer
        from model import CustomSaccadeModel
        import torch
        
        model_path = args.get('model_path')
        tasks = args.get('tasks', ['pro_saccade', 'anti_saccade', 'delay_pro', 'delay_anti'])
        output_file = args.get('output_file')
        
        if not model_path:
            self.logger.error("必须指定模型文件路径 (--model-path)")
            return
        
        if not os.path.exists(model_path):
            self.logger.error(f"模型文件不存在: {model_path}")
            return
        
        try:
            # 加载模型
            self.logger.info(f"加载模型: {model_path}")
            model = CustomSaccadeModel()
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.eval()
            
            # 创建训练器实例用于评估
            trainer = EngineeredTrainer(None, 'checkpoints')
            trainer.model = model
            
            # 执行评估
            self.logger.info("开始模型评估...")
            batch_size = 32
            num_trials = 1000
            
            # 获取详细分析配置
            detailed_analysis = args.get('detail', False)
            
            # 使用统一的评估方法
            results = trainer.evaluate_tasks(tasks, batch_size, num_trials, detailed_analysis=detailed_analysis)
            
            # 打印结果
            print("\n=== 模型评估结果 ===")
            for task, performance in results.items():
                print(f"{task}: {performance:.4f}")
                self.logger.info(f"{task} 任务性能: {performance:.4f}")
            
            # 计算平均性能
            avg_performance = sum(results.values()) / len(results)
            results['average'] = avg_performance
            
            print(f"平均性能: {avg_performance:.4f}")
            self.logger.info(f"平均性能: {avg_performance:.4f}")
            
            # 保存结果
            if output_file:
                import json
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                self.logger.info(f"评估结果已保存到: {output_file}")
            
        except Exception as e:
            self.logger.error(f"评估过程中出错: {e}")
            if args.get('debug'):
                raise
    
    def _handle_config_command(self, args: Dict[str, Any]):
        """处理配置命令"""
        config_command = args.get('config_command')
        
        if config_command == 'create':
            name = args['name']
            base_config = args.get('base_config', 'random_training')
            
            # 创建配置
            config = self.config_manager.create_experiment_config(
                experiment_name=name,
                base_config=base_config,
                hyperparameters={
                    'phase1_epochs': args.get('phase1_epochs'),
                    'phase2_epochs': args.get('phase2_epochs'),
                    'batch_size': args.get('batch_size'),
                    'learning_rate': args.get('learning_rate')
                }
            )
            
            print(f"配置 {name} 创建成功")
        
        elif config_command == 'view':
            name = args['name']
            try:
                config_dict = self.config_manager.load_config(name)
                print(json.dumps(config_dict, indent=2, ensure_ascii=False))
            except FileNotFoundError:
                print(f"配置文件 {name} 不存在")
    



def main():
    """主函数"""
    cli = CLIInterface()
    cli.run()


if __name__ == "__main__":
    main()