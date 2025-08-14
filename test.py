from analysis.taskset import analyze_saccade_taskspace

# 分析任务空间（自动包含delay epoch）
tsa, h_trans = analyze_saccade_taskspace(
    model_path="checkpoints/random_experiment_20250814_104210/model/model_final.pth",
    epochs=None,  # 使用自动检测，会包含delay epoch
    dim_reduction_type='PCA'
)