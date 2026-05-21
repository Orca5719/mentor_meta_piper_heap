"""
Piper机器人独立Agent配置
不依赖Hydra，直接创建MENTORAgent用于实时训练。
参数与 cfgs/agent/mentor_mw.yaml 保持一致。
"""

from agents.mentor_mw import MENTORAgent


def create_piper_agent(obs_shape, action_shape, device):
    """创建用于Piper机械臂训练的MENTORAgent。

    Args:
        obs_shape: 观测空间形状，如 (9, 256, 256)
        action_shape: 动作空间形状，如 (4,)
        device: torch.device

    Returns:
        MENTORAgent 实例
    """
    return MENTORAgent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        device=device,
        # 优化器
        lr=1e-4,
        lr_actor_ratio=1,
        # 编码器
        encoder_type='scratch',
        resnet_fix=True,
        feature_dim=50,
        pretrained_factor=1.,
        oneXone_reg_scale=0.,
        oneXone_reg_ratio=0.5,
        # 网络
        hidden_dim=1024,
        # Critic
        critic_target_tau=0.01,
        # 休眠机制
        dormant_threshold=0.025,
        target_dormant_ratio=0.2,
        dormant_temp=10,
        target_lambda=0.5,
        lambda_temp=50,
        # 扰动
        perturb_interval=50000,
        min_perturb_factor=0.2,
        max_perturb_factor=0.95,
        perturb_rate=2,
        # 探索
        num_expl_steps=2000,
        stddev_type='awake',
        stddev_schedule='linear(1.0,0.1,500000)',
        stddev_clip=0.3,
        # IQL expectile
        expectile=0.9,
        # 日志
        use_tb= True,
        # 辅助损失
        aux_loss_scale_warmup=-1,
        aux_loss_scale_warmsteps=-1,
        aux_loss_scale=0.002,
        aux_loss_type="",
        # Task-oriented Perturbation
        tp_set_size=10,
        # MoE
        moe_gate_dim=256,
        moe_hidden_dim=256,
        num_experts=16,
        top_k=4,
        dropout=0.1,
        # 人工干预奖励加权
        intervention_bonus=2.0,
    )
