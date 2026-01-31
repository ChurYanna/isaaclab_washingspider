# source/washingspider/washingspider/robots/washingspider_cfg.py
from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

# 你的 USD 路径（本地文件）
_THIS_DIR = Path(__file__).resolve().parent
USD_PATH = (_THIS_DIR.parent / "assets" / "usd" / "WashingSpider.usd").as_posix()

# 12 个驱动 revolute：fr1 fr2 fr3 fl1 fl2 fl3 br1 br2 br3 bl1 bl2 bl3
DRIVE_JOINT_REGEX = r"^(f|b)(r|l)[123]$"

WASHINGSPIDER_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        # 如果你希望更稳定的接触/奖励，通常建议开接触传感器（性能会稍降）
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # base 初始位置：先抬高一点，避免一开始穿地
        pos=(0.0, 0.0, 0.4),
        rot=(1.0, 0.0, 0.0, 0.0),  # wxyz
        # 初始关节角：先全部 0，后面可按站立姿态改
        joint_pos={DRIVE_JOINT_REGEX: 0.0},
        joint_vel={DRIVE_JOINT_REGEX: 0.0},
    ),
    actuators={
        # 只对 12 个 revolute 做驱动；带 xyz 的球踝不会被 regex 命中，因此天然不驱动
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[DRIVE_JOINT_REGEX],
            stiffness=40.0,   # 先用保守值，后面按站立/步态再调
            damping=2.0,
        ),
    },
)