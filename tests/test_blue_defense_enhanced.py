import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sim_core.entities.aircraft import Aircraft
from sim_core.entities.missile import Missile
from agents.blue_rule.behavior_tree import BlueRuleAgent
from sim_core.maneuver_lib import ManeuverLibrary

def test_blue_rear_defense():
    print("\n>>> [Test 2] 蓝方防御逻辑测试 (后方偷袭场景) <<<")
    
    # 1. 场景：蓝机向东飞，红机在正后方 6km 处
    blue_jet = Aircraft(uid="Blue_Target", team=1, init_pos=[0, 0, 5000], init_vel=[250, 0, 0])
    red_jet = Aircraft(uid="Red_Hunter", team=0, init_pos=[-6000, 0, 5000], init_vel=[300, 0, 0])
    
    blue_brain = BlueRuleAgent(uid="Blue_Target", team=1)
    
    # 2. 红方发射导弹
    missile = Missile(uid="M_Kill", team=0, launcher=red_jet, target=blue_jet)
    all_missiles = [missile]
    
    print(f"态势: 导弹距离蓝机 {np.linalg.norm(blue_jet.pos - missile.pos):.1f} 米，位于正后方。")
    
    # 3. 蓝方决策
    action_id = blue_brain.get_action(blue_jet, all_missiles)
    
    print(f"蓝方决策 Action ID: {action_id}")
    
    # 4. 验证
    # 预期：应该检测到威胁，并执行急转 (Notch 或 Break)
    # 危险动作：ACTION_MAINTAIN (0), ACTION_CLIMB (3) 等无视威胁的动作
    evasive_actions = [
        ManeuverLibrary.ACTION_BREAK_LEFT, 
        ManeuverLibrary.ACTION_BREAK_RIGHT,
        ManeuverLibrary.ACTION_NOTCH_LEFT,
        ManeuverLibrary.ACTION_NOTCH_RIGHT
    ]
    
    if action_id in evasive_actions:
        print("✅ 测试通过：蓝方感知到后方威胁并执行了规避机动。")
    elif action_id == ManeuverLibrary.ACTION_MAINTAIN:
        print("❌ 测试失败：蓝方无视了后方导弹，继续平飞！")
        print("   -> 请检查 behavior_tree.py 中 CheckIncomingMissile 的逻辑，是否只检查了雷达前方？")
    else:
        print(f"⚠️ 警告：蓝方执行了非典型动作 {action_id}，需人工判断是否合理。")

if __name__ == "__main__":
    test_blue_rear_defense()