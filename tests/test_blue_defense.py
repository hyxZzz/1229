import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sim_core.entities.aircraft import Aircraft
from sim_core.entities.missile import Missile
from agents.blue_rule.behavior_tree import BlueRuleAgent
from sim_core.maneuver_lib import ManeuverLibrary

def test_blue_evasion():
    print(">>> 开始蓝方防御逻辑测试 (Evasion Test) <<<")
    
    # 1. 场景设置
    # 蓝机：在 (0,0) 向东飞
    blue_jet = Aircraft(uid="Blue_1", team=1, init_pos=[0, 0, 5000], init_vel=[250, 0, 0])
    blue_brain = BlueRuleAgent(uid="Blue_1", team=1)
    
    # 红机：在蓝机左后方 (模拟偷袭)
    red_jet = Aircraft(uid="Red_1", team=0, init_pos=[-5000, 5000, 5000], init_vel=[300, 0, 0])
    
    # 2. 发射导弹
    # 此时距离约 7km (sqrt(5000^2 + 5000^2))，小于阈值 15km，应该立刻触发规避
    missile = Missile(uid="M1", team=0, launcher=red_jet, target=blue_jet)
    all_missiles = [missile]
    
    print(f"Scenario: Missile launched at distance {np.linalg.norm(blue_jet.pos - red_jet.pos):.0f}m")
    
    # 3. 运行决策逻辑
    action_id = blue_brain.get_action(blue_jet, all_missiles)
    
    print(f"Blue Action Decision ID: {action_id}")
    
    # 4. 验证
    # 导弹在左后方，BlueRule logic 应该检测到威胁，并输出 Notch Left / Notch Right
    # 而不是 Action 0 (Maintain)
    
    if action_id in [ManeuverLibrary.ACTION_NOTCH_LEFT, ManeuverLibrary.ACTION_NOTCH_RIGHT]:
        print("✅ 测试通过：蓝方检测到导弹并执行了切向机动！")
    elif action_id == ManeuverLibrary.ACTION_MAINTAIN:
        print("❌ 测试失败：蓝方还在巡航，无视了导弹威胁！")
    else:
        print(f"⚠️ 警告：蓝方输出了意外的动作 ID: {action_id}")

    # --- 附加测试：远距离无威胁 ---
    print("\n[附加测试] 远距离情况...")
    far_missile = Missile(uid="M2", team=0, launcher=red_jet, target=blue_jet)
    far_missile.pos = np.array([-50000, 0, 5000]) # 50km 外
    action_id_far = blue_brain.get_action(blue_jet, [far_missile])
    
    if action_id_far == ManeuverLibrary.ACTION_MAINTAIN:
        print("✅ 远距离测试通过：蓝方保持巡航。")
    else:
        print(f"❌ 远距离测试失败：蓝方反应过度，动作: {action_id_far}")

if __name__ == "__main__":
    test_blue_evasion()
