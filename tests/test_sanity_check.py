import sys
import os
import torch
import yaml
import shutil
import numpy as np

# 确保能导入项目根目录的模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.red_rl.trainer import RedTrainer

def test_sanity_training():
    print("\n>>> [Test 3] 训练循环冒烟测试 (Sanity Check) <<<")
    
    # 1. 读取并修改配置为“极速模式”
    # 假设你的配置文件在项目根目录的 configs/train_config.yaml
    # 注意：这里使用相对路径，确保在项目根目录下运行或根据实际情况调整
    config_path = os.path.join(os.path.dirname(__file__), '../configs/train_config.yaml')
    
    if not os.path.exists(config_path):
        print(f"❌ 找不到配置文件: {config_path}")
        return

    with open(config_path, 'r', encoding='utf-8') as f:
        full_cfg = yaml.safe_load(f)
        
    cfg = full_cfg['train']
    # 修改参数以快速运行
    cfg['steps_per_epoch'] = 100  # 仅采集 100 步
    cfg['epochs'] = 1             # 仅跑 1 个 Epoch
    cfg['train_iters'] = 2        # 仅更新 2 次
    cfg['batch_size'] = 64        # 小 Batch
    cfg['log_dir'] = "./logs_debug"
    cfg['target_kl'] = 100.0
    
    print("配置已修改: steps=100, train_iters=2 (不影响原文件)")

    # 2. 初始化 Trainer
    try:
        trainer = RedTrainer(cfg)
        print("Trainer 初始化成功。")
    except Exception as e:
        print(f"❌ Trainer 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. 运行数据采集
    print("正在尝试采集数据 (Collect Rollouts)...")
    try:
        trainer.collect_rollouts()
        print("✅ 数据采集成功。")
    except Exception as e:
        print(f"❌ 数据采集失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. 运行网络更新
    print("正在尝试网络更新 (Update)...")
    try:
        loss_pi, loss_v, kl = trainer.update()
        print(f"✅ 更新完成: LossPi={loss_pi:.4f}, LossV={loss_v:.4f}, KL={kl:.4f}")
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("❌ 显存不足 (OOM)！请在 trainer.py 中减小 Batch Size 或使用梯度累积。")
        else:
            print(f"❌ 更新失败: {e}")
            import traceback
            traceback.print_exc()
        return
    except Exception as e:
        print(f"❌ 更新遇到未知错误: {e}")
        import traceback
        traceback.print_exc()
        return

    # 5. 检查梯度 (确保计算图没断)
    print("正在检查梯度流...")
    has_grad = False
    for name, param in trainer.policy.named_parameters():
        if param.grad is not None:
            # print(f"  Param {name} has grad norm: {param.grad.norm().item():.4f}")
            has_grad = True
            break
            
    if has_grad:
        print("✅ 梯度检查通过：计算图正常。")
    else:
        print("❌ 梯度检查失败：所有参数梯度为 None！请检查 Loss 计算部分。")
        
    # 清理垃圾文件
    if os.path.exists("./logs_debug"):
        try:
            shutil.rmtree("./logs_debug")
            print("已清理测试日志目录。")
        except:
            pass
    print("测试结束。")

if __name__ == "__main__":
    # 修正了这里的调用名称
    test_sanity_training()