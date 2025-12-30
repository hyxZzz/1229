import os
import time
import argparse
from agents.red_rl.trainer import RedTrainer
import yaml

def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--steps', type=int, default=2000, help="Steps per epoch")
    args = parser.parse_args()
    
    # 配置参数
    yaml_cfg = load_config("configs/train_config.yaml")
    config = yaml_cfg['train']
    
    # 允许命令行参数覆盖 YAML 参数
    if args.steps:
        config['steps_per_epoch'] = args.steps

    # 创建保存目录
    if not os.path.exists('./checkpoints'):
        os.makedirs('./checkpoints')
        
    trainer = RedTrainer(config)
    
    print(">>> Start Training...")
    start_time = time.time()
    
    for epoch in range(args.epochs):
        # 1. 采集数据
        trainer.collect_rollouts()
        
        # 2. 更新网络
        # [修改] 接收 4 个返回值
        loss_pi, loss_v, kl, entropy = trainer.update()
        
        # 3. 日志打印
        if epoch % 1 == 0: # 每个 epoch 都打印
            elapsed = time.time() - start_time
            print(f"Epoch {epoch} | Time {elapsed:.0f}s | "
                  f"LossPi {loss_pi:.4f} | LossV {loss_v:.4f} | KL {kl:.4f} | Entropy {entropy:.4f}")
            
        # 4. 定期保存
        if epoch % 10 == 0:
            trainer.save_model(f"./checkpoints/model_epoch_{epoch}.pt")

if __name__ == "__main__":
    main()