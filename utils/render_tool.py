import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class RenderTool:
    def __init__(self, map_size=100000):
        self.history = [] # 存储每一帧的所有实体状态
        self.map_limit = map_size

    def record_frame(self, sim):
        """
        在每一帧仿真结束时调用，快照当前状态
        """
        frame_data = {
            'reds': [],
            'blues': [],
            'missiles': []
        }
        
        # 记录红方
        for p in sim.aircrafts:
            if p.is_active:
                entry = {'pos': p.pos.copy(), 'team': p.team, 'uid': p.uid}
                if p.team == 0:
                    frame_data['reds'].append(entry)
                else:
                    frame_data['blues'].append(entry)
                    
        # 记录导弹
        for m in sim.missiles:
            if m.is_active:
                frame_data['missiles'].append({
                    'pos': m.pos.copy(), 
                    'team': m.team,
                    'uid': m.uid
                })
                
        self.history.append(frame_data)

    def animate(self, save_path=None):
        """
        生成动画
        """
        print(f"Generating Animation for {len(self.history)} frames...")
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 设置轴范围
        limit = self.map_limit
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_zlim(0, 20000)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Altitude (m)")
        
        # 初始化绘图对象
        red_dots, = ax.plot([], [], [], 'r^', label='Red Team') # 红方飞机
        blue_dots, = ax.plot([], [], [], 'b^', label='Blue Team') # 蓝方飞机
        missile_dots, = ax.plot([], [], [], 'k.', markersize=2, label='Missiles') # 导弹
        
        title_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)

        def update(frame_idx):
            data = self.history[frame_idx]
            
            # 更新红方
            if data['reds']:
                r_pos = np.array([d['pos'] for d in data['reds']])
                red_dots.set_data(r_pos[:, 0], r_pos[:, 1])
                red_dots.set_3d_properties(r_pos[:, 2])
            else:
                red_dots.set_data([], [])
                red_dots.set_3d_properties([])

            # 更新蓝方
            if data['blues']:
                b_pos = np.array([d['pos'] for d in data['blues']])
                blue_dots.set_data(b_pos[:, 0], b_pos[:, 1])
                blue_dots.set_3d_properties(b_pos[:, 2])
            else:
                blue_dots.set_data([], [])
                blue_dots.set_3d_properties([])
                
            # 更新导弹
            if data['missiles']:
                m_pos = np.array([d['pos'] for d in data['missiles']])
                missile_dots.set_data(m_pos[:, 0], m_pos[:, 1])
                missile_dots.set_3d_properties(m_pos[:, 2])
            else:
                missile_dots.set_data([], [])
                missile_dots.set_3d_properties([])
                
            title_text.set_text(f"Time: {frame_idx * 0.1:.1f}s | Entities: {len(data['reds']) + len(data['blues'])}")
            return red_dots, blue_dots, missile_dots, title_text

        ani = animation.FuncAnimation(fig, update, frames=len(self.history), interval=50, blit=False)
        
        plt.legend()
        if save_path:
            # 需要安装 ffmpeg
            ani.save(save_path, writer='ffmpeg', fps=20)
            print(f"Animation saved to {save_path}")
        else:
            plt.show()