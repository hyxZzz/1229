import numpy as np
from sim_core.entities.base import Entity
from utils.geometry import get_distance, get_vector, normalize

class Missile(Entity):
    def __init__(self, uid, team, launcher, target):
        super().__init__(uid, team)
        self.launcher_uid = launcher.uid
        self.target = target 
        
        self.pos = launcher.pos.copy()
        launch_dir = normalize(launcher.vel)
        self.vel = launcher.vel + launch_dir * 50.0 
        
        # --- 性能参数微调 ---
        self.max_speed = 1500.0 
        self.thrust = 60000.0    # 保持大推力，保证启动快
        self.burn_time = 8.0     # [修改] 缩短动力时间，模拟燃料耗尽后的滑行
        self.time_alive = 0.0
        self.turn_rate_max = np.deg2rad(40.0) 
        self.fov = np.deg2rad(60.0) 
        
        self.N_pn = 4.0 
        self.state = "Boost" 

    def replan(self, enemies):
        best_target = None
        min_cost = float('inf')
        my_dir = normalize(self.vel)
        
        for enemy in enemies:
            if not enemy.is_active: continue
            dist = get_distance(self.pos, enemy.pos)
            if dist > 30000: continue
            
            vec_to_enemy = normalize(enemy.pos - self.pos)
            angle = np.arccos(np.clip(np.dot(my_dir, vec_to_enemy), -1, 1))
            if angle > self.fov: continue
            
            cost = dist + angle * 5000 
            if cost < min_cost:
                min_cost = cost
                best_target = enemy
                
        if best_target:
            self.target = best_target
            return True
        else:
            self.is_active = False
            return False

    def update(self, dt, enemies):
        if not self.is_active: return False, None
        self.time_alive += dt
        
        if not self.target.is_active:
            if not self.replan(enemies):
                self.is_active = False
                return False, None

        r_vec = self.target.pos - self.pos
        dist = np.linalg.norm(r_vec)
        v_rel = self.target.vel - self.vel
        omega_vec = np.cross(r_vec, v_rel) / (dist**2 + 1e-6)
        
        acc_cmd_vec = np.cross(omega_vec, self.vel)
        acc_cmd_dir = normalize(acc_cmd_vec)
        if np.linalg.norm(acc_cmd_vec) < 1e-3: acc_cmd_dir = np.zeros(3)

        effective_vc = max(100.0, -np.dot(v_rel, r_vec/(dist+1e-6))) 
        acc_magnitude = self.N_pn * effective_vc * np.linalg.norm(omega_vec)
        acc_guidance = acc_magnitude * acc_cmd_dir

        gravity_comp = np.array([0, 0, 9.81])
        max_acc = 40.0 * 9.81
        
        total_cmd_acc = acc_guidance + gravity_comp
        if np.linalg.norm(total_cmd_acc) > max_acc:
            total_cmd_acc = normalize(total_cmd_acc) * max_acc

        thrust_vec = np.zeros(3)
        # [逻辑] 只有在动力时间内才有推力
        if self.time_alive < self.burn_time:
            thrust_vec = normalize(self.vel) * (self.thrust / 200.0) 
            self.state = "Boost"
        else:
            self.state = "Coast" # 滑行段
            
        # [修改] 阻力系数微调
        v_mag = np.linalg.norm(self.vel)
        current_g = np.linalg.norm(total_cmd_acc) / 9.81
        
        # 基础阻力增大 (限制极速)
        cd0 = 0.001 
        # 诱导阻力增大 (加大机动惩罚)
        k_induced = 0.0001 
        
        cd_total = cd0 + k_induced * (current_g ** 2)
        
        # 阻力计算
        drag_acc = cd_total * (v_mag ** 2) * 0.01
        drag_vec = -normalize(self.vel) * drag_acc
        
        real_gravity = np.array([0, 0, -9.81])
        final_acc = total_cmd_acc + thrust_vec + drag_vec + real_gravity
        
        self.vel += final_acc * dt
        self.pos += self.vel * dt

        if dist < max(150.0, v_mag * dt) and self.time_alive > 0.5:
            self.is_active = False
            self.target.hp -= 100
            self.target.is_active = False
            return True, self.target.uid
            
        if self.pos[2] < 0 or v_mag < 100:
            self.is_active = False
            
        return False, None