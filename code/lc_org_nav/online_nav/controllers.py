"""
低层规则控制器模块

包含：
- PurePursuitController: 横向控制（转向）
- SpeedPIDController: 纵向控制（速度）
- DockingController: 停靠控制
"""

import math
import carla
from typing import List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from lc_org_nav.online_nav.geometry_utils import get_rel_xy


@dataclass
class ControlOutput:
    """控制输出"""
    throttle: float  # [0, 1]
    brake: float     # [0, 1]
    steer: float     # [-1, 1]
    hand_brake: bool = False


class PurePursuitController:
    """Pure Pursuit横向控制器"""
    
    def __init__(
        self,
        lookahead_distance: float = 8.0,  # 优化：默认8m（G1阶段用10m，G2阶段用6m）
        wheelbase: float = 2.9,  # 车辆轴距（米），CARLA默认值
        max_steer_angle: float = 0.7,  # 最大转向角（弧度）
    ):
        """
        初始化Pure Pursuit控制器
        
        Args:
            lookahead_distance: 前视距离（米）
            wheelbase: 车辆轴距（米）
            max_steer_angle: 最大转向角（弧度）
        """
        self.lookahead_distance = lookahead_distance
        self.wheelbase = wheelbase
        self.max_steer_angle = max_steer_angle
        self.current_route_idx = 0
    
    def reset(self):
        """重置控制器状态"""
        self.current_route_idx = 0
    
    def _world_to_vehicle_coords(
        self,
        world_point: carla.Location,
        vehicle_transform: carla.Transform
    ) -> Tuple[float, float]:
        """
        将世界坐标转换为车辆坐标系
        
        Args:
            world_point: 世界坐标点
            vehicle_transform: 车辆变换
        
        Returns:
            (x_local, y_local) 车辆坐标系下的坐标
        """
        # 车辆位置和朝向
        vehicle_loc = vehicle_transform.location
        vehicle_yaw_rad = math.radians(vehicle_transform.rotation.yaw)
        
        # 计算相对位置
        dx = world_point.x - vehicle_loc.x
        dy = world_point.y - vehicle_loc.y
        
        # 旋转到车辆坐标系
        cos_yaw = math.cos(vehicle_yaw_rad)
        sin_yaw = math.sin(vehicle_yaw_rad)
        
        x_local = dx * cos_yaw + dy * sin_yaw
        y_local = -dx * sin_yaw + dy * cos_yaw
        
        return x_local, y_local
    
    def _find_lookahead_point(
        self,
        vehicle_transform: carla.Transform,
        route: List[carla.Waypoint]
    ) -> Optional[carla.Waypoint]:
        """
        找到前视距离处的路径点
        
        Args:
            vehicle_transform: 车辆当前变换
            route: 路径waypoint列表
        
        Returns:
            前视点waypoint，如果失败返回None
        """
        if not route:
            return None
        
        vehicle_loc = vehicle_transform.location
        
        # 从当前索引开始查找
        start_idx = max(0, self.current_route_idx - 5)  # 允许稍微回退
        
        accumulated_dist = 0.0
        prev_wp_loc = vehicle_loc
        
        for i in range(start_idx, len(route)):
            wp = route[i]
            wp_loc = wp.transform.location
            
            # 计算累积距离
            dist = math.sqrt(
                (wp_loc.x - prev_wp_loc.x)**2 +
                (wp_loc.y - prev_wp_loc.y)**2
            )
            accumulated_dist += dist
            
            if accumulated_dist >= self.lookahead_distance:
                self.current_route_idx = i
                return wp
            
            prev_wp_loc = wp_loc
        
        # 如果没找到，返回最后一个waypoint
        if route:
            self.current_route_idx = len(route) - 1
            return route[-1]
        
        return None
    
    def compute_steer(
        self,
        vehicle_transform: carla.Transform,
        route: List[carla.Waypoint]
    ) -> float:
        """
        计算转向角
        
        Args:
            vehicle_transform: 车辆当前变换
            route: 路径waypoint列表
        
        Returns:
            转向角 [-1, 1]
        """
        if not route:
            return 0.0
        
        # 找到前视点
        lookahead_wp = self._find_lookahead_point(vehicle_transform, route)
        if lookahead_wp is None:
            return 0.0
        
        # 转换为车辆坐标系
        x_local, y_local = self._world_to_vehicle_coords(
            lookahead_wp.transform.location,
            vehicle_transform
        )
        
        # 计算曲率
        # κ = 2 * y_local / Ld^2
        if abs(self.lookahead_distance) > 1e-6:
            curvature = 2.0 * y_local / (self.lookahead_distance ** 2)
        else:
            curvature = 0.0
        
        # 计算转向角
        # δ = arctan(κ * L)
        steer_angle = math.atan(curvature * self.wheelbase)
        
        # 归一化到 [-1, 1]
        steer = steer_angle / self.max_steer_angle
        steer = max(-1.0, min(1.0, steer))
        
        return steer


class SpeedPIDController:
    """速度PID控制器"""
    
    def __init__(
        self,
        kp: float = 0.8,  # 优化：从0.5提高到0.8（G1阶段更激进）
        ki: float = 0.0,  # 保持0，避免积分拖累
        kd: float = 0.15,  # 优化：从0.1提高到0.15
        dt: float = 0.05,  # 控制周期（秒）
        max_throttle: float = 0.8,
        max_brake: float = 1.0,
    ):
        """
        初始化PID控制器
        
        Args:
            kp: 比例系数
            ki: 积分系数
            kd: 微分系数
            dt: 控制周期（秒）
            max_throttle: 最大油门
            max_brake: 最大刹车
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.max_throttle = max_throttle
        self.max_brake = max_brake
        
        # PID状态
        self.integral = 0.0
        self.prev_error = 0.0
    
    def reset(self):
        """重置PID状态"""
        self.integral = 0.0
        self.prev_error = 0.0
    
    def compute_control(
        self,
        target_speed: float,
        current_speed: float
    ) -> Tuple[float, float]:
        """
        计算油门和刹车
        
        Args:
            target_speed: 目标速度（m/s）
            current_speed: 当前速度（m/s）
        
        Returns:
            (throttle, brake) 都在 [0, 1] 范围内
        """
        # 计算误差
        error = target_speed - current_speed
        
        # PID计算
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt
        
        u = self.kp * error + self.ki * self.integral + self.kd * derivative
        
        # 更新历史误差
        self.prev_error = error
        
        # 映射到油门/刹车
        if u > 0:
            throttle = min(self.max_throttle, u)
            brake = 0.0
        else:
            throttle = 0.0
            brake = min(self.max_brake, abs(u))
        
        return throttle, brake


class DockingController:
    """停靠控制器（几何控制）"""
    
    def __init__(
        self,
        # 注意：论文评测 NavSucc@4m 是 ego->目标车中心距离 < 4m，
        # 因此Dock目标点不能设置为“后方4m”，否则最理想也只能=4m而判失败。
        target_longitudinal: float = 3.0,  # 目标纵向位置（米，正值表示目标车后方）
        target_lateral: float = 0.0,       # 目标横向位置（米，0表示对齐到目标车中心线）
        k_longitudinal: float = 0.5,       # 纵向控制增益（基础值，会自适应调整）
        k_lateral: float = 0.5,             # 横向控制增益（初始值0.5）
        k_yaw: float = 0.3,                 # 朝向误差控制增益（新增）
        max_speed: float = 2.5,             # 最大速度（m/s，从2.0提高到2.5）
        success_radius: float = 0.5,       # 成功判定半径（米）
        success_speed_threshold: float = 0.3,  # 成功判定速度阈值（m/s）
        dt: float = 0.05,                  # 控制周期（秒）
        a_min: float = -4.0,               # 纵向加速度下限（m/s^2）
        a_max: float = 2.0,                # 纵向加速度上限（m/s^2）
        ki_speed: float = 0.15,            # 速度环积分（小积分补稳态）
        integral_limit: float = 3.0,       # 积分限幅
        max_throttle: float = 0.8,         # 油门限幅
        max_brake: float = 1.0,            # 刹车限幅
        # 横向：Dock阶段“追踪目标车后方虚拟对接点”的纯追踪参数（Step 3.2 关键增强）
        wheelbase: float = 2.9,
        max_steer_angle: float = 0.7,      # rad
        dock_lookahead: float = 4.0,       # m（Dock阶段更小的lookahead，减少大弧线）
        # === Step 3.2+（lane_misalignment专项增强）===
        # Dock 内对齐门控：先对齐（横向/航向），再闭合距离
        align_y_thresh: float = 0.5,       # m
        align_psi_thresh_deg: float = 10.0,  # deg
        align_speed: float = 1.0,          # m/s（未对齐时的最大“蠕行”速度）
        # Dock 横向：Stanley（比纯追踪更容易消掉低速横向残差）
        use_stanley: bool = True,
        k_stanley: float = 1.0,
        stanley_speed_epsilon: float = 0.5,  # m/s（低速防发散）
    ):
        """
        初始化停靠控制器
        
        坐标系定义：
        - x_rel > 0: ego在目标车后方
        - x_rel < 0: ego在目标车前方
        - y_rel > 0: ego在目标车右侧
        
        Args:
            target_longitudinal: 目标纵向位置（米，目标车后方4m）
            target_lateral: 目标横向位置（米，右侧0.5m）
            k_longitudinal: 纵向控制增益
            k_lateral: 横向控制增益
            max_speed: 最大速度（m/s）
            success_radius: 成功判定半径（米）
            success_speed_threshold: 成功判定速度阈值（m/s）
        """
        self.target_longitudinal = target_longitudinal  # 期望x_rel = 4.0
        self.target_lateral = target_lateral  # 期望y_rel = 0.5
        self.k_longitudinal_base = k_longitudinal  # 基础增益（用于自适应）
        self.k_lateral = k_lateral
        self.k_yaw = k_yaw  # 朝向误差控制增益
        self.max_speed = max_speed
        self.success_radius = success_radius
        self.success_speed_threshold = success_speed_threshold

        # 控制周期与纵向限幅
        self.dt = dt
        self.a_min = a_min
        self.a_max = a_max
        self.ki_speed = ki_speed
        self.integral_limit = integral_limit
        self.max_throttle = max_throttle
        self.max_brake = max_brake
        self.wheelbase = wheelbase
        self.max_steer_angle = max_steer_angle
        self.dock_lookahead = dock_lookahead

        # 对齐门控与Stanley
        self.align_y_thresh = float(align_y_thresh)
        self.align_psi_thresh_deg = float(align_psi_thresh_deg)
        self.align_speed = float(align_speed)
        self.use_stanley = bool(use_stanley)
        self.k_stanley = float(k_stanley)
        self.stanley_speed_epsilon = float(stanley_speed_epsilon)

        # 纵向积分状态（anti-windup）
        self._v_int = 0.0

    def reset(self) -> None:
        """重置Dock阶段内部状态（建议在进入DOCK时调用）"""
        self._v_int = 0.0

    def _world_to_vehicle_coords(
        self,
        world_point: carla.Location,
        vehicle_transform: carla.Transform
    ) -> Tuple[float, float]:
        """将世界坐标转换到ego车辆坐标系（x前,y右）"""
        vehicle_loc = vehicle_transform.location
        yaw = math.radians(vehicle_transform.rotation.yaw)
        dx = world_point.x - vehicle_loc.x
        dy = world_point.y - vehicle_loc.y
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        x_local = dx * cos_yaw + dy * sin_yaw
        y_local = -dx * sin_yaw + dy * cos_yaw
        return x_local, y_local

    @staticmethod
    def _wrap_pi(a: float) -> float:
        """Wrap angle (rad) to [-pi, pi]."""
        while a > math.pi:
            a -= 2.0 * math.pi
        while a < -math.pi:
            a += 2.0 * math.pi
        return a

    def _compute_alignment_errors(
        self,
        ego_transform: carla.Transform,
        ref_point: carla.Location,
        ref_yaw_deg: float
    ) -> Tuple[float, float, float]:
        """
        计算 Dock 对齐误差（以ref_yaw为参考方向）：
        - e_y: 横向偏差（m，右侧为正）
        - e_psi: 航向误差（rad，定义为 ref_yaw - ego_yaw，wrap到[-pi,pi]）
        - e_psi_deg: 航向误差（deg）
        """
        ego_loc = ego_transform.location
        ref_yaw = math.radians(float(ref_yaw_deg))
        ego_yaw = math.radians(float(ego_transform.rotation.yaw))

        # ego 相对 ref_point 的位移，投影到 ref frame（x前,y右）
        dx = ego_loc.x - ref_point.x
        dy = ego_loc.y - ref_point.y
        cos_y = math.cos(ref_yaw)
        sin_y = math.sin(ref_yaw)
        x_local = dx * cos_y + dy * sin_y
        y_local = -dx * sin_y + dy * cos_y

        e_y = float(y_local)
        e_psi = self._wrap_pi(ref_yaw - ego_yaw)
        e_psi_deg = math.degrees(e_psi)
        return e_y, e_psi, e_psi_deg

    def _compute_stanley_steer(
        self,
        ego_transform: carla.Transform,
        current_speed: float,
        e_y: float,
        e_psi: float
    ) -> float:
        """
        Stanley控制（简化版）：
          steer_angle(rad) = e_psi - atan(k * e_y / (v + eps))
        注意：这里 e_y 为“右侧为正”，因此需要一个负号让“右偏 → 左打”。
        返回CARLA steer ∈ [-1,1]。
        """
        v = max(float(current_speed), float(self.stanley_speed_epsilon))
        crosstrack_term = math.atan2(self.k_stanley * e_y, v)
        steer_angle = e_psi - crosstrack_term
        steer = steer_angle / max(1e-6, self.max_steer_angle)
        return max(-1.0, min(1.0, steer))

    def _compute_pure_pursuit_steer_to_point(
        self,
        ego_transform: carla.Transform,
        target_point: carla.Location
    ) -> float:
        """
        纯追踪：直接朝一个目标点打方向盘（用于Dock虚拟对接点）。
        返回CARLA steer ∈ [-1,1]。
        """
        x_local, y_local = self._world_to_vehicle_coords(target_point, ego_transform)
        ld = max(self.dock_lookahead, math.sqrt(x_local * x_local + y_local * y_local))
        if ld < 1e-3:
            return 0.0
        curvature = 2.0 * y_local / (ld * ld)
        steer_angle = math.atan(curvature * self.wheelbase)
        steer = steer_angle / max(1e-6, self.max_steer_angle)
        return max(-1.0, min(1.0, steer))
    
    def _world_to_target_frame(
        self,
        ego_transform: carla.Transform,
        target_transform: carla.Transform
    ) -> Tuple[float, float]:
        """
        将ego车辆位置转换到目标车辆的局部坐标系
        
        使用统一的get_rel_xy函数，确保与DOCK触发逻辑一致
        """
        return get_rel_xy(ego_transform, target_transform)
    
    def compute_control(
        self,
        ego_vehicle: carla.Vehicle,
        target_vehicle: carla.Vehicle,
        ref_point: Optional[carla.Location] = None,
        ref_yaw_deg: Optional[float] = None,
    ) -> Tuple[ControlOutput, float, float, dict]:
        """
        计算停靠控制
        
        Args:
            ego_vehicle: ego车辆
            target_vehicle: 目标车辆
            ref_point: Dock参考点（优先使用lane-snapped的docking waypoint位置）
            ref_yaw_deg: Dock参考航向（优先使用lane waypoint yaw，而不是目标车yaw）
        
        Returns:
            (control, x_rel, y_rel, diag)
        """
        ego_transform = ego_vehicle.get_transform()
        target_transform = target_vehicle.get_transform()
        
        # 计算相对位置（标准化坐标系）
        x_rel, y_rel = self._world_to_target_frame(ego_transform, target_transform)
        
        # 目标停靠点定义
        x_rel_des = self.target_longitudinal  # 期望x_rel = 4.0（目标车后方4m）
        y_rel_des = self.target_lateral       # 期望y_rel = 0.5（右侧0.5m）
        
        # 纵向误差（不要用abs！）
        # e_long > 0: ego在目标停靠点后方，需要前进
        # e_long < 0: ego在目标停靠点前方，需要减速/刹车
        raw_e_long = x_rel - x_rel_des
        
        # 限制有效误差范围，避免极端值
        e_long = max(-5.0, min(15.0, raw_e_long))
        
        # 自适应纵向增益：远快近慢
        # 距离越远，增益越大；越近越小
        if abs(e_long) > 6.0:
            k_long = 0.8  # 远处：快速接近
        elif abs(e_long) > 3.0:
            k_long = 0.5  # 中距离：中等速度
        else:
            k_long = 0.3  # 近处：慢速精细调整
        
        # 纵向速度指令：只允许前进（Dock阶段不倒车）
        v_cmd = k_long * e_long
        if v_cmd < 0.0:
            v_cmd = 0.0
        v_cmd = min(v_cmd, self.max_speed)
        
        # 横向误差（仍用于“刹停/成功判定”等，但转向主控改为追踪虚拟对接点）
        e_lat = y_rel - y_rel_des
        
        # 朝向误差（用于小幅修正，避免车身方向与目标车方向严重不一致）
        # 计算目标车和ego的朝向差（rad）
        yaw_t_deg = float(target_transform.rotation.yaw)
        yaw_e_deg = float(ego_transform.rotation.yaw)
        yaw_t = math.radians(yaw_t_deg)
        yaw_e = math.radians(yaw_e_deg)
        yaw_err = self._wrap_pi(yaw_t - yaw_e)
        
        # 获取当前速度（横/纵向都需要）
        vel = ego_vehicle.get_velocity()
        current_speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
        
        # === Step 3.2+: Dock 对齐误差（优先使用 lane waypoint yaw）===
        # 如果上层没提供ref_point/ref_yaw，则退化到“目标车yaw + 几何对接点”
        diag_ref_point = ref_point
        diag_ref_yaw_deg = float(ref_yaw_deg) if ref_yaw_deg is not None else None
        if diag_ref_point is None:
            back_dist = float(self.target_longitudinal)
            lat_offset = float(self.target_lateral)
            fx = math.cos(yaw_t)
            fy = math.sin(yaw_t)
            bx = -fx
            by = -fy
            rx = math.cos(yaw_t - math.pi / 2.0)
            ry = math.sin(yaw_t - math.pi / 2.0)
            ref_x = target_transform.location.x + back_dist * bx + lat_offset * rx
            ref_y = target_transform.location.y + back_dist * by + lat_offset * ry
            ref_z = target_transform.location.z
            diag_ref_point = carla.Location(ref_x, ref_y, ref_z)
        if diag_ref_yaw_deg is None:
            diag_ref_yaw_deg = yaw_t_deg

        e_y, e_psi, e_psi_deg = self._compute_alignment_errors(
            ego_transform=ego_transform,
            ref_point=diag_ref_point,
            ref_yaw_deg=float(diag_ref_yaw_deg),
        )

        # === Step B: Alignment Gating（未对齐时先对齐，再慢速闭合）===
        psi_thresh = float(self.align_psi_thresh_deg)
        y_thresh = float(self.align_y_thresh)
        need_align = (abs(e_y) > y_thresh) or (abs(e_psi_deg) > psi_thresh)
        dock_submode = "ALIGN" if need_align else "CLOSE"
        
        # 对齐门控：未对齐时限制“蠕行”速度，避免继续猛贴近导致4m内对不齐
        if need_align:
            v_cmd = min(v_cmd, max(0.0, self.align_speed))

        # 横向控制：优先Stanley（更容易在低速把横向残差压下去）
        if self.use_stanley:
            steer = self._compute_stanley_steer(
                ego_transform=ego_transform,
                current_speed=current_speed,
                e_y=e_y,
                e_psi=e_psi,
            )
        else:
            # 退化：纯追踪 ref_point + 小yaw修正（保留老路径，便于对比）
            steer_pp = self._compute_pure_pursuit_steer_to_point(ego_transform, diag_ref_point)
            steer_yaw = max(-1.0, min(1.0, (self.k_yaw * yaw_err) / 0.6))  # 0.6rad≈34deg归一化
            steer = max(-1.0, min(1.0, steer_pp + 0.35 * steer_yaw))

        # 纵向：速度环 + anti-windup + 互斥饱和
        v_err = v_cmd - current_speed
        kp_v = 0.8
        kd_v = 0.0  # 先保持简单，主要靠限幅+积分稳定

        # 先积分（后面可能因饱和回退）
        self._v_int += v_err * self.dt
        self._v_int = max(-self.integral_limit, min(self.integral_limit, self._v_int))

        a_raw = kp_v * v_err + self.ki_speed * self._v_int + kd_v * 0.0
        a_cmd = max(self.a_min, min(self.a_max, a_raw))

        # anti-windup：饱和且误差继续把控制推向饱和，则回退本次积分
        if a_cmd != a_raw:
            if (a_cmd >= self.a_max and v_err > 0.0) or (a_cmd <= self.a_min and v_err < 0.0):
                self._v_int -= v_err * self.dt
                self._v_int = max(-self.integral_limit, min(self.integral_limit, self._v_int))

        # accel->throttle/brake（互斥）
        if a_cmd >= 0.0:
            throttle = (a_cmd / max(1e-6, self.a_max)) * self.max_throttle
            throttle = max(0.0, min(self.max_throttle, throttle))
            brake = 0.0
        else:
            brake = (-a_cmd / max(1e-6, abs(self.a_min))) * self.max_brake
            brake = max(0.0, min(self.max_brake, brake))
            throttle = 0.0
        
        # 如果非常接近目标，强制刹停（避免抖动/过冲）
        if abs(e_long) < 0.3 and abs(e_lat) < 0.3:
            throttle = 0.0
            brake = 1.0
        
        control = ControlOutput(
            throttle=throttle,
            brake=brake,
            steer=steer,
            hand_brake=(abs(e_long) < 0.3 and abs(e_lat) < 0.3 and current_speed < 0.3)
        )
        
        diag = {
            "dock_submode": dock_submode,
            "e_y": float(e_y),
            "e_psi_deg": float(e_psi_deg),
            "ref_yaw_deg": float(diag_ref_yaw_deg),
        }

        return control, x_rel, y_rel, diag
    
    def is_success(
        self,
        x_rel: float,
        y_rel: float,
        current_speed: float
    ) -> bool:
        """
        判断是否成功停靠
        
        Args:
            x_rel: 纵向相对位置（目标车后方为正）
            y_rel: 横向相对位置（右侧为正）
            current_speed: 当前速度
        
        Returns:
            是否成功
        """
        # 纵向误差（不要用abs！）
        e_long = abs(x_rel - self.target_longitudinal)
        
        # 横向误差
        e_lat = abs(y_rel - self.target_lateral)
        
        # 检查速度
        speed_ok = current_speed < self.success_speed_threshold
        
        # 综合判定
        cond_long = e_long < self.success_radius  # 距离4m的误差 < 0.5m
        cond_lat = e_lat < self.success_radius    # 横向偏差 < 0.5m
        
        position_ok = cond_long and cond_lat
        
        return position_ok and speed_ok

