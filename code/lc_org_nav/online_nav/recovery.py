"""
恢复策略模块

包含：
- StuckDetector: 卡死检测
- RecoveryPolicy: 恢复策略（倒车+重规划）
"""

import math
import carla
from typing import List, Optional, Deque, Tuple
from collections import deque
from dataclasses import dataclass
import time


@dataclass
class VehicleState:
    """车辆状态快照"""
    location: carla.Location
    speed: float  # m/s
    timestamp: float


class StuckDetector:
    """卡死检测器"""
    
    def __init__(
        self,
        time_window: float = 3.0,  # 时间窗口（秒）
        min_speed: float = 0.3,     # 最小速度阈值（m/s）
        min_displacement: float = 1.0,  # 最小位移阈值（米）
        check_interval: int = 10,   # 检查间隔（步数）
    ):
        """
        初始化卡死检测器
        
        Args:
            time_window: 检测时间窗口（秒）
            min_speed: 最小速度阈值（m/s）
            min_displacement: 最小位移阈值（米）
            check_interval: 检查间隔（步数）
        """
        self.time_window = time_window
        self.min_speed = min_speed
        self.min_displacement = min_displacement
        self.check_interval = check_interval
        
        # 状态历史
        self.state_history: Deque[VehicleState] = deque()
        self.step_count = 0
        self.is_stuck = False
    
    def reset(self):
        """重置检测器状态"""
        self.state_history.clear()
        self.step_count = 0
        self.is_stuck = False
    
    def update(
        self,
        vehicle: carla.Vehicle,
        current_time: Optional[float] = None
    ) -> bool:
        """
        更新状态并检测是否卡死
        
        Args:
            vehicle: 车辆对象
            current_time: 当前时间戳（如果为None则使用time.time()）
        
        Returns:
            是否卡死
        """
        if current_time is None:
            current_time = time.time()
        
        self.step_count += 1
        
        # 只在指定间隔检查
        if self.step_count % self.check_interval != 0:
            return self.is_stuck
        
        # 获取当前状态
        transform = vehicle.get_transform()
        velocity = vehicle.get_velocity()
        speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        
        state = VehicleState(
            location=transform.location,
            speed=speed,
            timestamp=current_time
        )
        
        # 添加到历史
        self.state_history.append(state)
        
        # 移除超出时间窗口的状态
        while (self.state_history and 
               current_time - self.state_history[0].timestamp > self.time_window):
            self.state_history.popleft()
        
        # 如果历史不足，不判定为卡死
        if len(self.state_history) < 3:
            self.is_stuck = False
            return False
        
        # 计算平均速度
        avg_speed = sum(s.speed for s in self.state_history) / len(self.state_history)
        
        # 计算总位移
        first_state = self.state_history[0]
        last_state = self.state_history[-1]
        displacement = math.sqrt(
            (last_state.location.x - first_state.location.x)**2 +
            (last_state.location.y - first_state.location.y)**2
        )
        
        # 判定卡死
        self.is_stuck = (
            avg_speed < self.min_speed and
            displacement < self.min_displacement
        )
        
        return self.is_stuck


class RecoveryPolicy:
    """恢复策略（倒车+重规划）"""
    
    def __init__(
        self,
        backup_distance: float = 2.5,  # 倒车距离（米）
        backup_speed: float = -2.0,    # 倒车速度（m/s，负值）
        max_recovery_attempts: int = 3,  # 每个阶段最大恢复次数
    ):
        """
        初始化恢复策略
        
        Args:
            backup_distance: 倒车距离（米）
            backup_speed: 倒车速度（m/s）
            max_recovery_attempts: 每个阶段最大恢复次数
        """
        self.backup_distance = backup_distance
        self.backup_speed = backup_speed
        self.max_recovery_attempts = max_recovery_attempts
        
        # 恢复状态
        self.recovery_attempts = {}  # {phase: count}
        self.is_recovering = False
        self.backup_start_location: Optional[carla.Location] = None
        self.backup_target_distance = 0.0
    
    def reset(self, phase: str):
        """重置指定阶段的恢复计数"""
        self.recovery_attempts[phase] = 0
        self.is_recovering = False
        self.backup_start_location = None
    
    def can_recover(self, phase: str) -> bool:
        """
        检查是否可以执行恢复
        
        Args:
            phase: 当前阶段（"G1", "G2", "G3"）
        
        Returns:
            是否可以恢复
        """
        count = self.recovery_attempts.get(phase, 0)
        return count < self.max_recovery_attempts
    
    def start_recovery(
        self,
        vehicle: carla.Vehicle,
        phase: str
    ):
        """
        开始恢复（倒车）
        
        Args:
            vehicle: 车辆对象
            phase: 当前阶段
        """
        if not self.can_recover(phase):
            return
        
        self.is_recovering = True
        self.backup_start_location = vehicle.get_transform().location
        self.backup_target_distance = self.backup_distance
        
        # 增加恢复计数
        self.recovery_attempts[phase] = self.recovery_attempts.get(phase, 0) + 1
    
    def update_recovery(
        self,
        vehicle: carla.Vehicle
    ) -> Tuple[carla.VehicleControl, bool]:
        """
        更新恢复状态（倒车控制）
        
        Args:
            vehicle: 车辆对象
        
        Returns:
            (control, is_recovery_done)
        """
        if not self.is_recovering:
            return carla.VehicleControl(), False
        
        current_location = vehicle.get_transform().location
        
        # 计算已倒车距离
        if self.backup_start_location:
            backup_dist = math.sqrt(
                (current_location.x - self.backup_start_location.x)**2 +
                (current_location.y - self.backup_start_location.y)**2
            )
        else:
            backup_dist = 0.0
        
        # 如果倒车距离足够，结束恢复
        if backup_dist >= self.backup_target_distance:
            self.is_recovering = False
            return carla.VehicleControl(throttle=0.0, brake=1.0), True
        
        # 倒车控制
        control = carla.VehicleControl(
            throttle=0.0,
            brake=0.0,
            steer=0.0,
            reverse=True  # 倒车
        )
        
        # 设置倒车速度（通过油门控制）
        # 注意：CARLA中reverse=True时，throttle控制倒车速度
        target_backup_speed = abs(self.backup_speed)
        current_velocity = vehicle.get_velocity()
        current_speed = math.sqrt(current_velocity.x**2 + current_velocity.y**2 + current_velocity.z**2)
        
        if current_speed < target_backup_speed:
            control.throttle = 0.3  # 倒车油门
        else:
            control.brake = 0.3
        
        return control, False
    
    def finish_recovery(self):
        """完成恢复"""
        self.is_recovering = False
        self.backup_start_location = None

