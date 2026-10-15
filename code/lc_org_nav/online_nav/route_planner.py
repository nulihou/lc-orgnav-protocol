"""
路径规划模块

功能：基于CARLA GlobalRoutePlanner生成车道中心线路径
"""

import carla
from typing import List, Optional
import sys
import os
import math


class RoutePlanner:
    """基于CARLA GlobalRoutePlanner的路径规划器"""
    
    def __init__(
        self,
        carla_map: carla.Map,
        sampling_resolution: float = 2.0,
        carla_python_api_path: Optional[str] = None,
        mode: str = "auto"
    ):
        """
        初始化路径规划器
        
        Args:
            carla_map: CARLA地图对象
            sampling_resolution: 路径采样分辨率（米）
            carla_python_api_path: CARLA PythonAPI路径（可选，会尝试自动查找）
        """
        self.carla_map = carla_map
        self.sampling_resolution = sampling_resolution
        self.carla_python_api_path = carla_python_api_path
        self.mode = mode
        self._grp = None
        self._initialized = False
    
    def _find_carla_python_api(self) -> Optional[str]:
        """尝试查找CARLA PythonAPI路径"""
        # 从环境变量获取
        carla_root = os.environ.get("CARLA_ROOT")
        if carla_root:
            python_api_path = os.path.join(carla_root, "PythonAPI")
            if os.path.exists(python_api_path):
                return python_api_path
        
        # 从PYTHONPATH获取
        for path in sys.path:
            if "PythonAPI" in path or "carla" in path.lower():
                if os.path.exists(path):
                    # 检查是否是PythonAPI目录
                    if os.path.basename(path) == "PythonAPI" or "PythonAPI" in path:
                        return path
        
        # 检查常见路径
        common_paths = [
            "E:/carla/WindowsNoEditor/PythonAPI",
            "C:/CARLA/PythonAPI",
            "D:/CARLA/PythonAPI",
        ]
        for path in common_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def _initialize_planner(self):
        """初始化GlobalRoutePlanner"""
        if self._initialized:
            return
        
        GlobalRoutePlanner = None
        GlobalRoutePlannerDAO = None
        
        # 尝试直接导入
        try:
            from agents.navigation.global_route_planner import GlobalRoutePlanner
            from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
        except ImportError:
            # 如果直接导入失败，尝试添加路径
            python_api_path = self.carla_python_api_path or self._find_carla_python_api()
            
            if python_api_path:
                # 确保PythonAPI目录在sys.path中（如果不在）
                if python_api_path not in sys.path:
                    sys.path.insert(0, python_api_path)
                
                # 检查 agents 目录是否存在（兼容CARLA不同发行版的目录结构）
                #
                # 常见两种结构：
                # 1) <CARLA_ROOT>/PythonAPI/agents/...
                # 2) <CARLA_ROOT>/PythonAPI/carla/agents/...
                agents_path = os.path.join(python_api_path, "agents")
                alt_agents_path = os.path.join(python_api_path, "carla", "agents")

                if os.path.exists(agents_path):
                    # 结构(1)：直接可用
                    pass
                elif os.path.exists(alt_agents_path):
                    # 结构(2)：需要把 PythonAPI/carla 加入 sys.path，才能 import agents.*
                    agents_path = alt_agents_path
                    carla_pkg_path = os.path.join(python_api_path, "carla")
                    if carla_pkg_path not in sys.path:
                        sys.path.insert(0, carla_pkg_path)
                else:
                    raise ImportError(
                        f"Could not find agents directory in {python_api_path}\n"
                        f"Expected one of:\n"
                        f"  - {os.path.join(python_api_path, 'agents')}\n"
                        f"  - {os.path.join(python_api_path, 'carla', 'agents')}\n"
                        f"Please check your CARLA PythonAPI installation / path configuration"
                    )
                
                # 再次尝试导入
                try:
                    from agents.navigation.global_route_planner import GlobalRoutePlanner
                    from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
                except ImportError as e:
                    # 检查具体的文件是否存在
                    grp_file = os.path.join(agents_path, "navigation", "global_route_planner.py")
                    dao_file = os.path.join(agents_path, "navigation", "global_route_planner_dao.py")
                    
                    error_msg = (
                        f"Could not import GlobalRoutePlanner: {e}\n"
                        f"PythonAPI path: {python_api_path}\n"
                        f"Agents path: {agents_path}\n"
                        f"GRP file exists: {os.path.exists(grp_file)}\n"
                        f"DAO file exists: {os.path.exists(dao_file)}\n"
                        f"sys.path entries: {[p for p in sys.path if 'PythonAPI' in p or 'carla' in p.lower()]}"
                    )
                    raise ImportError(error_msg)
            else:
                raise ImportError(
                    f"Could not import GlobalRoutePlanner and could not find CARLA PythonAPI path\n"
                    f"Please provide --carla-python-api-path or set CARLA_ROOT environment variable\n"
                    f"Expected structure: <CARLA_ROOT>/PythonAPI/agents/navigation/global_route_planner.py"
                )
        
        # 创建GlobalRoutePlanner
        if GlobalRoutePlanner is None or GlobalRoutePlannerDAO is None:
            raise RuntimeError("Failed to import GlobalRoutePlanner classes")
        
        try:
            dao = GlobalRoutePlannerDAO(self.carla_map, sampling_resolution=self.sampling_resolution)
            self._grp = GlobalRoutePlanner(dao)
            self._grp.setup()
            self._initialized = True
            print(f"[RoutePlanner] GlobalRoutePlanner initialized successfully (sampling_resolution={self.sampling_resolution}m)")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize GlobalRoutePlanner: {e}")
    
    def _plan_route_simple(
        self,
        start_location: carla.Location,
        goal_location: carla.Location
    ) -> List[carla.Waypoint]:
        """
        简化的路径规划（当GlobalRoutePlanner不可用时使用）
        直接使用起点和终点之间的waypoint序列
        """
        waypoints = []
        
        # 获取起点waypoint
        start_wp = self.get_waypoint_at_location(start_location)
        if start_wp is None:
            return []
        
        # 获取终点waypoint
        goal_wp = self.get_waypoint_at_location(goal_location)
        if goal_wp is None:
            return []
        
        # 如果起点和终点在同一车道，直接返回
        if start_wp.road_id == goal_wp.road_id and start_wp.lane_id == goal_wp.lane_id:
            # 沿着车道生成waypoint序列
            current_wp = start_wp
            waypoints.append(current_wp)
            
            # 计算距离
            dist = math.sqrt(
                (goal_location.x - start_location.x)**2 +
                (goal_location.y - start_location.y)**2
            )
            
            # 沿着车道前进
            step_size = 2.0
            traveled = 0.0
            max_steps = int(dist / step_size) + 10
            
            for _ in range(max_steps):
                if traveled >= dist:
                    break
                
                next_wps = current_wp.next(step_size)
                if not next_wps:
                    break
                
                current_wp = next_wps[0]
                waypoints.append(current_wp)
                traveled += step_size
        
        # 如果不在同一车道，返回起点和终点的waypoint
        if not waypoints:
            waypoints = [start_wp, goal_wp]
        
        return waypoints
    
    def plan_route_between(
        self,
        start_location: carla.Location,
        goal_location: carla.Location
    ) -> List[carla.Waypoint]:
        """
        规划从起点到终点的路径
        
        优先使用CARLA官方的GlobalRoutePlanner.trace_route()进行正式规划
        
        Args:
            start_location: 起点位置
            goal_location: 终点位置
        
        Returns:
            waypoint列表，表示路径
        """
        # 先尝试使用GlobalRoutePlanner
        if self.mode == "simple":
            return self._plan_route_simple(start_location, goal_location)

        if not self._initialized:
            try:
                self._initialize_planner()
            except (ImportError, RuntimeError) as e:
                if self.mode == "global":
                    raise
                # 如果GlobalRoutePlanner不可用，使用简化方法
                print(f"[RoutePlanner] GlobalRoutePlanner not available: {e}")
                print("[RoutePlanner] Falling back to simple route planning")
                return self._plan_route_simple(start_location, goal_location)
        
        if self._grp is not None:
            try:
                # 使用GlobalRoutePlanner.trace_route()进行正式规划
                # trace_route返回: List[Tuple[Waypoint, RoadOption]]
                route = self._grp.trace_route(start_location, goal_location)
                
                if not route:
                    print(f"[RoutePlanner] trace_route returned empty route, using simple planning")
                    return self._plan_route_simple(start_location, goal_location)
                
                # 提取waypoint列表（忽略RoadOption）
                waypoints = [waypoint for waypoint, road_option in route]
                
                if not waypoints:
                    print(f"[RoutePlanner] No waypoints extracted from route, using simple planning")
                    return self._plan_route_simple(start_location, goal_location)
                
                print(f"[RoutePlanner] Generated route with {len(waypoints)} waypoints using trace_route")
                return waypoints
                
            except Exception as e:
                print(f"[RoutePlanner] trace_route failed: {e}, using simple route planning")
                import traceback
                traceback.print_exc()
                return self._plan_route_simple(start_location, goal_location)
        else:
            # 使用简化方法
            print("[RoutePlanner] GlobalRoutePlanner not initialized, using simple route planning")
            return self._plan_route_simple(start_location, goal_location)
    
    def plan_route_from_waypoint(
        self,
        start_wp: carla.Waypoint,
        goal_wp: carla.Waypoint
    ) -> List[carla.Waypoint]:
        """
        从waypoint到waypoint规划路径
        
        Args:
            start_wp: 起点waypoint
            goal_wp: 终点waypoint
        
        Returns:
            waypoint列表，表示路径
        """
        return self.plan_route_between(
            start_wp.transform.location,
            goal_wp.transform.location
        )
    
    def get_waypoint_at_location(
        self,
        location: carla.Location
    ) -> Optional[carla.Waypoint]:
        """
        获取指定位置的waypoint
        
        Args:
            location: 位置
        
        Returns:
            waypoint，如果失败返回None
        """
        wp = self.carla_map.get_waypoint(
            location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
        
        if wp is None:
            # 尝试使用Any类型
            wp = self.carla_map.get_waypoint(
                location,
                project_to_road=True,
                lane_type=carla.LaneType.Any
            )
        
        return wp
