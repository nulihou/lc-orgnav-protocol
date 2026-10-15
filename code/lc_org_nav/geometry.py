
import math
from typing import Tuple

def ego_to_target_geometry(
    ego_x: float, ego_y: float, ego_yaw: float,
    tgt_x: float, tgt_y: float
) -> Tuple[float, float]:
    """Compute distance and relative bearing from ego to target.

    Args:
        ego_x, ego_y, ego_yaw: ego pose in world (yaw in radians)
        tgt_x, tgt_y: target position in world

    Returns:
        d: distance
        theta: bearing in ego frame, in radians, wrapped to (-pi, pi]
    """
    dx = tgt_x - ego_x
    dy = tgt_y - ego_y
    d = math.sqrt(dx * dx + dy * dy)
    global_angle = math.atan2(dy, dx)
    theta = global_angle - ego_yaw
    # wrap to (-pi, pi]
    while theta <= -math.pi:
        theta += 2 * math.pi
    while theta > math.pi:
        theta -= 2 * math.pi
    return d, theta
