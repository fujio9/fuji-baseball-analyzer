"""速度・加速度・角速度計算モジュール"""

import numpy as np
from typing import List, Optional, Dict


def calculate_velocity(
    positions: List[Optional[np.ndarray]],
    fps: float = 30.0
) -> List[Optional[float]]:
    """位置から速度を計算する
    
    Args:
        positions: 各フレームの位置座標のリスト
        fps: フレームレート（デフォルト: 30.0）
    
    Returns:
        各フレームの速度のリスト（m/s相当、正規化座標ベース）
    """
    if len(positions) < 2:
        return [None] * len(positions)
    
    velocities = [None]  # 最初のフレームは速度なし
    
    dt = 1.0 / fps
    
    for i in range(1, len(positions)):
        if positions[i] is None or positions[i-1] is None:
            velocities.append(None)
            continue
        
        # 位置の差分
        delta_pos = positions[i] - positions[i-1]
        
        # 速度（正規化座標での距離/時間）
        velocity = np.linalg.norm(delta_pos) / dt
        velocities.append(float(velocity))
    
    return velocities


def calculate_acceleration(
    velocities: List[Optional[float]],
    fps: float = 30.0
) -> List[Optional[float]]:
    """速度から加速度を計算する
    
    Args:
        velocities: 各フレームの速度のリスト
        fps: フレームレート（デフォルト: 30.0）
    
    Returns:
        各フレームの加速度のリスト
    """
    if len(velocities) < 2:
        return [None] * len(velocities)
    
    accelerations = [None]  # 最初のフレームは加速度なし
    
    dt = 1.0 / fps
    
    for i in range(1, len(velocities)):
        if velocities[i] is None or velocities[i-1] is None:
            accelerations.append(None)
            continue
        
        # 速度の差分
        delta_v = velocities[i] - velocities[i-1]
        
        # 加速度
        acceleration = delta_v / dt
        accelerations.append(float(acceleration))
    
    return accelerations


def calculate_angular_velocity(
    angles: List[Optional[float]],
    fps: float = 30.0
) -> List[Optional[float]]:
    """角度から角速度を計算する
    
    Args:
        angles: 各フレームの角度のリスト（度）
        fps: フレームレート（デフォルト: 30.0）
    
    Returns:
        各フレームの角速度のリスト（度/秒）
    """
    if len(angles) < 2:
        return [None] * len(angles)
    
    angular_velocities = [None]  # 最初のフレームは角速度なし
    
    dt = 1.0 / fps
    
    for i in range(1, len(angles)):
        if angles[i] is None or angles[i-1] is None:
            angular_velocities.append(None)
            continue
        
        # 角度の差分（-180から180の範囲に正規化）
        delta_angle = angles[i] - angles[i-1]
        if delta_angle > 180:
            delta_angle -= 360
        elif delta_angle < -180:
            delta_angle += 360
        
        # 角速度（度/秒）
        angular_velocity = delta_angle / dt
        angular_velocities.append(float(angular_velocity))
    
    return angular_velocities


def calculate_wrist_velocity(
    landmarks_list: List[Optional[Dict[str, Dict[str, float]]]],
    fps: float = 30.0
) -> List[Optional[float]]:
    """右手首の速度を計算する
    
    Args:
        landmarks_list: 各フレームのランドマーク辞書のリスト
        fps: フレームレート（デフォルト: 30.0）
    
    Returns:
        各フレームの右手首の速度のリスト
    """
    positions = []
    
    for landmarks in landmarks_list:
        if landmarks is None or "right_wrist" not in landmarks:
            positions.append(None)
            continue
        
        wrist = landmarks["right_wrist"]
        positions.append(np.array([wrist["x"], wrist["y"], wrist.get("z", 0.0)]))
    
    return calculate_velocity(positions, fps)




