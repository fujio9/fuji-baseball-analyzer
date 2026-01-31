"""投球フェーズ推定モジュール"""

import numpy as np
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass


@dataclass
class PitchingPhase:
    """投球フェーズ情報"""
    name: str
    start_frame: int
    end_frame: int


def detect_pitching_phases(
    landmarks_list: List[Optional[Dict[str, Dict[str, float]]]],
    elbow_angles: List[Optional[float]],
    wrist_velocities: List[Optional[float]]
) -> List[PitchingPhase]:
    """投球フェーズを推定する
    
    Args:
        landmarks_list: 各フレームのランドマーク辞書のリスト
        elbow_angles: 各フレームの右肘角度のリスト
        wrist_velocities: 各フレームの右手首速度のリスト
    
    Returns:
        投球フェーズのリスト
    """
    phases = []
    
    if len(landmarks_list) == 0:
        return phases
    
    # 右手首のY座標（高さ）を取得
    wrist_y_positions = []
    for landmarks in landmarks_list:
        if landmarks is None or "right_wrist" not in landmarks:
            wrist_y_positions.append(None)
        else:
            wrist_y_positions.append(landmarks["right_wrist"]["y"])
    
    # テイクバック: 手首が最も高い位置
    max_y_idx = None
    max_y = -1.0
    for i, y in enumerate(wrist_y_positions):
        if y is not None and y > max_y:
            max_y = y
            max_y_idx = i
    
    # リリース: 手首速度が最大
    max_vel_idx = None
    max_vel = -1.0
    for i, vel in enumerate(wrist_velocities):
        if vel is not None and vel > max_vel:
            max_vel = vel
            max_vel_idx = i
    
    # トップ: テイクバックとリリースの中間
    if max_y_idx is not None and max_vel_idx is not None:
        top_idx = (max_y_idx + max_vel_idx) // 2
    elif max_y_idx is not None:
        top_idx = max_y_idx
    elif max_vel_idx is not None:
        top_idx = max_vel_idx
    else:
        top_idx = len(landmarks_list) // 4
    
    # フェーズを定義
    total_frames = len(landmarks_list)
    
    # テイクバック: 開始～トップ
    if max_y_idx is not None:
        phases.append(PitchingPhase(
            name="テイクバック",
            start_frame=0,
            end_frame=min(top_idx, total_frames - 1)
        ))
    
    # トップ: トップ付近
    if top_idx is not None:
        phases.append(PitchingPhase(
            name="トップ",
            start_frame=max(0, top_idx - 5),
            end_frame=min(top_idx + 5, total_frames - 1)
        ))
    
    # リリース: リリース付近
    if max_vel_idx is not None:
        phases.append(PitchingPhase(
            name="リリース",
            start_frame=max(0, max_vel_idx - 3),
            end_frame=min(max_vel_idx + 3, total_frames - 1)
        ))
    
    # フォロースルー: リリース後
    if max_vel_idx is not None:
        phases.append(PitchingPhase(
            name="フォロースルー",
            start_frame=min(max_vel_idx + 3, total_frames - 1),
            end_frame=total_frames - 1
        ))
    
    return phases


def calculate_phase_summary(
    phases: List[PitchingPhase],
    angles_data: Dict[str, List[Optional[float]]],
    velocities: List[Optional[float]]
) -> Dict[str, Dict[str, float]]:
    """フェーズごとのサマリーを計算する
    
    Args:
        phases: 投球フェーズのリスト
        angles_data: 各種角度データ
        velocities: 速度データ
    
    Returns:
        フェーズごとのサマリーデータ
    """
    summary = {}
    
    for phase in phases:
        phase_data = {}
        
        # 各角度の最大値・最小値を計算
        for angle_name, angles in angles_data.items():
            phase_angles = [
                angles[i] for i in range(phase.start_frame, phase.end_frame + 1)
                if i < len(angles) and angles[i] is not None
            ]
            if phase_angles:
                phase_data[f"{angle_name}_max"] = max(phase_angles)
                phase_data[f"{angle_name}_min"] = min(phase_angles)
                phase_data[f"{angle_name}_avg"] = sum(phase_angles) / len(phase_angles)
        
        # 速度の最大値を計算
        phase_velocities = [
            velocities[i] for i in range(phase.start_frame, phase.end_frame + 1)
            if i < len(velocities) and velocities[i] is not None
        ]
        if phase_velocities:
            phase_data["max_velocity"] = max(phase_velocities)
            phase_data["avg_velocity"] = sum(phase_velocities) / len(phase_velocities)
        
        summary[phase.name] = phase_data
    
    return summary




