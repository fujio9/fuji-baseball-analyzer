"""投球フォーム用メトリクス計算モジュール"""

from typing import List, Optional, Dict, Any

import numpy as np


def _nanmax_safe(values: List[Optional[float]]) -> Optional[float]:
    """None を除外して最大値を返すユーティリティ"""
    valid = [v for v in values if v is not None]
    if not valid:
        return None
    return float(np.max(valid))


def _argmax_safe(values: List[Optional[float]]) -> Optional[int]:
    """None を除外して最大値のインデックスを返す（なければ None）"""
    best_idx: Optional[int] = None
    best_val: float = -1e9
    for i, v in enumerate(values):
        if v is None:
            continue
        if best_idx is None or v > best_val:
            best_val = float(v)
            best_idx = i
    return best_idx


def compute_pitching_metrics(
    landmarks_list: List[Optional[Dict[str, Dict[str, float]]]],
    elbow_angles: List[Optional[float]],
    all_angles: Dict[str, List[Optional[float]]],
    wrist_velocities: List[Optional[float]],
) -> Dict[str, Any]:
    """投球フォーム解析用の代表的なメトリクスを計算する

    Args:
        landmarks_list: 各フレームのランドマーク辞書のリスト
        elbow_angles: 各フレームの右肘角度
        all_angles: calculate_all_angles_from_landmarks() の結果
        wrist_velocities: 各フレームの手首速度

    Returns:
        メトリクス辞書
    """

    metrics: Dict[str, Any] = {
        "max_elbow_angle": None,
        "release_elbow_angle": None,
        "torso_angle_at_release": None,
        "shoulder_angle_at_release": None,
        "hip_angle_at_release": None,
        "max_elbow_frame": None,
        "release_frame": None,
    }

    # 最大肘角度とそのフレーム
    max_elbow = _nanmax_safe(elbow_angles)
    metrics["max_elbow_angle"] = max_elbow
    if max_elbow is not None:
        # 最初に最大値をとるフレーム
        for i, v in enumerate(elbow_angles):
            if v is not None and abs(v - max_elbow) < 1e-6:
                metrics["max_elbow_frame"] = i
                break

    # リリースフレーム：手首速度が最大のフレーム
    release_frame = _argmax_safe(wrist_velocities)
    metrics["release_frame"] = release_frame

    if release_frame is not None:
        # リリース時肘角度
        if 0 <= release_frame < len(elbow_angles):
            metrics["release_elbow_angle"] = elbow_angles[release_frame]

        # 体幹傾き・肩ライン角度・骨盤ライン角度
        torso_list = all_angles.get("torso_axis", [])
        shoulder_line_list = all_angles.get("shoulder_line", [])
        hip_line_list = all_angles.get("hip_line", [])

        if 0 <= release_frame < len(torso_list):
            metrics["torso_angle_at_release"] = torso_list[release_frame]
        if 0 <= release_frame < len(shoulder_line_list):
            metrics["shoulder_angle_at_release"] = shoulder_line_list[release_frame]
        if 0 <= release_frame < len(hip_line_list):
            metrics["hip_angle_at_release"] = hip_line_list[release_frame]

    return metrics

{
  "cells": [],
  "metadata": {
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 2
}