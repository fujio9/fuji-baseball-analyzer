"""関節角度計算モジュール"""

import numpy as np
from typing import Optional, List, Dict


def calc_angle(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray
) -> float:
    """3点から角度を計算する（汎用関数）

    Args:
        a: 第1点の座標 (x, y, z) または (x, y)
        b: 第2点の座標（角度の頂点）(x, y, z) または (x, y)
        c: 第3点の座標 (x, y, z) または (x, y)

    Returns:
        角度（度）
    """
    vector1 = b - a
    vector2 = c - b

    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    vector1_normalized = vector1 / norm1
    vector2_normalized = vector2 / norm2

    dot_product = np.dot(vector1_normalized, vector2_normalized)
    dot_product = np.clip(dot_product, -1.0, 1.0)

    angle_rad = np.arccos(dot_product)
    angle_deg = np.degrees(angle_rad)

    return float(angle_deg)


def calculate_elbow_angle(
    shoulder: np.ndarray,
    elbow: np.ndarray,
    wrist: np.ndarray
) -> float:
    """3点から肘の角度を計算する"""
    return calc_angle(shoulder, elbow, wrist)


def calculate_shoulder_angle(
    elbow: np.ndarray,
    shoulder: np.ndarray,
    hip: np.ndarray
) -> float:
    """肩の角度を計算する（肘-肩-腰）"""
    return calc_angle(elbow, shoulder, hip)


def calculate_knee_angle(
    hip: np.ndarray,
    knee: np.ndarray,
    ankle: np.ndarray
) -> float:
    """膝の角度を計算する（腰-膝-足首）"""
    return calc_angle(hip, knee, ankle)


def calculate_hip_angle(
    shoulder: np.ndarray,
    hip: np.ndarray,
    knee: np.ndarray
) -> float:
    """腰の角度を計算する（肩-腰-膝）"""
    return calc_angle(shoulder, hip, knee)


def calculate_all_angles_from_landmarks(
    landmarks_list: List[Optional[Dict[str, Dict[str, float]]]]
) -> Dict[str, List[Optional[float]]]:
    """全フレームの各種角度を計算する

    Args:
        landmarks_list: 各フレームのランドマーク辞書のリスト

    Returns:
        各種角度のリストを含む辞書
        {
            "right_elbow": [...],
            "right_shoulder": [...],
            "right_knee": [...],
            "right_hip": [...],
            "torso_axis": [...],
            "shoulder_line": [...],
            "hip_line": [...],
        }
    """

    def extract_coord(
        landmarks: Optional[Dict[str, Dict[str, float]]],
        name: str
    ) -> Optional[np.ndarray]:
        if landmarks is None or name not in landmarks:
            return None
        lm = landmarks[name]
        return np.array([lm["x"], lm["y"], lm.get("z", 0.0)])

    def line_angle_vs_vertical(p1: np.ndarray, p2: np.ndarray) -> float:
        """垂直に対する線分の角度を計算する（度）"""
        v = p2 - p1
        v2d = np.array([v[0], v[1]])
        norm_v = np.linalg.norm(v2d)
        if norm_v == 0.0:
            return 0.0
        # 垂直ベクトル（画面座標系で下向きを正とする）
        vertical = np.array([0.0, 1.0])
        dot = float(np.dot(v2d / norm_v, vertical))
        dot = float(np.clip(dot, -1.0, 1.0))
        angle_rad = float(np.arccos(dot))
        return float(np.degrees(angle_rad))

    def line_angle_vs_horizontal(p1: np.ndarray, p2: np.ndarray) -> float:
        """水平に対する線分の角度を計算する（度）"""
        v = p2 - p1
        v2d = np.array([v[0], v[1]])
        norm_v = np.linalg.norm(v2d)
        if norm_v == 0.0:
            return 0.0
        horizontal = np.array([1.0, 0.0])
        dot = float(np.dot(v2d / norm_v, horizontal))
        dot = float(np.clip(dot, -1.0, 1.0))
        angle_rad = float(np.arccos(dot))
        return float(np.degrees(angle_rad))

    results: Dict[str, List[Optional[float]]] = {
        "right_elbow": [],
        "right_shoulder": [],
        "right_knee": [],
        "right_hip": [],
        "torso_axis": [],
        "shoulder_line": [],
        "hip_line": [],
    }

    for landmarks in landmarks_list:
        right_shoulder = extract_coord(landmarks, "right_shoulder")
        left_shoulder = extract_coord(landmarks, "left_shoulder")
        right_elbow = extract_coord(landmarks, "right_elbow")
        right_wrist = extract_coord(landmarks, "right_wrist")
        right_hip = extract_coord(landmarks, "right_hip")
        left_hip = extract_coord(landmarks, "left_hip")
        right_knee = extract_coord(landmarks, "right_knee")
        right_ankle = extract_coord(landmarks, "right_ankle")

        # 右肘
        if right_shoulder is not None and right_elbow is not None and right_wrist is not None:
            results["right_elbow"].append(
                calculate_elbow_angle(right_shoulder, right_elbow, right_wrist)
            )
        else:
            results["right_elbow"].append(None)

        # 右肩
        if right_elbow is not None and right_shoulder is not None and right_hip is not None:
            results["right_shoulder"].append(
                calculate_shoulder_angle(right_elbow, right_shoulder, right_hip)
            )
        else:
            results["right_shoulder"].append(None)

        # 右膝
        if right_hip is not None and right_knee is not None and right_ankle is not None:
            results["right_knee"].append(
                calculate_knee_angle(right_hip, right_knee, right_ankle)
            )
        else:
            results["right_knee"].append(None)

        # 右腰
        if right_shoulder is not None and right_hip is not None and right_knee is not None:
            results["right_hip"].append(
                calculate_hip_angle(right_shoulder, right_hip, right_knee)
            )
        else:
            results["right_hip"].append(None)

        # 体幹軸（肩センター〜腰センター）
        if right_shoulder is not None and left_shoulder is not None and right_hip is not None and left_hip is not None:
            shoulder_center = (right_shoulder + left_shoulder) / 2.0
            hip_center = (right_hip + left_hip) / 2.0
            torso_angle = line_angle_vs_vertical(hip_center, shoulder_center)
            results["torso_axis"].append(torso_angle)
        else:
            results["torso_axis"].append(None)

        # 肩ライン角度（左右肩を結ぶ線の水平に対する角度）
        if right_shoulder is not None and left_shoulder is not None:
            shoulder_line_angle = line_angle_vs_horizontal(left_shoulder, right_shoulder)
            results["shoulder_line"].append(shoulder_line_angle)
        else:
            results["shoulder_line"].append(None)

        # 骨盤ライン角度（左右腰を結ぶ線の水平に対する角度）
        if right_hip is not None and left_hip is not None:
            hip_line_angle = line_angle_vs_horizontal(left_hip, right_hip)
            results["hip_line"].append(hip_line_angle)
        else:
            results["hip_line"].append(None)

    return results


