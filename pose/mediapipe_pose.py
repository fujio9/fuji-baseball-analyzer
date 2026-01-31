"""MediaPipe を使った姿勢推定モジュール"""

import numpy as np
from typing import Dict, Optional, Any
import cv2
import mediapipe as mp


# MediaPipe Pose ランドマークインデックス定数
class PoseLandmark:
    """MediaPipe Pose ランドマークインデックス"""
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


def initialize_pose() -> mp.solutions.pose.Pose:
    """MediaPipe Pose オブジェクトを初期化する
    
    Returns:
        MediaPipe Pose オブジェクト
    """
    return mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )


def detect_pose(
    pose: mp.solutions.pose.Pose,
    frame: np.ndarray
) -> Optional[Any]:
    """フレームから姿勢を検出する
    
    Args:
        pose: MediaPipe Pose オブジェクト
        frame: 入力フレーム (BGR形式のnumpy配列)
    
    Returns:
        検出されたランドマーク（検出失敗時はNone）
    """
    # MediaPipe は RGB 形式を期待するため変換
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 姿勢検出
    results = pose.process(rgb_frame)
    
    if results.pose_landmarks:
        return results.pose_landmarks
    
    return None


def normalize_landmarks(
    landmarks: Any
) -> Dict[str, Dict[str, float]]:
    """ランドマークを正規化済み座標の辞書に変換する
    
    Args:
        landmarks: MediaPipe のランドマークリスト
    
    Returns:
        正規化済みランドマークの辞書
        {
            "landmark_name": {
                "x": float (0-1),
                "y": float (0-1),
                "z": float,
                "visibility": float (0-1)
            },
            ...
        }
    """
    landmark_dict: Dict[str, Dict[str, float]] = {}
    
    # ランドマーク名のマッピング
    landmark_names = {
        0: "nose",
        1: "left_eye_inner",
        2: "left_eye",
        3: "left_eye_outer",
        4: "right_eye_inner",
        5: "right_eye",
        6: "right_eye_outer",
        7: "left_ear",
        8: "right_ear",
        9: "mouth_left",
        10: "mouth_right",
        11: "left_shoulder",
        12: "right_shoulder",
        13: "left_elbow",
        14: "right_elbow",
        15: "left_wrist",
        16: "right_wrist",
        17: "left_pinky",
        18: "right_pinky",
        19: "left_index",
        20: "right_index",
        21: "left_thumb",
        22: "right_thumb",
        23: "left_hip",
        24: "right_hip",
        25: "left_knee",
        26: "right_knee",
        27: "left_ankle",
        28: "right_ankle",
        29: "left_heel",
        30: "right_heel",
        31: "left_foot_index",
        32: "right_foot_index",
    }
    
    for idx, landmark in enumerate(landmarks.landmark):
        name = landmark_names.get(idx, f"landmark_{idx}")
        landmark_dict[name] = {
            "x": landmark.x,  # 既に正規化済み (0-1)
            "y": landmark.y,  # 既に正規化済み (0-1)
            "z": landmark.z,
            "visibility": landmark.visibility
        }
    
    return landmark_dict


def process_frame(
    pose: mp.solutions.pose.Pose,
    frame: np.ndarray
) -> Optional[Dict[str, Dict[str, float]]]:
    """フレームを処理して正規化済みランドマークを返す
    
    Args:
        pose: MediaPipe Pose オブジェクト
        frame: 入力フレーム (BGR形式のnumpy配列)
    
    Returns:
        正規化済みランドマークの辞書（検出失敗時はNone）
    """
    landmarks = detect_pose(pose, frame)
    
    if landmarks is None:
        return None
    
    return normalize_landmarks(landmarks)


def calc_angle(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray
) -> float:
    """3点から角度を計算する
    
    Args:
        a: 第1点の座標 (x, y, z) または (x, y)
        b: 第2点の座標（角度の頂点）(x, y, z) または (x, y)
        c: 第3点の座標 (x, y, z) または (x, y)
    
    Returns:
        角度（度）
    """
    # aからbへのベクトル
    vector1 = b - a
    
    # bからcへのベクトル
    vector2 = c - b
    
    # ベクトルのノルムを計算
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    
    # ゼロ除算を防ぐ
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    
    # 正規化
    vector1_normalized = vector1 / norm1
    vector2_normalized = vector2 / norm2
    
    # 内積を計算
    dot_product = np.dot(vector1_normalized, vector2_normalized)
    
    # 数値誤差による範囲外の値をクランプ
    dot_product = np.clip(dot_product, -1.0, 1.0)
    
    # 角度を計算（ラジアンから度に変換）
    angle_rad = np.arccos(dot_product)
    angle_deg = np.degrees(angle_rad)
    
    return float(angle_deg)


def calculate_right_elbow_angle(
    landmarks: Any
) -> Optional[float]:
    """MediaPipeのランドマークから右肘角度を計算する
    
    Args:
        landmarks: MediaPipe のランドマークリスト
    
    Returns:
        右肘の角度（度）（計算できない場合はNone）
    """
    if landmarks is None or len(landmarks.landmark) < 17:
        return None
    
    # 右肩(12), 右肘(14), 右手首(16) を取得
    right_shoulder = landmarks.landmark[PoseLandmark.RIGHT_SHOULDER]
    right_elbow = landmarks.landmark[PoseLandmark.RIGHT_ELBOW]
    right_wrist = landmarks.landmark[PoseLandmark.RIGHT_WRIST]
    
    # 可視性チェック
    if (right_shoulder.visibility < 0.5 or 
        right_elbow.visibility < 0.5 or 
        right_wrist.visibility < 0.5):
        return None
    
    # 座標をnumpy配列に変換
    shoulder = np.array([right_shoulder.x, right_shoulder.y, right_shoulder.z])
    elbow = np.array([right_elbow.x, right_elbow.y, right_elbow.z])
    wrist = np.array([right_wrist.x, right_wrist.y, right_wrist.z])
    
    # 角度を計算
    elbow_angle = calc_angle(shoulder, elbow, wrist)
    
    return elbow_angle

