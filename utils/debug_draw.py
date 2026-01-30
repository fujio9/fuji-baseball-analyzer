"""ランドマーク描画用デバッグ表示モジュール"""

import numpy as np
import cv2
from typing import Dict, Optional, Tuple, List
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2


# MediaPipe Pose の接続情報（骨格を描画するための接続）
POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS


def draw_landmarks_on_frame(
    frame: np.ndarray,
    landmarks: Dict[str, Dict[str, float]],
    visibility_threshold: float = 0.5
) -> np.ndarray:
    """フレームにランドマークを描画する
    
    Args:
        frame: 入力フレーム (BGR形式)
        landmarks: 正規化済みランドマーク辞書
        visibility_threshold: 可視性の閾値（この値未満のランドマークは描画しない）
    
    Returns:
        ランドマークが描画されたフレーム
    """
    frame_copy = frame.copy()
    height, width = frame_copy.shape[:2]
    
    # ランドマーク名からインデックスへのマッピング
    landmark_name_to_index = {
        "nose": 0,
        "left_eye_inner": 1,
        "left_eye": 2,
        "left_eye_outer": 3,
        "right_eye_inner": 4,
        "right_eye": 5,
        "right_eye_outer": 6,
        "left_ear": 7,
        "right_ear": 8,
        "mouth_left": 9,
        "mouth_right": 10,
        "left_shoulder": 11,
        "right_shoulder": 12,
        "left_elbow": 13,
        "right_elbow": 14,
        "left_wrist": 15,
        "right_wrist": 16,
        "left_pinky": 17,
        "right_pinky": 18,
        "left_index": 19,
        "right_index": 20,
        "left_thumb": 21,
        "right_thumb": 22,
        "left_hip": 23,
        "right_hip": 24,
        "left_knee": 25,
        "right_knee": 26,
        "left_ankle": 27,
        "right_ankle": 28,
        "left_heel": 29,
        "right_heel": 30,
        "left_foot_index": 31,
        "right_foot_index": 32,
    }
    
    # インデックスから座標へのマッピングを作成
    index_to_coord: Dict[int, Tuple[int, int]] = {}
    
    for name, landmark_data in landmarks.items():
        if name not in landmark_name_to_index:
            continue
        
        visibility = landmark_data.get("visibility", 1.0)
        if visibility < visibility_threshold:
            continue
        
        index = landmark_name_to_index[name]
        x = int(landmark_data["x"] * width)
        y = int(landmark_data["y"] * height)
        index_to_coord[index] = (x, y)
    
    # 骨格を描画（太めに描画して視認性を向上）
    for connection in POSE_CONNECTIONS:
        start_idx, end_idx = connection
        if start_idx in index_to_coord and end_idx in index_to_coord:
            start_point = index_to_coord[start_idx]
            end_point = index_to_coord[end_idx]
            cv2.line(frame_copy, start_point, end_point, (0, 255, 0), 4)

    # 体幹軸・肩ライン・骨盤ラインを追加描画
    left_shoulder_idx = 11
    right_shoulder_idx = 12
    left_hip_idx = 23
    right_hip_idx = 24

    # 肩センターと腰センターを計算
    if (
        left_shoulder_idx in index_to_coord
        and right_shoulder_idx in index_to_coord
        and left_hip_idx in index_to_coord
        and right_hip_idx in index_to_coord
    ):
        ls = index_to_coord[left_shoulder_idx]
        rs = index_to_coord[right_shoulder_idx]
        lh = index_to_coord[left_hip_idx]
        rh = index_to_coord[right_hip_idx]

        shoulder_center = (
            int((ls[0] + rs[0]) / 2),
            int((ls[1] + rs[1]) / 2),
        )
        hip_center = (
            int((lh[0] + rh[0]) / 2),
            int((lh[1] + rh[1]) / 2),
        )

        # 体幹軸（黄色）
        cv2.line(frame_copy, shoulder_center, hip_center, (0, 255, 255), 4)

        # 肩ライン（青）
        cv2.line(frame_copy, ls, rs, (255, 0, 0), 4)

        # 骨盤ライン（緑・既存より太く）
        cv2.line(frame_copy, lh, rh, (0, 200, 0), 4)
    
    # ランドマーク点を描画
    for index, coord in index_to_coord.items():
        cv2.circle(frame_copy, coord, 5, (0, 0, 255), -1)
        # インデックス番号を表示（オプション）
        cv2.putText(
            frame_copy,
            str(index),
            (coord[0] + 5, coord[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (255, 255, 255),
            1
        )
    
    return frame_copy


def draw_landmarks_simple(
    frame: np.ndarray,
    landmarks: Dict[str, Dict[str, float]],
    visibility_threshold: float = 0.5
) -> np.ndarray:
    """シンプルなランドマーク描画（点のみ）
    
    Args:
        frame: 入力フレーム (BGR形式)
        landmarks: 正規化済みランドマーク辞書
        visibility_threshold: 可視性の閾値
    
    Returns:
        ランドマークが描画されたフレーム
    """
    frame_copy = frame.copy()
    height, width = frame_copy.shape[:2]
    
    for name, landmark_data in landmarks.items():
        visibility = landmark_data.get("visibility", 1.0)
        if visibility < visibility_threshold:
            continue
        
        x = int(landmark_data["x"] * width)
        y = int(landmark_data["y"] * height)
        
        # 重要な関節を大きく描画
        if name in ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                    "left_wrist", "right_wrist", "left_hip", "right_hip",
                    "left_knee", "right_knee", "left_ankle", "right_ankle"]:
            cv2.circle(frame_copy, (x, y), 8, (0, 255, 0), -1)
        else:
            cv2.circle(frame_copy, (x, y), 4, (0, 0, 255), -1)
        
        # ランドマーク名を表示
        cv2.putText(
            frame_copy,
            name.replace("_", " "),
            (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1
        )
    
    return frame_copy


def draw_landmarks_with_mediapipe(
    frame: np.ndarray,
    landmarks: landmark_pb2.NormalizedLandmarkList
) -> np.ndarray:
    """MediaPipe の drawing_utils を使用してランドマークを描画する
    
    Args:
        frame: 入力フレーム (BGR形式)
        landmarks: MediaPipe のランドマークリスト
    
    Returns:
        ランドマークが描画されたフレーム
    """
    frame_copy = frame.copy()
    frame_rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
    
    # MediaPipe の drawing_utils を使用（線を太くして視認性を向上）
    mp.solutions.drawing_utils.draw_landmarks(
        frame_rgb,
        landmarks,
        mp.solutions.pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
            color=(0, 0, 255),
            thickness=4,
            circle_radius=2
        ),
        connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
            color=(0, 255, 0),
            thickness=4
        )
    )
    
    # RGB から BGR に戻す
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    return frame_bgr


def draw_specific_landmarks(
    frame: np.ndarray,
    landmarks: Dict[str, Dict[str, float]],
    landmark_names: List[str],
    color: Tuple[int, int, int] = (0, 255, 0),
    radius: int = 5
) -> np.ndarray:
    """指定されたランドマークのみを描画する
    
    Args:
        frame: 入力フレーム (BGR形式)
        landmarks: 正規化済みランドマーク辞書
        landmark_names: 描画するランドマーク名のリスト
        color: 描画色 (B, G, R)
        radius: 円の半径
    
    Returns:
        ランドマークが描画されたフレーム
    """
    frame_copy = frame.copy()
    height, width = frame_copy.shape[:2]
    
    for name in landmark_names:
        if name not in landmarks:
            continue
        
        landmark_data = landmarks[name]
        visibility = landmark_data.get("visibility", 1.0)
        if visibility < 0.5:
            continue
        
        x = int(landmark_data["x"] * width)
        y = int(landmark_data["y"] * height)
        
        cv2.circle(frame_copy, (x, y), radius, color, -1)
        cv2.putText(
            frame_copy,
            name,
            (x + radius + 2, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )
    
    return frame_copy


def draw_trail_skeleton(
    frame: np.ndarray,
    history_landmarks: List[Optional[Dict[str, Dict[str, float]]]],
    max_history: int = 5,
    decay_base: float = 0.92,
    visibility_threshold: float = 0.5
) -> np.ndarray:
    """過去フレーム分の骨格を残像として描画する
    
    Args:
        frame: 入力フレーム (BGR形式)
        history_landmarks: 過去フレームのランドマーク辞書のリスト（新しい順）
        max_history: 描画する最大履歴フレーム数
        visibility_threshold: 可視性の閾値
    
    Returns:
        残像骨格が描画されたフレーム
    """
    frame_copy = frame.copy()
    
    # 最大履歴数までに制限
    history_to_draw = history_landmarks[:max_history]
    
    # 古い順（逆順）に処理して、新しいフレームを上に描画
    for idx, landmarks in enumerate(reversed(history_to_draw)):
        if landmarks is None:
            continue
        
        # 透明度を指数減衰で計算（古いほど薄く）
        # 最新フレーム（idx=0）は alpha=1.0、古いほど decay_base**idx で減衰
        alpha = decay_base ** idx
        alpha = max(0.1, min(1.0, alpha))
        
        # 骨格を描画した一時フレームを作成
        skeleton_frame = draw_landmarks_on_frame(
            np.zeros_like(frame_copy),
            landmarks,
            visibility_threshold
        )
        
        # 半透明合成
        frame_copy = cv2.addWeighted(frame_copy, 1.0 - alpha, skeleton_frame, alpha, 0)
    
    return frame_copy



