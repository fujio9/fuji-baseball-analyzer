"""ランドマーク描画の使用例"""

import cv2
import numpy as np
from pose.mediapipe_pose import initialize_pose, process_frame
from utils.debug_draw import (
    draw_landmarks_on_frame,
    draw_landmarks_simple,
    draw_landmarks_with_mediapipe,
    draw_specific_landmarks
)


def debug_draw_example(video_path: str) -> None:
    """動画ファイルからランドマークを描画する例
    
    Args:
        video_path: 動画ファイルのパス
    """
    cap = cv2.VideoCapture(video_path)
    pose = initialize_pose()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 姿勢推定
        landmarks_dict = process_frame(pose, frame)
        
        if landmarks_dict is not None:
            # 方法1: 骨格とランドマークを描画
            frame_with_landmarks = draw_landmarks_on_frame(frame, landmarks_dict)
            
            # 方法2: シンプルな描画（点のみ）
            # frame_with_landmarks = draw_landmarks_simple(frame, landmarks_dict)
            
            # 方法3: 特定のランドマークのみ描画
            # frame_with_landmarks = draw_specific_landmarks(
            #     frame,
            #     landmarks_dict,
            #     ["right_shoulder", "right_elbow", "right_wrist"]
            # )
        else:
            frame_with_landmarks = frame
        
        # 表示
        cv2.imshow("Debug: Pose Landmarks", frame_with_landmarks)
        
        # 'q' キーで終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


def debug_draw_single_frame(frame: np.ndarray) -> np.ndarray:
    """単一フレームにランドマークを描画する例
    
    Args:
        frame: 入力フレーム (BGR形式)
    
    Returns:
        ランドマークが描画されたフレーム
    """
    pose = initialize_pose()
    landmarks_dict = process_frame(pose, frame)
    
    if landmarks_dict is not None:
        return draw_landmarks_on_frame(frame, landmarks_dict)
    
    return frame


if __name__ == "__main__":
    # 使用例
    # debug_draw_example("path/to/video.mp4")
    pass






