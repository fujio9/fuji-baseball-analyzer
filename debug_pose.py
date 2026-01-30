"""pose/mediapipe_pose.py の動作確認用デバッグスクリプト"""

import argparse
import sys
from typing import Optional, List
import cv2
import numpy as np
import matplotlib.pyplot as plt

from pose.mediapipe_pose import initialize_pose, process_frame
from utils.debug_draw import draw_landmarks_on_frame

# 骨格線のみ表示モード
SKELETON_ONLY = False


def draw_skeleton(
    frame: np.ndarray,
    landmarks: dict,
    visibility_threshold: float = 0.5
) -> np.ndarray:
    """骨格（点と線）を描画する
    
    Args:
        frame: 入力フレーム (BGR形式)
        landmarks: 正規化済みランドマーク辞書
        visibility_threshold: 可視性の閾値
    
    Returns:
        骨格が描画されたフレーム
    """
    return draw_landmarks_on_frame(frame, landmarks, visibility_threshold)


def _compute_right_elbow_angle(landmarks: dict) -> Optional[float]:
    """正規化ランドマーク辞書から右肘角度（肩-肘-手首）を計算する
    
    Args:
        landmarks: 正規化済みランドマーク辞書
    
    Returns:
        右肘角度（度）（計算できない場合はNone）
    """
    required_keys = ["right_shoulder", "right_elbow", "right_wrist"]
    if not all(key in landmarks for key in required_keys):
        return None

    rs = landmarks["right_shoulder"]
    re = landmarks["right_elbow"]
    rw = landmarks["right_wrist"]

    # visibility チェック（一旦無効化）
    # if (rs.get("visibility", 1.0) < 0.5 or
    #     re.get("visibility", 1.0) < 0.5 or
    #     rw.get("visibility", 1.0) < 0.5):
    #     return None

    shoulder = np.array([rs["x"], rs["y"], rs.get("z", 0.0)])
    elbow = np.array([re["x"], re["y"], re.get("z", 0.0)])
    wrist = np.array([rw["x"], rw["y"], rw.get("z", 0.0)])

    # 肩→肘, 肘→手首 ベクトル
    v1 = elbow - shoulder
    v2 = wrist - elbow

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0.0 or n2 == 0.0:
        return None

    v1 /= n1
    v2 /= n2

    dot = float(np.dot(v1, v2))
    dot = float(np.clip(dot, -1.0, 1.0))

    angle_rad = float(np.arccos(dot))
    angle_deg = float(np.degrees(angle_rad))
    return angle_deg


def process_video(video_path: str) -> None:
    """動画ファイルを処理する
    
    Args:
        video_path: 動画ファイルのパス
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"エラー: 動画ファイル '{video_path}' を開けませんでした", file=sys.stderr)
        sys.exit(1)
    
    pose = initialize_pose()
    angles: List[Optional[float]] = []
    
    print("動画を再生中... (qキーで終了)")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 姿勢推定
        landmarks_dict = process_frame(pose, frame)
        
        if landmarks_dict is not None:
            # 骨格を描画
            if SKELETON_ONLY:
                background = np.zeros_like(frame)
                frame_with_skeleton = draw_skeleton(background, landmarks_dict)
            else:
                frame_with_skeleton = draw_skeleton(frame, landmarks_dict)

            # 右肘角度を計算して表示
            elbow_angle = _compute_right_elbow_angle(landmarks_dict)
            angles.append(elbow_angle)
            print(f"elbow_angle: {elbow_angle}")
            if elbow_angle is not None:
                cv2.putText(
                    frame_with_skeleton,
                    f"Elbow: {elbow_angle:.1f}°",
                    (50, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2.0,
                    (0, 0, 255),
                    4,
                )
        else:
            frame_with_skeleton = frame
            angles.append(None)
            cv2.putText(
                frame_with_skeleton,
                "No pose detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )
        
        # 表示
        cv2.imshow("Pose Detection Debug", frame_with_skeleton)
        
        # qキーで終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # グラフを表示
    if angles:
        frame_numbers = list(range(len(angles)))
        valid_angles = [angle if angle is not None else 0.0 for angle in angles]
        
        plt.figure(figsize=(10, 6))
        plt.plot(frame_numbers, valid_angles)
        plt.xlabel("Frame")
        plt.ylabel("Angle (deg)")
        plt.title("Right Elbow Angle Over Time")
        plt.grid(True)
        plt.show()


def process_webcam(camera_index: int = 0) -> None:
    """Webカメラ入力を処理する
    
    Args:
        camera_index: カメラのインデックス（デフォルト: 0）
    """
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"エラー: カメラ {camera_index} を開けませんでした", file=sys.stderr)
        sys.exit(1)
    
    pose = initialize_pose()
    
    print("Webカメラから入力中... (qキーで終了)")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("エラー: フレームを読み込めませんでした", file=sys.stderr)
            break
        
        # 姿勢推定
        landmarks_dict = process_frame(pose, frame)
        
        if landmarks_dict is not None:
            # 骨格を描画
            if SKELETON_ONLY:
                background = np.zeros_like(frame)
                frame_with_skeleton = draw_skeleton(background, landmarks_dict)
            else:
                frame_with_skeleton = draw_skeleton(frame, landmarks_dict)

            # 右肘角度を計算して表示
            elbow_angle = _compute_right_elbow_angle(landmarks_dict)
            print(f"elbow_angle: {elbow_angle}")
            if elbow_angle is not None:
                cv2.putText(
                    frame_with_skeleton,
                    f"Elbow: {elbow_angle:.1f}°",
                    (50, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2.0,
                    (0, 0, 255),
                    4,
                )
        else:
            frame_with_skeleton = frame
            cv2.putText(
                frame_with_skeleton,
                "No pose detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )
        
        # 表示
        cv2.imshow("Pose Detection Debug", frame_with_skeleton)
        
        # qキーで終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    """メイン関数"""
    global SKELETON_ONLY
    
    parser = argparse.ArgumentParser(
        description="pose/mediapipe_pose.py の動作確認用デバッグスクリプト"
    )
    parser.add_argument(
        "--video",
        type=str,
        help="動画ファイルのパス（指定しない場合はWebカメラを使用）"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="カメラのインデックス（デフォルト: 0）"
    )
    parser.add_argument(
        "--skeleton-only",
        action="store_true",
        help="骨格線のみ表示モード"
    )
    
    args = parser.parse_args()
    
    # コマンドライン引数で skeleton_only を設定
    SKELETON_ONLY = args.skeleton_only
    
    if args.video:
        process_video(args.video)
    else:
        process_webcam(args.camera)


if __name__ == "__main__":
    main()



