"""Streamlit UI アプリケーション"""

import streamlit as st
import numpy as np
import cv2
from typing import Optional, Dict, List, Tuple, Any
import tempfile
import os
import io
import matplotlib.pyplot as plt
import mediapipe as mp

from pose.mediapipe_pose import initialize_pose, process_frame
from analysis.angles import (
    calculate_elbow_angle,
    calculate_all_angles_from_landmarks,
)
from analysis.velocity import calculate_wrist_velocity, calculate_angular_velocity
from analysis.phases import detect_pitching_phases, calculate_phase_summary
from analysis.metrics import compute_pitching_metrics
from analysis.evaluator import evaluate_pitching_form
from utils.debug_draw import draw_landmarks_on_frame, draw_trail_skeleton

# MediaPipe の描画ユーティリティ
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# ページ設定：ワイドレイアウト
st.set_page_config(layout="wide")

# CSS: コンテナ幅を100%に
st.markdown(
    """
    <style>
    .block-container {
        max-width: 100% !important;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


def _save_uploaded_file_to_temp(uploaded_file: Any) -> Optional[str]:
    """
    Cloud Run 対応: /tmp フォルダに一時ファイルを保存
    
    Args:
        uploaded_file: Streamlitのアップロードファイルオブジェクト
    
    Returns:
        一時ファイルのパス（失敗時はNone）
    """
    if uploaded_file is None:
        st.error("動画ファイルがアップロードされていません")
        return None

    try:
        # read() で一度だけバイト列取得
        if hasattr(uploaded_file, 'seek'):
            uploaded_file.seek(0)  # ファイルポインタを先頭に戻す
        file_bytes = uploaded_file.read()
        if not file_bytes:
            st.error("アップロードされた動画が空です")
            return None

        # /tmp フォルダに明示的に保存（Cloud Run 対応）
        tmp_dir = "/tmp"
        os.makedirs(tmp_dir, exist_ok=True)
        
        # ファイル拡張子を取得
        if hasattr(uploaded_file, 'name') and uploaded_file.name:
            file_ext = os.path.splitext(uploaded_file.name)[1] or ".mp4"
        else:
            file_ext = ".mp4"
        
        # 一時ファイルパスを生成
        tmp_file_path = tempfile.mktemp(suffix=file_ext, dir=tmp_dir)
        
        # ファイルに書き込み
        with open(tmp_file_path, "wb") as tmp_file:
            tmp_file.write(file_bytes)
        
        return tmp_file_path

    except Exception as e:
        st.error(f"一時ファイル生成中にエラー発生: {e}")
        return None


def _read_frames_from_video(
    video_path: str,
    max_frames: int = 1000,
    max_width: int = 1280,
    frame_skip: int = 1,
    progress_container: Any = None
) -> Optional[List[np.ndarray]]:
    """動画ファイルからフレームを読み込む（Cloud Run 対応：大きな動画でもタイムアウトしない）
    
    Args:
        video_path: 動画ファイルのパス
        max_frames: 最大フレーム数（メモリ節約）
        max_width: 最大幅（リサイズ）
        frame_skip: フレームスキップ数（1=全フレーム、2=1フレームおき）
        progress_container: 進行状況表示用のコンテナ
    
    Returns:
        フレームのリスト（失敗時はNone）
    """
    cap = cv2.VideoCapture(video_path)
    frames: List[np.ndarray] = []
    
    if not cap.isOpened():
        if progress_container:
            progress_container.error("動画ファイルを開けませんでした")
        return None
    
    try:
        # 動画情報を取得
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if progress_container:
            progress_container.info(f"動画情報: {width}x{height}, {total_frames}フレーム, {fps:.1f}fps")
        
        # リサイズが必要か判定
        resize_needed = width > max_width
        if resize_needed:
            scale = max_width / width
            new_width = max_width
            new_height = int(height * scale)
            if progress_container:
                progress_container.info(f"動画をリサイズ: {new_width}x{new_height}")
        
        # フレームスキップを調整（動画が長すぎる場合）
        if total_frames > max_frames * frame_skip:
            frame_skip = max(1, total_frames // max_frames)
            if progress_container:
                progress_container.info(f"フレームスキップ: {frame_skip}（メモリ節約のため）")
        
        frame_count = 0
        read_count = 0
        
        progress_bar = None
        if progress_container:
            progress_bar = progress_container.progress(0)
            status_text = progress_container.empty()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # フレームスキップ
            if frame_count % frame_skip != 0:
                frame_count += 1
                continue
            
            # リサイズ
            if resize_needed:
                frame = cv2.resize(frame, (new_width, new_height))
            
            frames.append(frame)
            read_count += 1
            
            # 進行状況更新
            if progress_bar and frame_count % 10 == 0:
                progress = min(1.0, frame_count / total_frames)
                progress_bar.progress(progress)
                if status_text:
                    status_text.text(f"読み込み中: {read_count}/{min(total_frames // frame_skip, max_frames)} フレーム")
            
            # 最大フレーム数に達したら終了
            if read_count >= max_frames:
                if progress_container:
                    progress_container.warning(f"最大フレーム数（{max_frames}）に達したため、読み込みを終了しました")
                break
            
            frame_count += 1
        
        if progress_bar:
            progress_bar.progress(1.0)
        if progress_container:
            progress_container.success(f"✅ {read_count} フレームを読み込みました")
        
        return frames if frames else None
    
    finally:
        cap.release()


def load_video_frames(
    uploaded_file: Any,
    progress_container: Any = None
) -> Optional[List[np.ndarray]]:
    """アップロードされた動画ファイルからフレームを読み込む（Cloud Run 対応）
    
    Args:
        uploaded_file: Streamlitのアップロードファイルオブジェクト
        progress_container: 進行状況表示用のコンテナ
    
    Returns:
        フレームのリスト（失敗時はNone）
    """
    if uploaded_file is None:
        return None
    
    tmp_path = _save_uploaded_file_to_temp(uploaded_file)
    if tmp_path is None:
        return None
    
    try:
        return _read_frames_from_video(tmp_path, progress_container=progress_container)
    finally:
        # 一時ファイルを削除（Cloud Run では /tmp は自動クリーンアップされるが、明示的に削除）
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except Exception:
            pass  # 削除失敗は無視


def process_video_frames(
    frames: List[np.ndarray],
    progress_container: Any = None
) -> List[Optional[Dict[str, Dict[str, float]]]]:
    """動画フレームを処理してランドマークを取得（Cloud Run 対応：進行状況表示）
    
    Args:
        frames: フレームのリスト
        progress_container: 進行状況表示用のコンテナ
    
    Returns:
        各フレームのランドマーク辞書のリスト
    """
    pose = initialize_pose()
    results = []
    total_frames = len(frames)
    
    progress_bar = None
    status_text = None
    if progress_container:
        progress_bar = progress_container.progress(0)
        status_text = progress_container.empty()
    
    for idx, frame in enumerate(frames):
        landmarks = process_frame(pose, frame)
        results.append(landmarks)
        
        # 進行状況更新（10フレームごと）
        if progress_bar and idx % 10 == 0:
            progress = (idx + 1) / total_frames
            progress_bar.progress(progress)
            if status_text:
                status_text.text(f"姿勢推定中: {idx + 1}/{total_frames} フレーム")
    
    if progress_bar:
        progress_bar.progress(1.0)
    if status_text:
        status_text.text(f"✅ {total_frames} フレームの姿勢推定が完了しました")
    
    return results


def _extract_landmark_coordinates(
    landmarks: Dict[str, Dict[str, float]],
    landmark_name: str
) -> Optional[np.ndarray]:
    """ランドマーク辞書から指定されたランドマークの座標を抽出する
    
    Args:
        landmarks: ランドマーク辞書
        landmark_name: ランドマーク名
    
    Returns:
        座標配列 (x, y, z)（存在しない場合はNone）
    """
    if landmark_name not in landmarks:
        return None
    
    landmark = landmarks[landmark_name]
    return np.array([landmark["x"], landmark["y"], landmark["z"]])


def _calculate_single_elbow_angle(
    landmarks: Optional[Dict[str, Dict[str, float]]]
) -> Optional[float]:
    """1フレームの右肘の角度を計算する
    
    Args:
        landmarks: ランドマーク辞書
    
    Returns:
        右肘の角度（度）（計算できない場合はNone）
    """
    if landmarks is None:
        return None
    
    shoulder = _extract_landmark_coordinates(landmarks, "right_shoulder")
    elbow = _extract_landmark_coordinates(landmarks, "right_elbow")
    wrist = _extract_landmark_coordinates(landmarks, "right_wrist")
    
    if shoulder is None or elbow is None or wrist is None:
        return None
    
    return calculate_elbow_angle(shoulder, elbow, wrist)


def calculate_elbow_angles_from_landmarks(
    landmarks_list: List[Optional[Dict[str, Dict[str, float]]]]
) -> List[Optional[float]]:
    """ランドマークから肘の角度を計算
    
    Args:
        landmarks_list: 各フレームのランドマーク辞書のリスト
    
    Returns:
        各フレームの肘の角度のリスト
    """
    return [_calculate_single_elbow_angle(landmarks) for landmarks in landmarks_list]


def _render_video_upload() -> Optional[Any]:
    """動画アップロードUIを表示する
    
    Returns:
        アップロードされたファイルオブジェクト
    """
    return st.file_uploader(
        "動画ファイルをアップロードしてください",
        type=["mp4", "avi", "mov", "mkv"]
    )


def _render_video_preview(uploaded_file: Any) -> None:
    """動画プレビューを表示する
    
    Args:
        uploaded_file: アップロードされたファイルオブジェクト
    """
    st.subheader("アップロードされた動画")
    st.video(uploaded_file)


def _render_frame_viewer(
    frames: List[np.ndarray],
    frame_idx: int,
    landmarks: Optional[Dict[str, Dict[str, float]]],
    elbow_angle: Optional[float],
    torso_angle: Optional[float] = None,
    shoulder_line_angle: Optional[float] = None,
    hip_line_angle: Optional[float] = None,
) -> None:
    """フレームビューアーを表示する
    
    Args:
        frames: フレームのリスト
        frame_idx: 表示するフレームのインデックス
        landmarks: ランドマーク辞書
        elbow_angle: 肘の角度
    """
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("現在のフレーム")
        frame_copy = frames[frame_idx].copy()
        if landmarks is not None:
            frame_with_skeleton = draw_landmarks_on_frame(frame_copy, landmarks)
        else:
            frame_with_skeleton = frame_copy
        frame_rgb = cv2.cvtColor(frame_with_skeleton, cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, use_container_width=True)
    
    with col2:
        st.write("解析情報")
        if landmarks is not None:
            st.success("姿勢検出: 成功")
            cols = st.columns(2)
            with cols[0]:
                if elbow_angle is not None:
                    st.metric("右肘の角度", f"{elbow_angle:.1f}°")
                else:
                    st.metric("右肘の角度", "N/A")
                if torso_angle is not None:
                    st.metric("体幹傾き", f"{torso_angle:.1f}°")
                else:
                    st.metric("体幹傾き", "N/A")
            with cols[1]:
                if shoulder_line_angle is not None:
                    st.metric("肩ライン角度", f"{shoulder_line_angle:.1f}°")
                else:
                    st.metric("肩ライン角度", "N/A")
                if hip_line_angle is not None:
                    st.metric("骨盤角度", f"{hip_line_angle:.1f}°")
                else:
                    st.metric("骨盤角度", "N/A")
        else:
            st.error("姿勢検出: 失敗")


def create_annotated_video(
    frames: List[np.ndarray],
    landmarks_list: List[Optional[Dict[str, Dict[str, float]]]],
    background_color: Optional[str] = None,
    trail_mode: bool = False,
    max_trail_history: int = 5,
    trail_decay: float = 0.92,
) -> Optional[str]:
    """骨格描画済みの解析動画を生成する
    
    Args:
        frames: フレームのリスト
        landmarks_list: 各フレームのランドマーク辞書のリスト
        background_color: 背景色 ('white', 'black', None=元動画)
        trail_mode: 残像トレイルモード（True: 残像表示, False: 通常）
        max_trail_history: 残像として表示する最大フレーム数
    
    Returns:
        生成された動画ファイルのパス（失敗時はNone）
    """
    if not frames or not landmarks_list:
        st.warning("フレームまたはランドマークが空です")
        return None
    
    # Cloud Run 対応: /tmp フォルダに一時ファイルを保存
    tmp_dir = "/tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    output_path = tempfile.mktemp(suffix='.mp4', dir=tmp_dir)
    
    # 動画のサイズとFPSを取得（最初のフレームから）
    if len(frames[0].shape) < 2:
        st.error("フレームの形状が不正です")
        return None
    
    # frames[0].shape[:2] の順序は (height, width) で正しい
    height, width = frames[0].shape[:2]
    
    # 動画サイズの妥当性確認
    if width <= 0 or height <= 0:
        st.error(f"動画サイズが不正です: {width}x{height}")
        return None
    
    fps = 30.0  # デフォルトFPS
    
    # Cloud Run 対応: 複数のコーデックを順番に試す
    # 優先順位: avc1 (H.264) → mp4v (MPEG-4 Part 2) → XVID
    codecs_to_try = [
        ('avc1', 'H.264 (avc1)'),
        ('mp4v', 'MPEG-4 Part 2 (mp4v)'),
        ('XVID', 'XVID'),
    ]
    
    out = None
    used_codec = None
    for codec_name, codec_desc in codecs_to_try:
        fourcc = cv2.VideoWriter_fourcc(*codec_name)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if out.isOpened():
            used_codec = codec_desc
            break
        else:
            if out is not None:
                out.release()
            out = None
    
    if out is None or not out.isOpened():
        st.error("利用可能な動画コーデックが見つかりません。ffmpeg入りOpenCVが必要です")
        return None
    
    # 使用したコーデックをログ出力（デバッグ用）
    if used_codec:
        st.info(f"動画コーデック: {used_codec}")
    
    written_frames = 0
    skipped_frames = 0
    history_landmarks: List[Optional[Dict[str, Dict[str, float]]]] = []
    
    try:
        for idx, (frame, landmarks) in enumerate(zip(frames, landmarks_list)):
            # フレームが空のときはスキップ
            if frame is None or frame.size == 0:
                skipped_frames += 1
                st.warning(f"フレーム {idx} が空のためスキップしました")
                continue
            
            # 背景色に応じてフレームを準備
            if background_color == 'white':
                # 白背景
                frame_copy = np.ones((height, width, 3), dtype=np.uint8) * 255
            elif background_color == 'black':
                # 黒背景
                frame_copy = np.zeros((height, width, 3), dtype=np.uint8)
            else:
                # 元動画を使用
                frame_copy = frame.copy()
            
            # 残像トレイルモードの場合
            if trail_mode and landmarks is not None:
                # 履歴に現在のランドマークを追加
                history_landmarks.insert(0, landmarks)
                # 最大履歴数を超えたら古いものを削除
                if len(history_landmarks) > max_trail_history:
                    history_landmarks.pop()
                
                # 残像骨格を描画
                frame_with_skeleton = draw_trail_skeleton(
                    frame_copy,
                    history_landmarks,
                    max_trail_history,
                    decay_base=trail_decay,
                )
            elif landmarks is not None:
                # 通常の骨格描画
                frame_with_skeleton = draw_landmarks_on_frame(frame_copy, landmarks)
            else:
                frame_with_skeleton = frame_copy
            
            # フレームのチャンネル数を確認し、必要に応じて BGR 3チャンネルに変換
            if len(frame_with_skeleton.shape) == 2:
                # グレースケール (1チャンネル) → BGR
                frame_with_skeleton = cv2.cvtColor(frame_with_skeleton, cv2.COLOR_GRAY2BGR)
            elif len(frame_with_skeleton.shape) == 3:
                if frame_with_skeleton.shape[2] == 4:
                    # BGRA (4チャンネル) → BGR
                    frame_with_skeleton = cv2.cvtColor(frame_with_skeleton, cv2.COLOR_BGRA2BGR)
                elif frame_with_skeleton.shape[2] == 1:
                    # 1チャンネル → BGR
                    frame_with_skeleton = cv2.cvtColor(frame_with_skeleton, cv2.COLOR_GRAY2BGR)
                elif frame_with_skeleton.shape[2] != 3:
                    st.warning(f"フレーム {idx}: 予期しないチャンネル数 ({frame_with_skeleton.shape[2]})")
                    continue
            
            # フレームサイズが一致するか確認
            if frame_with_skeleton.shape[:2] != (height, width):
                st.warning(f"フレーム {idx}: サイズ不一致 ({frame_with_skeleton.shape[:2]} vs ({height}, {width}))")
                # リサイズ
                frame_with_skeleton = cv2.resize(frame_with_skeleton, (width, height))
            
            out.write(frame_with_skeleton)
            written_frames += 1
            
    finally:
        out.release()
    
    # 書き込み後のファイルサイズを確認
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        st.info(f"動画生成完了: {output_path}, サイズ: {file_size / (1024*1024):.2f} MB, 書き込みフレーム数: {written_frames}, スキップ: {skipped_frames}")
        
        if file_size == 0:
            st.error("生成された動画ファイルが0バイトです")
            return None
    else:
        st.error(f"動画ファイルが生成されませんでした: {output_path}")
        return None
    
    return output_path


def process_video_with_pose(input_path: str) -> str:
    """
    骨格描画済み動画を生成し、出力動画パスを返す
    
    Args:
        input_path: 入力動画ファイルのパス
    
    Returns:
        出力動画ファイルのパス（失敗時は空文字列）
    """
    # 出力動画パスを生成（/tmp フォルダに保存）
    tmp_dir = "/tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    output_path = tempfile.mktemp(suffix=".mp4", dir=tmp_dir)
    
    # 動画を読み込む
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        st.error(f"動画ファイルを開けませんでした: {input_path}")
        return ""
    
    # 動画のプロパティを取得
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # フレーム数が0の場合はデフォルト値を設定
    if fps == 0:
        fps = 30
    
    # MediaPipe Pose を初期化
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5
    )
    
    # 動画ライターを初期化（H.264 コーデックを試行）
    codecs_to_try = [
        ('avc1', 'H.264 (avc1)'),
        ('mp4v', 'MPEG-4 Part 2 (mp4v)'),
        ('XVID', 'XVID'),
    ]
    
    out = None
    used_codec = None
    for codec_name, codec_desc in codecs_to_try:
        fourcc = cv2.VideoWriter_fourcc(*codec_name)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if out.isOpened():
            used_codec = codec_desc
            break
        else:
            if out is not None:
                out.release()
            out = None
    
    if out is None or not out.isOpened():
        st.error("利用可能な動画コーデックが見つかりません。ffmpeg入りOpenCVが必要です")
        cap.release()
        return ""
    
    # フレームを読み込んで処理
    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # MediaPipe は RGB 形式を期待するため変換
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 姿勢検出
            results = pose.process(rgb_frame)
            
            # 骨格を描画
            annotated_frame = frame.copy()
            if results.pose_landmarks:
                # MediaPipe の描画ユーティリティは RGB 形式を期待するため変換
                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                # MediaPipe の描画ユーティリティを使用して骨格を描画（RGB形式）
                mp_drawing.draw_landmarks(
                    annotated_frame_rgb,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )
                
                # BGR 形式に戻す
                annotated_frame = cv2.cvtColor(annotated_frame_rgb, cv2.COLOR_RGB2BGR)
            
            # フレームを書き込み
            out.write(annotated_frame)
            frame_count += 1
    
    finally:
        cap.release()
        out.release()
        # MediaPipe Pose オブジェクトは自動的にクリーンアップされる
    
    # 書き込み後のファイルサイズを確認
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        if file_size == 0:
            st.error("生成された動画ファイルが0バイトです")
            return ""
        return output_path
    else:
        st.error(f"動画ファイルが生成されませんでした: {output_path}")
        return ""


def _resize_image_if_needed(
    image: np.ndarray,
    max_pixels: int = 150000000  # PIL制限（178956970）より少し小さい値
) -> np.ndarray:
    """
    画像が大きすぎる場合にリサイズする
    
    Args:
        image: 入力画像（BGR形式）
        max_pixels: 最大ピクセル数
    
    Returns:
        リサイズされた画像（必要に応じて）
    """
    if image is None:
        return image
    
    height, width = image.shape[:2]
    current_pixels = width * height
    
    # 最大ピクセル数を超えている場合のみリサイズ
    if current_pixels <= max_pixels:
        return image
    
    # アスペクト比を計算
    aspect_ratio = width / height
    
    # 最大ピクセル数に収まるように新しいサイズを計算
    # width * height = max_pixels かつ width / height = aspect_ratio
    # height = sqrt(max_pixels / aspect_ratio)
    # width = height * aspect_ratio
    new_height = int(np.sqrt(max_pixels / aspect_ratio))
    new_width = int(new_height * aspect_ratio)
    
    # リサイズ実行
    try:
        resized_image = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA  # 縮小時に適した補間方法
        )
        return resized_image
    except Exception as e:
        # リサイズに失敗した場合は元の画像を返す
        st.warning(f"画像のリサイズに失敗しました: {e}")
        return image


def detect_pitch_phase(
    landmarks: Optional[Dict[str, Dict[str, float]]],
    prev_landmarks: Optional[Dict[str, Dict[str, float]]],
    wrist_velocity: Optional[float] = None,
    current_phase: str = "windup"
) -> str:
    """
    ランドマークから投球フェーズを判定する（簡易ルールベース）
    
    Args:
        landmarks: 現在フレームのランドマーク
        prev_landmarks: 前フレームのランドマーク
        wrist_velocity: 手首速度（オプション）
        current_phase: 現在のフェーズ（デフォルト: "windup"）
    
    Returns:
        フェーズ名: "windup", "stride", "foot_plant", "acceleration", "follow_through"
    """
    if landmarks is None:
        return current_phase  # ランドマークが存在しない場合は前のフェーズを維持
    
    # ランドマークから座標を取得
    right_ankle = landmarks.get("right_ankle", {})
    left_ankle = landmarks.get("left_ankle", {})
    
    right_ankle_y = right_ankle.get("y", 0.5) if right_ankle else 0.5
    left_ankle_x = left_ankle.get("x", 0.5) if left_ankle else 0.5
    left_ankle_y = left_ankle.get("y", 0.5) if left_ankle else 0.5
    
    # 前フレームとの比較
    if prev_landmarks:
        prev_right_ankle = prev_landmarks.get("right_ankle", {})
        prev_left_ankle = prev_landmarks.get("left_ankle", {})
        
        prev_right_ankle_y = prev_right_ankle.get("y", 0.5) if prev_right_ankle else 0.5
        prev_left_ankle_x = prev_left_ankle.get("x", 0.5) if prev_left_ankle else 0.5
        prev_left_ankle_y = prev_left_ankle.get("y", 0.5) if prev_left_ankle else 0.5
        
        # フェーズの順序性を考慮した判定
        # windup: 右足首が上昇中（前フレームより上）
        if current_phase in ["windup"] and right_ankle_y < prev_right_ankle_y - 0.01:
            return "windup"
        
        # stride: 左足首が前進中（X座標が増加）
        if current_phase in ["windup", "stride"] and left_ankle_x > prev_left_ankle_x + 0.01:
            return "stride"
        
        # foot_plant: 左足首Yが急停止（変化が小さい）
        if current_phase in ["stride", "foot_plant"] and abs(left_ankle_y - prev_left_ankle_y) < 0.005:
            return "foot_plant"
    
    # acceleration: 手首速度が最大付近
    if wrist_velocity is not None and wrist_velocity > 0.5:
        if current_phase in ["foot_plant", "acceleration"]:
            return "acceleration"
    
    # follow_through: 手首速度が低下中または後半
    if current_phase in ["acceleration", "follow_through"]:
        if wrist_velocity is not None and wrist_velocity < 0.3:
            return "follow_through"
        elif current_phase == "follow_through":
            return "follow_through"
    
    # デフォルトは現在のフェーズを維持
    return current_phase


def generate_strobe_image(video_path: str, step: int = 5, mode: str = "normal") -> Optional[np.ndarray]:
    """
    動画から一定間隔でフレームを抽出し、骨格を描画して横方向に連結したストロボ画像を生成
    
    Args:
        video_path: 入力動画ファイルのパス
        step: フレーム抽出間隔（デフォルト: 5）
        mode: 表示モード "normal"（元動画+骨格）または "skeleton"（骨格線のみ）
    
    Returns:
        連結されたストロボ画像（numpy.ndarray、BGR形式）。失敗時はNone
    """
    # 無効なmode値の場合は"normal"にフォールバック
    if mode not in ["normal", "skeleton"]:
        mode = "normal"
    # 動画を読み込む
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"動画ファイルを開けませんでした: {video_path}")
        return None
    
    # MediaPipe Pose を初期化
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5
    )
    
    # 描画済みフレームを格納するリスト
    annotated_frames = []
    frame_index = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # step間隔でフレームを抽出
            if frame_index % step == 0:
                # MediaPipe は RGB 形式を期待するため変換
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 姿勢検出
                results = pose.process(rgb_frame)
                
                # モードに応じて背景を準備
                if mode == "skeleton":
                    # skeletonモード：黒背景を生成
                    annotated_frame = np.zeros_like(frame)
                else:
                    # normalモード：元のフレームを使用
                    annotated_frame = frame.copy()
                
                # 骨格を描画（両モード共通）
                # 骨格が検出された場合のみ描画
                if results.pose_landmarks:
                    # MediaPipe の描画ユーティリティは RGB 形式を期待するため変換
                    annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    
                    # MediaPipe の描画ユーティリティを使用して骨格を描画（RGB形式）
                    # skeletonモードでは黒背景に対して明るい色で描画
                    if mode == "skeleton":
                        # skeletonモード：明るい色（白または黄色）で骨格を描画
                        mp_drawing.draw_landmarks(
                            annotated_frame_rgb,
                            results.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=3, circle_radius=3),  # 白い点
                            mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=3)  # 黄色い線
                        )
                    else:
                        # normalモード：通常の色で骨格を描画
                        mp_drawing.draw_landmarks(
                            annotated_frame_rgb,
                            results.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                        )
                    
                    # BGR 形式に戻す
                    annotated_frame = cv2.cvtColor(annotated_frame_rgb, cv2.COLOR_RGB2BGR)
                
                # 描画済みフレームをリストに追加
                # skeletonモードで骨格が検出されない場合でも、黒背景を追加
                annotated_frames.append(annotated_frame)
            
            frame_index += 1
    
    finally:
        cap.release()
        # MediaPipe Pose オブジェクトは自動的にクリーンアップされる
    
    # フレームが抽出されなかった場合
    if not annotated_frames:
        st.warning("ストロボ画像を生成するためのフレームが抽出されませんでした")
        return None
    
    # 横方向に連結
    try:
        strobe_image = cv2.hconcat(annotated_frames)
        
        # 連結後の画像サイズを確認
        height, width = strobe_image.shape[:2]
        original_pixels = width * height
        
        # PIL制限を超えないようにリサイズ
        strobe_image = _resize_image_if_needed(strobe_image)
        
        # リサイズ後のサイズを確認
        resized_height, resized_width = strobe_image.shape[:2]
        resized_pixels = resized_width * resized_height
        
        # リサイズが行われた場合に警告を表示
        if original_pixels != resized_pixels:
            st.info(f"ストロボ画像をリサイズしました: {width}x{height} → {resized_width}x{resized_height} ピクセル")
        elif width > 10000:  # 幅が10000ピクセルを超える場合（リサイズされなかった場合）
            st.warning(f"生成されたストロボ画像のサイズが大きいです（幅: {width}px）。表示に時間がかかる場合があります。")
        
        return strobe_image
    
    except Exception as e:
        st.error(f"ストロボ画像の連結中にエラーが発生しました: {e}")
        return None


def generate_phase_strobes(
    video_path: str,
    step: int = 5,
    mode: str = "normal"
) -> Dict[str, Optional[np.ndarray]]:
    """
    動画を5フェーズに分割し、各フェーズごとにストロボ画像を生成
    
    Args:
        video_path: 入力動画ファイルのパス
        step: フレーム抽出間隔（デフォルト: 5）
        mode: 表示モード "normal"（元動画+骨格）または "skeleton"（骨格線のみ）
    
    Returns:
        フェーズごとのストロボ画像の辞書
        {
            "windup": image or None,
            "stride": image or None,
            "foot_plant": image or None,
            "acceleration": image or None,
            "follow_through": image or None
        }
    """
    # 無効なmode値の場合は"normal"にフォールバック
    if mode not in ["normal", "skeleton"]:
        mode = "normal"
    
    # 動画を読み込む
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"動画ファイルを開けませんでした: {video_path}")
        return {
            "windup": None,
            "stride": None,
            "foot_plant": None,
            "acceleration": None,
            "follow_through": None
        }
    
    # MediaPipe Pose を初期化
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5
    )
    
    # フェーズごとのフレームを格納する辞書
    phase_frames: Dict[str, List[np.ndarray]] = {
        "windup": [],
        "stride": [],
        "foot_plant": [],
        "acceleration": [],
        "follow_through": []
    }
    
    # 全フレームを処理してフェーズを判定
    frames = []
    landmarks_list = []
    wrist_velocities = []
    frame_index = 0
    current_phase = "windup"
    prev_landmarks = None
    
    try:
        # まず全フレームを読み込んでランドマークを取得
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frames.append(frame)
            
            # MediaPipe は RGB 形式を期待するため変換
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 姿勢検出
            results = pose.process(rgb_frame)
            
            # ランドマークを取得
            landmarks = None
            if results.pose_landmarks:
                from pose.mediapipe_pose import normalize_landmarks
                landmarks = normalize_landmarks(results.pose_landmarks)
            
            landmarks_list.append(landmarks)
            frame_index += 1
        
        # 手首速度を計算
        if landmarks_list:
            wrist_velocities = calculate_wrist_velocity(landmarks_list)
        
        # 各フレームでフェーズを判定
        for i, (frame, landmarks) in enumerate(zip(frames, landmarks_list)):
            # step間隔でフレームを抽出
            if i % step == 0:
                wrist_vel = wrist_velocities[i] if i < len(wrist_velocities) else None
                
                # フェーズを判定
                current_phase = detect_pitch_phase(
                    landmarks,
                    prev_landmarks,
                    wrist_vel,
                    current_phase
                )
                
                # フェーズごとにフレームを分類
                if current_phase in phase_frames:
                    phase_frames[current_phase].append(frame)
                
                prev_landmarks = landmarks
    
    finally:
        cap.release()
        # MediaPipe Pose オブジェクトは自動的にクリーンアップされる
    
    # 各フェーズのストロボ画像を生成
    phase_strobes: Dict[str, Optional[np.ndarray]] = {}
    
    for phase_name, frames_list in phase_frames.items():
        if not frames_list:
            phase_strobes[phase_name] = None
            continue
        
        # フェーズごとのフレームに対してストロボ画像を生成
        annotated_frames = []
        
        # MediaPipe Pose を再初期化
        pose_phase = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )
        
        try:
            for frame in frames_list:
                # MediaPipe は RGB 形式を期待するため変換
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 姿勢検出
                results = pose_phase.process(rgb_frame)
                
                # モードに応じて背景を準備
                if mode == "skeleton":
                    annotated_frame = np.zeros_like(frame)
                else:
                    annotated_frame = frame.copy()
                
                # 骨格を描画
                if results.pose_landmarks:
                    annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    
                    if mode == "skeleton":
                        mp_drawing.draw_landmarks(
                            annotated_frame_rgb,
                            results.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=3, circle_radius=3),
                            mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=3)
                        )
                    else:
                        mp_drawing.draw_landmarks(
                            annotated_frame_rgb,
                            results.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                        )
                    
                    annotated_frame = cv2.cvtColor(annotated_frame_rgb, cv2.COLOR_RGB2BGR)
                
                annotated_frames.append(annotated_frame)
            
            # 横方向に連結
            if annotated_frames:
                try:
                    strobe_image = cv2.hconcat(annotated_frames)
                    # PIL制限を超えないようにリサイズ
                    strobe_image = _resize_image_if_needed(strobe_image)
                    phase_strobes[phase_name] = strobe_image
                except Exception as e:
                    st.warning(f"{phase_name}フェーズのストロボ画像生成に失敗しました: {e}")
                    phase_strobes[phase_name] = None
            else:
                phase_strobes[phase_name] = None
        
        finally:
            # MediaPipe Pose オブジェクトは自動的にクリーンアップされる
            pass
    
    return phase_strobes


def select_phase_keyframes(
    frames: List[np.ndarray],
    landmarks_list: List[Optional[Dict[str, Dict[str, float]]]],
    wrist_velocities: List[Optional[float]]
) -> Dict[str, Optional[int]]:
    """
    各フェーズの代表フレームインデックスを選択
    
    Args:
        frames: フレームのリスト
        landmarks_list: 各フレームのランドマーク辞書のリスト
        wrist_velocities: 各フレームの手首速度のリスト
    
    Returns:
        フェーズごとの代表フレームインデックスの辞書
        {
            "windup": frame_index or None,
            "stride": frame_index or None,
            "plant": frame_index or None,
            "acceleration": frame_index or None,
            "follow": frame_index or None
        }
    """
    # フェーズごとのフレームインデックスを格納
    phase_frames: Dict[str, List[int]] = {
        "windup": [],
        "stride": [],
        "plant": [],
        "acceleration": [],
        "follow": []
    }
    
    # フェーズごとにフレームを分類
    current_phase = "windup"
    prev_landmarks = None
    
    for i, landmarks in enumerate(landmarks_list):
        wrist_vel = wrist_velocities[i] if i < len(wrist_velocities) else None
        current_phase = detect_pitch_phase(landmarks, prev_landmarks, wrist_vel, current_phase)
        
        # フェーズ名のマッピング（foot_plant -> plant, follow_through -> follow）
        phase_key = current_phase
        if current_phase == "foot_plant":
            phase_key = "plant"
        elif current_phase == "follow_through":
            phase_key = "follow"
        
        if phase_key in phase_frames:
            phase_frames[phase_key].append(i)
        
        prev_landmarks = landmarks
    
    # 各フェーズの代表フレームを選択
    keyframes: Dict[str, Optional[int]] = {}
    
    # windup: 右足首Y座標が最大（最も足が上がった瞬間）
    if phase_frames["windup"]:
        max_y = -1.0
        keyframe_idx = None
        for idx in phase_frames["windup"]:
            landmarks = landmarks_list[idx]
            if landmarks and "right_ankle" in landmarks:
                y = landmarks["right_ankle"]["y"]
                if y > max_y:
                    max_y = y
                    keyframe_idx = idx
        keyframes["windup"] = keyframe_idx
    else:
        keyframes["windup"] = None
    
    # stride: 左足首X座標が最前方（最大値）
    if phase_frames["stride"]:
        max_x = -1.0
        keyframe_idx = None
        for idx in phase_frames["stride"]:
            landmarks = landmarks_list[idx]
            if landmarks and "left_ankle" in landmarks:
                x = landmarks["left_ankle"]["x"]
                if x > max_x:
                    max_x = x
                    keyframe_idx = idx
        keyframes["stride"] = keyframe_idx
    else:
        keyframes["stride"] = None
    
    # plant: 左足首Y速度が0に近い瞬間（接地）
    if phase_frames["plant"]:
        min_vel = float('inf')
        keyframe_idx = None
        prev_y = None
        for idx in phase_frames["plant"]:
            landmarks = landmarks_list[idx]
            if landmarks and "left_ankle" in landmarks:
                y = landmarks["left_ankle"]["y"]
                if prev_y is not None:
                    vel = abs(y - prev_y)
                    if vel < min_vel:
                        min_vel = vel
                        keyframe_idx = idx
                prev_y = y
        keyframes["plant"] = keyframe_idx
    else:
        keyframes["plant"] = None
    
    # acceleration: 手首速度最大（リリース直前）
    if phase_frames["acceleration"]:
        max_vel = -1.0
        keyframe_idx = None
        for idx in phase_frames["acceleration"]:
            vel = wrist_velocities[idx] if idx < len(wrist_velocities) else None
            if vel is not None and vel > max_vel:
                max_vel = vel
                keyframe_idx = idx
        keyframes["acceleration"] = keyframe_idx
    else:
        keyframes["acceleration"] = None
    
    # follow: 右肩Y座標が最小（体幹が最も前傾）
    if phase_frames["follow"]:
        min_y = float('inf')
        keyframe_idx = None
        for idx in phase_frames["follow"]:
            landmarks = landmarks_list[idx]
            if landmarks and "right_shoulder" in landmarks:
                y = landmarks["right_shoulder"]["y"]
                if y < min_y:
                    min_y = y
                    keyframe_idx = idx
        keyframes["follow"] = keyframe_idx
    else:
        keyframes["follow"] = None
    
    return keyframes


def generate_phase_keyframe_strobe(
    video_path: str,
    mode: str = "normal"
) -> Optional[np.ndarray]:
    """
    各フェーズの代表1フレームを抽出して横連結したストロボ画像を生成
    
    Args:
        video_path: 入力動画ファイルのパス
        mode: 表示モード "normal"（元動画+骨格）または "skeleton"（骨格線のみ）
    
    Returns:
        連結されたストロボ画像（numpy.ndarray、BGR形式）。失敗時はNone
    """
    # 無効なmode値の場合は"normal"にフォールバック
    if mode not in ["normal", "skeleton"]:
        mode = "normal"
    
    # 動画を読み込む
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"動画ファイルを開けませんでした: {video_path}")
        return None
    
    # MediaPipe Pose を初期化
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5
    )
    
    # 全フレームを読み込んでランドマークを取得
    frames = []
    landmarks_list = []
    frame_index = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frames.append(frame)
            
            # MediaPipe は RGB 形式を期待するため変換
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 姿勢検出
            results = pose.process(rgb_frame)
            
            # ランドマークを取得
            landmarks = None
            if results.pose_landmarks:
                from pose.mediapipe_pose import normalize_landmarks
                landmarks = normalize_landmarks(results.pose_landmarks)
            
            landmarks_list.append(landmarks)
            frame_index += 1
    
    finally:
        cap.release()
        # MediaPipe Pose オブジェクトは自動的にクリーンアップされる
    
    if not frames or not landmarks_list:
        st.warning("動画からフレームを読み込めませんでした")
        return None
    
    # 手首速度を計算
    wrist_velocities = calculate_wrist_velocity(landmarks_list)
    
    # 各フェーズの代表フレームを選択
    keyframes = select_phase_keyframes(frames, landmarks_list, wrist_velocities)
    
    # 代表フレームに対して骨格描画処理を実行
    annotated_keyframes = []
    phase_order = ["windup", "stride", "plant", "acceleration", "follow"]
    
    # MediaPipe Pose を再初期化
    pose_keyframe = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5
    )
    
    try:
        for phase_name in phase_order:
            keyframe_idx = keyframes.get(phase_name)
            
            if keyframe_idx is None or keyframe_idx >= len(frames):
                # 代表フレームが選択できない場合は、黒背景のプレースホルダーを追加
                if len(annotated_keyframes) > 0:
                    # 既存のフレームと同じサイズの黒背景を作成
                    placeholder = np.zeros_like(annotated_keyframes[0])
                    annotated_keyframes.append(placeholder)
                else:
                    # 最初のフレームがない場合は、最初のフレームのサイズを使用
                    if frames:
                        placeholder = np.zeros_like(frames[0])
                        annotated_keyframes.append(placeholder)
                continue
            
            frame = frames[keyframe_idx]
            
            # MediaPipe は RGB 形式を期待するため変換
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 姿勢検出
            results = pose_keyframe.process(rgb_frame)
            
            # モードに応じて背景を準備
            if mode == "skeleton":
                annotated_frame = np.zeros_like(frame)
            else:
                annotated_frame = frame.copy()
            
            # 骨格を描画
            if results.pose_landmarks:
                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                if mode == "skeleton":
                    mp_drawing.draw_landmarks(
                        annotated_frame_rgb,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=3, circle_radius=3),
                        mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=3)
                    )
                else:
                    mp_drawing.draw_landmarks(
                        annotated_frame_rgb,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                    )
                
                annotated_frame = cv2.cvtColor(annotated_frame_rgb, cv2.COLOR_RGB2BGR)
            
            annotated_keyframes.append(annotated_frame)
    
    finally:
        # MediaPipe Pose オブジェクトは自動的にクリーンアップされる
        pass
    
    # 横方向に連結
    if not annotated_keyframes:
        st.warning("代表フレームが選択できませんでした")
        return None
    
    try:
        strobe_image = cv2.hconcat(annotated_keyframes)
        
        # PIL制限を超えないようにリサイズ
        strobe_image = _resize_image_if_needed(strobe_image)
        
        return strobe_image
    
    except Exception as e:
        st.error(f"ストロボ画像の連結中にエラーが発生しました: {e}")
        return None


def calc_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Optional[float]:
    """
    3点から角度を計算（度数法）
    
    Args:
        a: 第1点の座標 (x, y) または (x, y, z)
        b: 第2点の座標（角度の頂点）(x, y) または (x, y, z)
        c: 第3点の座標 (x, y) または (x, y, z)
    
    Returns:
        角度（度）。計算できない場合はNone
    """
    try:
        # aからbへのベクトル
        vector1 = b - a
        
        # bからcへのベクトル
        vector2 = c - b
        
        # ベクトルのノルムを計算
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        
        # ゼロ除算を防ぐ
        if norm1 == 0.0 or norm2 == 0.0:
            return None
        
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
    except Exception:
        # エラー時はNoneを返す
        return None


def extract_angles_from_video(video_path: str) -> Dict[str, List[Optional[float]]]:
    """
    動画の全フレームを処理し、関節角度を抽出する
    
    Args:
        video_path: 入力動画ファイルのパス
    
    Returns:
        角度配列の辞書
        {
            "elbow": [...],    # 右肘角度（肩-肘-手首）
            "knee": [...],     # 右膝角度（股関節-膝-足首）
            "hip": [...]       # 右股関節角度（肩-股関節-膝）
        }
    """
    # 結果を格納する辞書
    angles = {
        "elbow": [],
        "knee": [],
        "hip": []
    }
    
    # MediaPipe Pose を初期化
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5
    )
    
    # 動画を読み込む
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"動画ファイルを開けませんでした: {video_path}")
        return angles
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # MediaPipe は RGB 形式を期待するため変換
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 姿勢検出
            results = pose.process(rgb_frame)
            
            # ランドマークが検出された場合
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # 必要なランドマークを取得
                # MediaPipe Pose ランドマークインデックス
                RIGHT_SHOULDER_IDX = 12
                RIGHT_ELBOW_IDX = 14
                RIGHT_WRIST_IDX = 16
                RIGHT_HIP_IDX = 24
                RIGHT_KNEE_IDX = 26
                RIGHT_ANKLE_IDX = 28
                
                # ランドマークの座標を取得（正規化座標 0-1）
                try:
                    shoulder = landmarks[RIGHT_SHOULDER_IDX]
                    elbow = landmarks[RIGHT_ELBOW_IDX]
                    wrist = landmarks[RIGHT_WRIST_IDX]
                    hip = landmarks[RIGHT_HIP_IDX]
                    knee = landmarks[RIGHT_KNEE_IDX]
                    ankle = landmarks[RIGHT_ANKLE_IDX]
                    
                    # 可視性チェック（0.5以上の場合のみ使用）
                    visibility_threshold = 0.5
                    
                    # 右肘角度を計算（肩-肘-手首）
                    if (shoulder.visibility >= visibility_threshold and
                        elbow.visibility >= visibility_threshold and
                        wrist.visibility >= visibility_threshold):
                        shoulder_pt = np.array([shoulder.x, shoulder.y, shoulder.z])
                        elbow_pt = np.array([elbow.x, elbow.y, elbow.z])
                        wrist_pt = np.array([wrist.x, wrist.y, wrist.z])
                        elbow_angle = calc_angle(shoulder_pt, elbow_pt, wrist_pt)
                        angles["elbow"].append(elbow_angle)
                    else:
                        angles["elbow"].append(None)
                    
                    # 右膝角度を計算（股関節-膝-足首）
                    if (hip.visibility >= visibility_threshold and
                        knee.visibility >= visibility_threshold and
                        ankle.visibility >= visibility_threshold):
                        hip_pt = np.array([hip.x, hip.y, hip.z])
                        knee_pt = np.array([knee.x, knee.y, knee.z])
                        ankle_pt = np.array([ankle.x, ankle.y, ankle.z])
                        knee_angle = calc_angle(hip_pt, knee_pt, ankle_pt)
                        angles["knee"].append(knee_angle)
                    else:
                        angles["knee"].append(None)
                    
                    # 右股関節角度を計算（肩-股関節-膝）
                    if (shoulder.visibility >= visibility_threshold and
                        hip.visibility >= visibility_threshold and
                        knee.visibility >= visibility_threshold):
                        shoulder_pt = np.array([shoulder.x, shoulder.y, shoulder.z])
                        hip_pt = np.array([hip.x, hip.y, hip.z])
                        knee_pt = np.array([knee.x, knee.y, knee.z])
                        hip_angle = calc_angle(shoulder_pt, hip_pt, knee_pt)
                        angles["hip"].append(hip_angle)
                    else:
                        angles["hip"].append(None)
                
                except (IndexError, AttributeError) as e:
                    # ランドマークが不足している場合はNoneを追加
                    angles["elbow"].append(None)
                    angles["knee"].append(None)
                    angles["hip"].append(None)
            else:
                # ランドマークが検出されなかった場合
                angles["elbow"].append(None)
                angles["knee"].append(None)
                angles["hip"].append(None)
    
    finally:
        cap.release()
        # MediaPipe Pose オブジェクトは自動的にクリーンアップされる
    
    return angles


def detect_peak_knee_frame(video_path: str) -> Optional[int]:
    """
    右膝(RIGHT_KNEE)の y 座標が最小となるフレーム番号を返す
    ＝ 足上げトップ（膝が最も高い位置）
    
    注意: MediaPipeのy座標は「下が大きい値」なので、
    膝が最も上にある時はy座標が最小値になる
    
    Args:
        video_path: 入力動画ファイルのパス
    
    Returns:
        足上げトップのフレーム番号（検出失敗時はNone）
    """
    # 右膝のy座標を格納するリスト
    knee_y_coords = []
    
    # MediaPipe Pose を初期化
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5
    )
    
    # 動画を読み込む
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"動画ファイルを開けませんでした: {video_path}")
        return None
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # MediaPipe は RGB 形式を期待するため変換
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 姿勢検出
            results = pose.process(rgb_frame)
            
            # ランドマークが検出された場合
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # 右膝のランドマークインデックス
                RIGHT_KNEE_IDX = 26
                
                try:
                    knee = landmarks[RIGHT_KNEE_IDX]
                    # 可視性チェック（0.5以上の場合のみ使用）
                    if knee.visibility >= 0.5:
                        # y座標を保存（MediaPipeは下が大きい値）
                        knee_y_coords.append(knee.y)
                    else:
                        # 検出されない場合はNoneを追加
                        knee_y_coords.append(None)
                except (IndexError, AttributeError):
                    # ランドマークが不足している場合はNoneを追加
                    knee_y_coords.append(None)
            else:
                # ランドマークが検出されなかった場合
                knee_y_coords.append(None)
    
    finally:
        cap.release()
        # MediaPipe Pose オブジェクトは自動的にクリーンアップされる
    
    # 有効なy座標のみを抽出
    valid_y_coords = [y for y in knee_y_coords if y is not None]
    
    if not valid_y_coords:
        # 有効なデータがない場合
        return None
    
    # 最小値（膝が最も高い位置）のインデックスを取得
    # 元のリストでのインデックスを取得するため、有効な値のみでargminを計算
    valid_indices = [i for i, y in enumerate(knee_y_coords) if y is not None]
    min_y_value = min(valid_y_coords)
    min_y_index_in_valid = valid_y_coords.index(min_y_value)
    peak_frame = valid_indices[min_y_index_in_valid]
    
    return peak_frame


def sync_angle_sequences(
    seq1: List[Optional[float]], 
    seq2: List[Optional[float]], 
    peak1: int, 
    peak2: int
) -> Tuple[List[Optional[float]], List[Optional[float]]]:
    """
    2つの角度配列をピーク位置で同期させる
    短い方に長さを合わせて返す
    
    Args:
        seq1: 第1の角度配列
        seq2: 第2の角度配列
        peak1: 第1配列のピーク位置（フレーム番号）
        peak2: 第2配列のピーク位置（フレーム番号）
    
    Returns:
        同期後の2つの角度配列のタプル
    """
    # オフセットを計算
    offset = peak1 - peak2
    
    # オフセットに応じて配列をシフト
    if offset > 0:
        # seq1が遅れている場合、seq1の先頭を削除
        seq1_synced = seq1[offset:]
        seq2_synced = seq2
    elif offset < 0:
        # seq2が遅れている場合、seq2の先頭を削除
        seq1_synced = seq1
        seq2_synced = seq2[-offset:]
    else:
        # オフセットがない場合、そのまま
        seq1_synced = seq1
        seq2_synced = seq2
    
    # 短い方の長さに合わせる
    min_len = min(len(seq1_synced), len(seq2_synced))
    seq1_synced = seq1_synced[:min_len]
    seq2_synced = seq2_synced[:min_len]
    
    return seq1_synced, seq2_synced


def _render_angle_chart(elbow_angles: List[Optional[float]]) -> None:
    """角度グラフを表示する
    
    Args:
        elbow_angles: 各フレームの肘の角度のリスト
    """
    if not any(angle is not None for angle in elbow_angles):
        return
    
    st.subheader("右肘の角度変化")
    
    fig, ax = plt.subplots()
    valid_angles = [angle if angle is not None else 0.0 for angle in elbow_angles]
    ax.plot(valid_angles)
    ax.set_xlabel("フレーム番号")
    ax.set_ylabel("角度（度）")
    ax.set_title("右肘の角度変化")
    ax.grid(True)
    st.pyplot(fig)


def _render_multi_analysis_charts(
    analysis_data: Dict[str, List[Optional[float]]],
    selected_analyses: List[str]
) -> None:
    """複数の解析結果をグラフ表示する
    
    Args:
        analysis_data: 解析データの辞書
        selected_analyses: 表示する解析種類のリスト
    """
    if not selected_analyses:
        return
    
    st.subheader("解析結果グラフ")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for analysis_name in selected_analyses:
        if analysis_name not in analysis_data:
            continue
        
        values = analysis_data[analysis_name]
        valid_values = [v if v is not None else 0.0 for v in values]
        
        ax.plot(valid_values, label=analysis_name, alpha=0.7)
    
    ax.set_xlabel("フレーム番号")
    ax.set_ylabel("値")
    ax.set_title("解析結果の時間変化")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)


def _render_phase_summary(phase_summary: Dict[str, Dict[str, float]]) -> None:
    """投球フェーズごとのサマリーを表示する
    
    Args:
        phase_summary: フェーズごとのサマリーデータ
    """
    if not phase_summary:
        return
    
    st.subheader("投球フェーズ別サマリー")
    
    for phase_name, phase_data in phase_summary.items():
        with st.expander(f"📊 {phase_name}", expanded=False):
            if phase_data:
                cols = st.columns(min(3, len(phase_data)))
                for idx, (key, value) in enumerate(phase_data.items()):
                    with cols[idx % len(cols)]:
                        st.metric(key.replace("_", " ").title(), f"{value:.2f}")
            else:
                st.info("データなし")


def _render_video_list_panel() -> None:
    """左側パネル：解析済み動画リストとサムネイル表示"""
    st.subheader("📁 解析済み動画")
    
    video_list = st.session_state.get("video_list", [])
    
    if not video_list:
        st.info("📤 動画をアップロードして解析を開始してください")
        return
    
    # 動画リストを表示
    for idx, video_data in enumerate(video_list):
        # カード風レイアウト
        with st.container():
            # 選択状態のハイライト
            is_selected = idx == st.session_state.get("selected_video_index", 0)
            border_color = "#1f77b4" if is_selected else "#e0e0e0"
            
            st.markdown(
                f"""
                <div style="border: 2px solid {border_color}; border-radius: 8px; padding: 10px; margin-bottom: 10px; 
                            background-color: {'#f0f8ff' if is_selected else '#ffffff'};">
                """,
                unsafe_allow_html=True
            )
            
            # 動画名
            st.markdown(f"**{idx + 1}. {video_data.get('name', f'動画 {idx + 1}')}**")
            
            # サムネイル表示
            frames = video_data.get("frames", [])
            if frames and len(frames) > 0:
                # 最初のフレームをサムネイルとして使用
                thumb_frame = frames[0].copy()
                thumb_rgb = cv2.cvtColor(thumb_frame, cv2.COLOR_BGR2RGB)
                st.image(thumb_rgb, use_container_width=True, caption="サムネイル")
            
            # 選択ボタン
            if st.button(f"選択", key=f"select_video_{idx}", use_container_width=True):
                st.session_state["selected_video_index"] = idx
                st.rerun()
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    # 統計情報
    st.markdown("---")
    st.markdown(f"**合計: {len(video_list)} 動画**")


def _render_analysis_tab(
    frames: List[np.ndarray],
    landmarks_list: List[Optional[Dict[str, Dict[str, float]]]],
    elbow_angles: List[Optional[float]]
) -> None:
    """解析結果タブ：メトリクスとフレームビューアー"""
    # 解析種類選択
    analysis_types = {
        "肘角度": "right_elbow",
        "肩角度": "right_shoulder",
        "膝角度": "right_knee",
        "腰角度": "right_hip",
        "体幹傾き": "torso_axis",
        "肩ライン角度": "shoulder_line",
        "骨盤ライン角度": "hip_line",
        "手首速度": "wrist_velocity",
        "投球フェーズ": "pitching_phases",
    }
    
    selected_analysis_names = st.multiselect(
        "📊 解析種類を選択",
        options=list(analysis_types.keys()),
        default=["肘角度", "体幹傾き"],
        key="analysis_type_select_tab"
    )
    
    # 解析データを計算
    analysis_data = {}
    phases = None
    phase_summary = None
    
    # 各種角度を計算
    if any(name in ["肘角度", "肩角度", "膝角度", "腰角度", "体幹傾き", "肩ライン角度", "骨盤ライン角度"] for name in selected_analysis_names):
        all_angles = calculate_all_angles_from_landmarks(landmarks_list)
        angle_mapping = {
            "肘角度": "right_elbow",
            "肩角度": "right_shoulder",
            "膝角度": "right_knee",
            "腰角度": "right_hip",
            "体幹傾き": "torso_axis",
            "肩ライン角度": "shoulder_line",
            "骨盤ライン角度": "hip_line",
        }
        for name, key in angle_mapping.items():
            if name in selected_analysis_names:
                analysis_data[name] = all_angles[key]
    
    # 手首速度を計算
    if "手首速度" in selected_analysis_names:
        wrist_velocities = calculate_wrist_velocity(landmarks_list)
        analysis_data["手首速度"] = wrist_velocities
    
    # 投球フェーズを推定
    if "投球フェーズ" in selected_analysis_names:
        wrist_velocities = calculate_wrist_velocity(landmarks_list)
        phases = detect_pitching_phases(landmarks_list, elbow_angles, wrist_velocities)
        if phases:
            all_angles = calculate_all_angles_from_landmarks(landmarks_list)
            phase_summary = calculate_phase_summary(phases, all_angles, wrist_velocities)
    
    # フレーム選択
    frame_idx = st.slider(
        "🎬 フレームを選択",
        min_value=0,
        max_value=len(frames) - 1,
        value=0,
        key="frame_slider_tab"
    )
    
    # 現在フレームの角度を取得
    torso_angle_val = None
    shoulder_line_angle_val = None
    hip_line_val = None
    
    if "体幹傾き" in selected_analysis_names and "torso_axis" in analysis_data:
        torso_angle_val = analysis_data["体幹傾き"][frame_idx] if frame_idx < len(analysis_data["体幹傾き"]) else None
    
    if "肩ライン角度" in selected_analysis_names and "shoulder_line" in analysis_data:
        shoulder_line_angle_val = analysis_data["肩ライン角度"][frame_idx] if frame_idx < len(analysis_data["肩ライン角度"]) else None
    
    if "骨盤ライン角度" in selected_analysis_names and "hip_line" in analysis_data:
        hip_line_val = analysis_data["骨盤ライン角度"][frame_idx] if frame_idx < len(analysis_data["骨盤ライン角度"]) else None
    
    # フレームビューアー表示
    _render_frame_viewer(
        frames,
        frame_idx,
        landmarks_list[frame_idx],
        elbow_angles[frame_idx] if "肘角度" in selected_analysis_names else None,
        torso_angle=torso_angle_val,
        shoulder_line_angle=shoulder_line_angle_val,
        hip_line_angle=hip_line_val,
    )
    
    # 投球フェーズサマリー表示
    if phase_summary:
        _render_phase_summary(phase_summary)


def _render_graph_tab(
    frames: List[np.ndarray],
    landmarks_list: List[Optional[Dict[str, Dict[str, float]]]],
    elbow_angles: List[Optional[float]]
) -> None:
    """グラフタブ：時系列グラフ表示"""
    # 解析種類選択
    analysis_types = {
        "肘角度": "right_elbow",
        "肩角度": "right_shoulder",
        "膝角度": "right_knee",
        "腰角度": "right_hip",
        "体幹傾き": "torso_axis",
        "肩ライン角度": "shoulder_line",
        "骨盤ライン角度": "hip_line",
        "手首速度": "wrist_velocity",
    }
    
    selected_analysis_names = st.multiselect(
        "📈 表示するグラフを選択",
        options=list(analysis_types.keys()),
        default=["肘角度", "体幹傾き"],
        key="graph_analysis_select"
    )
    
    if not selected_analysis_names:
        st.info("表示するグラフを選択してください")
        return
    
    # 解析データを計算
    analysis_data = {}
    
    # 各種角度を計算
    if any(name in ["肘角度", "肩角度", "膝角度", "腰角度", "体幹傾き", "肩ライン角度", "骨盤ライン角度"] for name in selected_analysis_names):
        all_angles = calculate_all_angles_from_landmarks(landmarks_list)
        angle_mapping = {
            "肘角度": "right_elbow",
            "肩角度": "right_shoulder",
            "膝角度": "right_knee",
            "腰角度": "right_hip",
            "体幹傾き": "torso_axis",
            "肩ライン角度": "shoulder_line",
            "骨盤ライン角度": "hip_line",
        }
        for name, key in angle_mapping.items():
            if name in selected_analysis_names:
                analysis_data[name] = all_angles[key]
    
    # 手首速度を計算
    if "手首速度" in selected_analysis_names:
        wrist_velocities = calculate_wrist_velocity(landmarks_list)
        analysis_data["手首速度"] = wrist_velocities
    
    # グラフ表示
    if analysis_data:
        _render_multi_analysis_charts(analysis_data, selected_analysis_names)
    else:
        st.warning("表示するデータがありません")


def _render_video_tab(
    frames: List[np.ndarray],
    landmarks_list: List[Optional[Dict[str, Dict[str, float]]]],
    video_data: Dict[str, Any]
) -> None:
    """解析動画タブ：骨格描画済み動画の生成と表示"""
    st.subheader("🎬 解析動画生成")
    
    # 表示モード選択
    display_mode_video = st.radio(
        "表示モード",
        ["通常骨格", "残像トレイル"],
        key="display_mode_video_radio_tab",
        horizontal=True
    )
    trail_mode = display_mode_video == "残像トレイル"
    
    # 残像フレーム数スライダー（残像モードの場合のみ表示）
    max_trail_history = 20
    trail_decay = 0.92
    if trail_mode:
        max_trail_history = st.slider(
            "残像フレーム数",
            min_value=5,
            max_value=30,
            value=20,
            key="trail_history_slider_tab"
        )
        trail_decay = st.slider(
            "残像の濃さ（大きいほど濃い）",
            min_value=0.85,
            max_value=0.98,
            value=0.92,
            step=0.01,
            key="trail_decay_slider_tab"
        )
    
    # 動画タイプ選択ボタン
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🎥 元動画+解析線を生成", key="generate_overlay_video_tab", use_container_width=True):
            with st.spinner("元動画に解析線を重ねた動画を生成しています..."):
                overlay_video_path = create_annotated_video(
                    frames,
                    landmarks_list,
                    background_color=None,
                    trail_mode=trail_mode,
                    max_trail_history=max_trail_history,
                    trail_decay=trail_decay,
                )
                video_data["annotated_overlay_path"] = overlay_video_path
                if overlay_video_path and os.path.exists(overlay_video_path):
                    file_size = os.path.getsize(overlay_video_path)
                    st.success(f"✅ 生成完了: {file_size / (1024*1024):.2f} MB")
                else:
                    st.error("❌ 動画の生成に失敗しました")
    
    with col2:
        # 背景色選択
        bg_color = st.radio("解析線のみ動画の背景色", ["黒", "白"], key="bg_color_radio_tab", horizontal=True)
        bg_color_value = "black" if bg_color == "黒" else "white"
        
        if st.button("🎨 解析線のみを生成", key="generate_skeleton_video_tab", use_container_width=True):
            with st.spinner(f"解析線のみの動画を生成しています（背景: {bg_color}）..."):
                skeleton_video_path = create_annotated_video(
                    frames,
                    landmarks_list,
                    background_color=bg_color_value,
                    trail_mode=trail_mode,
                    max_trail_history=max_trail_history,
                    trail_decay=trail_decay,
                )
                video_data["annotated_skeleton_path"] = skeleton_video_path
                if skeleton_video_path and os.path.exists(skeleton_video_path):
                    file_size = os.path.getsize(skeleton_video_path)
                    st.success(f"✅ 生成完了: {file_size / (1024*1024):.2f} MB")
                else:
                    st.error("❌ 動画の生成に失敗しました")
    
    # 動画タイプ選択
    video_options = []
    if video_data.get("annotated_overlay_path") and os.path.exists(video_data["annotated_overlay_path"]):
        video_options.append("元動画+解析線")
    if video_data.get("annotated_skeleton_path") and os.path.exists(video_data["annotated_skeleton_path"]):
        video_options.append("解析線のみ")
    
    if video_options:
        selected_type = st.radio("表示する動画を選択", video_options, key="video_type_radio_tab", horizontal=True)
        
        # 選択された動画のパスを取得
        if selected_type == "元動画+解析線":
            current_video_path = video_data.get("annotated_overlay_path")
        else:
            current_video_path = video_data.get("annotated_skeleton_path")
        
        # 動画表示・ダウンロード
        if current_video_path and os.path.exists(current_video_path):
            st.video(current_video_path)
            
            # ダウンロードボタン
            with open(current_video_path, "rb") as video_file:
                video_bytes = video_file.read()
                st.download_button(
                    label="📥 動画をダウンロード",
                    data=video_bytes,
                    file_name=f"pitching_analysis_{selected_type.replace('+', '_').replace(' ', '_')}.mp4",
                    mime="video/mp4",
                    key="download_video_button_tab",
                    use_container_width=True
                )
        else:
            st.warning("選択された動画ファイルが見つかりません。再度生成してください。")
    else:
        st.info("💡 上記のボタンで動画を生成してください")


def _render_evaluation_tab(
    frames: List[np.ndarray],
    landmarks_list: List[Optional[Dict[str, Dict[str, float]]]],
    elbow_angles: List[Optional[float]],
) -> None:
    """評価タブ：フォームの自動評価を表示"""
    if not frames or not landmarks_list:
        st.info("評価する動画がありません")
        return

    st.subheader("⭐ フォーム評価")

    # 解析に必要なデータを計算
    all_angles = calculate_all_angles_from_landmarks(landmarks_list)
    wrist_velocities = calculate_wrist_velocity(landmarks_list)
    metrics = compute_pitching_metrics(
        landmarks_list,
        elbow_angles,
        all_angles,
        wrist_velocities,
    )
    eval_result = evaluate_pitching_form(metrics)

    score = eval_result.get("score", 0)
    subscores = eval_result.get("subscores", {})
    comments = eval_result.get("comments", [])

    # スコア表示
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("総合スコア", f"{score} / 100")

    with col2:
        st.markdown("**代表的なメトリクス**")
        m = metrics
        st.write(
            {
                "最大肘角度": m.get("max_elbow_angle"),
                "リリース時肘角度": m.get("release_elbow_angle"),
                "体幹傾き（リリース時）": m.get("torso_angle_at_release"),
                "肩ライン角度（リリース時）": m.get("shoulder_angle_at_release"),
                "骨盤角度（リリース時）": m.get("hip_angle_at_release"),
                "リリースフレーム": m.get("release_frame"),
            }
        )

    st.markdown("---")

    # レーダーチャート（サブスコア）
    if subscores:
        st.markdown("#### レーダーチャート（指標別スコア）")
        labels = list(subscores.keys())
        values = [subscores[k] for k in labels]

        # レーダーチャート用に閉じる
        values.append(values[0])

        angles = np.linspace(0, 2 * np.pi, len(values), endpoint=True)

        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
        ax.plot(angles, values, "o-", linewidth=2)
        ax.fill(angles, values, alpha=0.25)
        ax.set_thetagrids(
            angles[:-1] * 180 / np.pi,
            labels,
            fontsize=10,
        )
        ax.set_ylim(0, 100)
        ax.set_title("フォーム評価レーダーチャート", pad=20)
        ax.grid(True)
        st.pyplot(fig)
    else:
        st.info("評価用のサブスコアを計算できませんでした。")

    # コメント表示
    st.markdown("#### コメント")
    for c in comments:
        st.markdown(f"- {c}")


def _render_video_detail_panel() -> None:
    """右側パネル：選択動画の詳細表示（タブ構成）"""
    video_list = st.session_state.get("video_list", [])
    
    if not video_list:
        st.info("📤 左側から動画を選択するか、新しい動画をアップロードして解析を開始してください")
        return
    
    selected_idx = st.session_state.get("selected_video_index", 0)
    if selected_idx >= len(video_list):
        selected_idx = 0
        st.session_state["selected_video_index"] = 0
    
    video_data = video_list[selected_idx]
    frames = video_data.get("frames", [])
    landmarks_list = video_data.get("landmarks", [])
    elbow_angles = video_data.get("elbow_angles", [])
    
    # 動画情報表示
    st.markdown(f"### 📹 {video_data.get('name', '動画')}")
    st.markdown(f"**フレーム数:** {len(frames)} | **解析済み:** ✅")
    
    # タブ構成
    tabs = st.tabs(["📊 解析結果", "📈 グラフ", "🎬 解析動画", "⭐ 評価"])
    
    with tabs[0]:
        _render_analysis_tab(frames, landmarks_list, elbow_angles)
    
    with tabs[1]:
        _render_graph_tab(frames, landmarks_list, elbow_angles)
    
    with tabs[2]:
        _render_video_tab(frames, landmarks_list, video_data)

    with tabs[3]:
        _render_evaluation_tab(frames, landmarks_list, elbow_angles)


def _render_analysis_results(
    frames: List[np.ndarray],
    landmarks_list: List[Optional[Dict[str, Dict[str, float]]]],
    elbow_angles: List[Optional[float]]
) -> None:
    """解析結果を表示する
    
    Args:
        frames: フレームのリスト
        landmarks_list: 各フレームのランドマーク辞書のリスト
        elbow_angles: 各フレームの肘の角度のリスト
    """
    st.subheader("解析結果")
    
    # 解析種類選択
    analysis_types = {
        "肘角度": "right_elbow",
        "肩角度": "right_shoulder",
        "膝角度": "right_knee",
        "腰角度": "right_hip",
        "体幹傾き": "torso_axis",
        "肩ライン角度": "shoulder_line",
        "骨盤ライン角度": "hip_line",
        "手首速度": "wrist_velocity",
        "投球フェーズ": "pitching_phases",
    }
    
    selected_analysis_names = st.multiselect(
        "解析種類を選択",
        options=list(analysis_types.keys()),
        default=["肘角度"],
        key="analysis_type_select"
    )
    
    # 解析データを計算
    analysis_data = {}
    phases = None
    phase_summary = None
    
    # 各種角度を計算
    if any(name in ["肘角度", "肩角度", "膝角度", "腰角度", "体幹傾き", "肩ライン角度", "骨盤ライン角度"] for name in selected_analysis_names):
        all_angles = calculate_all_angles_from_landmarks(landmarks_list)
        angle_mapping = {
            "肘角度": "right_elbow",
            "肩角度": "right_shoulder",
            "膝角度": "right_knee",
            "腰角度": "right_hip",
            "体幹傾き": "torso_axis",
            "肩ライン角度": "shoulder_line",
            "骨盤ライン角度": "hip_line",
        }
        for name, key in angle_mapping.items():
            if name in selected_analysis_names:
                analysis_data[name] = all_angles[key]
    
    # 手首速度を計算
    if "手首速度" in selected_analysis_names:
        wrist_velocities = calculate_wrist_velocity(landmarks_list)
        analysis_data["手首速度"] = wrist_velocities
    
    # 投球フェーズを推定
    if "投球フェーズ" in selected_analysis_names:
        wrist_velocities = calculate_wrist_velocity(landmarks_list)
        phases = detect_pitching_phases(landmarks_list, elbow_angles, wrist_velocities)
        if phases:
            all_angles = calculate_all_angles_from_landmarks(landmarks_list)
            phase_summary = calculate_phase_summary(phases, all_angles, wrist_velocities)
    
    # 表示形式選択
    display_mode = st.radio(
        "解析結果の表示形式",
        ["フレーム画像上に骨格描画", "解析線のみ動画"],
        key="display_mode_radio"
    )
    
    # フレーム選択
    frame_idx = st.slider(
        "フレームを選択",
        min_value=0,
        max_value=len(frames) - 1,
        value=0
    )
    
    # フレームビューアー表示
    if display_mode == "フレーム画像上に骨格描画":
        # 現在フレームの各種角度を取得
        torso_angle_val = None
        shoulder_line_angle_val = None
        hip_line_angle_val = None

        if "体幹傾き" in analysis_data:
            vals = analysis_data["体幹傾き"]
            if 0 <= frame_idx < len(vals):
                torso_angle_val = vals[frame_idx]

        if "肩ライン角度" in analysis_data:
            vals = analysis_data["肩ライン角度"]
            if 0 <= frame_idx < len(vals):
                shoulder_line_angle_val = vals[frame_idx]

        if "骨盤ライン角度" in analysis_data:
            vals = analysis_data["骨盤ライン角度"]
            if 0 <= frame_idx < len(vals):
                hip_line_angle_val = vals[frame_idx]

        _render_frame_viewer(
            frames,
            frame_idx,
            landmarks_list[frame_idx],
            elbow_angles[frame_idx] if "肘角度" in selected_analysis_names else None,
            torso_angle=torso_angle_val,
            shoulder_line_angle=shoulder_line_angle_val,
            hip_line_angle=hip_line_angle_val,
        )
    
    # 複数解析結果のグラフ表示
    if analysis_data:
        _render_multi_analysis_charts(analysis_data, selected_analysis_names)
    
    # 投球フェーズサマリー表示
    if phase_summary:
        _render_phase_summary(phase_summary)
    
    # 骨格描画済み動画を生成・表示
    st.subheader("骨格描画済み動画")
    
    # 表示モード選択
    display_mode_video = st.radio(
        "表示モード",
        ["通常骨格", "残像トレイル"],
        key="display_mode_video_radio",
        horizontal=True
    )
    trail_mode = display_mode_video == "残像トレイル"
    
    # 残像フレーム数スライダー（残像モードの場合のみ表示）
    max_trail_history = 20
    trail_decay = 0.92
    if trail_mode:
        max_trail_history = st.slider(
            "残像フレーム数",
            min_value=5,
            max_value=30,
            value=20,
            key="trail_history_slider"
        )
        trail_decay = st.slider(
            "残像の濃さ（大きいほど濃い）",
            min_value=0.85,
            max_value=0.98,
            value=0.92,
            step=0.01,
            key="trail_decay_slider"
        )
    
    # セッション状態の初期化
    video_overlay_key = "annotated_video_overlay_path"  # 元動画+解析線
    video_skeleton_key = "annotated_video_skeleton_path"  # 解析線のみ
    selected_video_type_key = "selected_video_type"  # 選択中の動画タイプ
    show_video_key = "show_annotated_video"
    
    if show_video_key not in st.session_state:
        st.session_state[show_video_key] = False
    if selected_video_type_key not in st.session_state:
        st.session_state[selected_video_type_key] = "overlay"
    
    # 動画タイプ選択ボタン
    col1, col2 = st.columns(2)
    with col1:
        if st.button("元動画+解析線を生成", key="generate_overlay_video"):
            with st.spinner("元動画に解析線を重ねた動画を生成しています..."):
                overlay_video_path = create_annotated_video(
                    frames,
                    landmarks_list,
                    background_color=None,
                    trail_mode=trail_mode,
                    max_trail_history=max_trail_history,
                    trail_decay=trail_decay,
                )
                st.session_state[video_overlay_key] = overlay_video_path
                if overlay_video_path and os.path.exists(overlay_video_path):
                    file_size = os.path.getsize(overlay_video_path)
                    st.success(f"生成完了: {file_size / (1024*1024):.2f} MB")
                else:
                    st.error("動画の生成に失敗しました")
    
    with col2:
        # 背景色選択
        bg_color = st.radio("解析線のみ動画の背景色", ["黒", "白"], key="bg_color_radio", horizontal=True)
        bg_color_value = "black" if bg_color == "黒" else "white"
        
        if st.button("解析線のみを生成", key="generate_skeleton_video"):
            with st.spinner(f"解析線のみの動画を生成しています（背景: {bg_color}）..."):
                skeleton_video_path = create_annotated_video(
                    frames,
                    landmarks_list,
                    background_color=bg_color_value,
                    trail_mode=trail_mode,
                    max_trail_history=max_trail_history,
                    trail_decay=trail_decay,
                )
                st.session_state[video_skeleton_key] = skeleton_video_path
                if skeleton_video_path and os.path.exists(skeleton_video_path):
                    file_size = os.path.getsize(skeleton_video_path)
                    st.success(f"生成完了: {file_size / (1024*1024):.2f} MB")
                else:
                    st.error("動画の生成に失敗しました")
    
    # 動画タイプ選択
    video_options = []
    if st.session_state.get(video_overlay_key) and os.path.exists(st.session_state[video_overlay_key]):
        video_options.append("元動画+解析線")
    if st.session_state.get(video_skeleton_key) and os.path.exists(st.session_state[video_skeleton_key]):
        video_options.append("解析線のみ")
    
    if video_options:
        selected_type = st.radio("表示する動画を選択", video_options, key="video_type_radio")
        st.session_state[selected_video_type_key] = "overlay" if selected_type == "元動画+解析線" else "skeleton"
        
        # 選択された動画のパスを取得
        if st.session_state[selected_video_type_key] == "overlay":
            current_video_path = st.session_state.get(video_overlay_key)
        else:
            current_video_path = st.session_state.get(video_skeleton_key)
        
        # 動画表示・ダウンロードボタン
        if current_video_path and os.path.exists(current_video_path):
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                if st.button("動画を表示", key="show_video_button"):
                    st.session_state[show_video_key] = True
                    st.rerun()
            
            with col2:
                if st.button("動画を非表示", key="hide_video_button"):
                    st.session_state[show_video_key] = False
                    st.rerun()
            
            with col3:
                # ダウンロードボタン
                with open(current_video_path, "rb") as video_file:
                    video_bytes = video_file.read()
                    st.download_button(
                        label="📥 動画をダウンロード",
                        data=video_bytes,
                        file_name=f"pitching_analysis_{selected_type.replace('+', '_').replace(' ', '_')}.mp4",
                        mime="video/mp4",
                        key="download_video_button"
                    )
            
            # セッション状態に基づいて動画を表示
            if st.session_state[show_video_key]:
                st.video(current_video_path)
                st.info(f"動画を再生中: {current_video_path}")
    else:
        st.info("上記のボタンで動画を生成してください")



def _process_video_analysis(
    uploaded_file: Any,
    progress_container: Any = None
) -> Tuple[
    Optional[List[np.ndarray]],
    Optional[List[Optional[Dict[str, Dict[str, float]]]]],
    Optional[List[Optional[float]]]
]:
    """動画解析を実行する（Cloud Run 対応：進行状況表示）
    
    Args:
        uploaded_file: アップロードされたファイルオブジェクト
        progress_container: 進行状況表示用のコンテナ
    
    Returns:
        (フレームリスト, ランドマークリスト, 角度リスト)のタプル
    """
    if progress_container:
        progress_container.info("📹 動画を読み込んでいます...")
    
    frames = load_video_frames(uploaded_file, progress_container=progress_container)
    
    if frames is None or len(frames) == 0:
        if progress_container:
            progress_container.error("❌ 動画の読み込みに失敗しました")
        return None, None, None
    
    if progress_container:
        progress_container.info("🤖 姿勢推定を実行しています...")
    
    landmarks_list = process_video_frames(frames, progress_container=progress_container)
    
    if progress_container:
        progress_container.info("📐 角度を計算しています...")
    
    elbow_angles = calculate_elbow_angles_from_landmarks(landmarks_list)
    
    if progress_container:
        progress_container.success(f"✅ 解析完了！{len(frames)} フレームを解析しました")
    
    return frames, landmarks_list, elbow_angles
def run_normal_analysis() -> None:
    """通常解析モードのUIと処理"""
    # 動画アップロードセクション
    st.subheader("📤 動画をアップロード")
    
    # 解析中はアップロードを無効化
    uploaded_file = None
    if not st.session_state["is_analyzing"]:
        uploaded_file = _render_video_upload()
    
    # アップロードされたファイルをセッション状態に保存
    if uploaded_file is not None:
        uploaded_file.seek(0)
        st.session_state["uploaded_file_name"] = uploaded_file.name
        st.session_state["uploaded_file_bytes"] = uploaded_file.read()
        uploaded_file.seek(0)  # 読み取り位置をリセット
    
    # 解析中でない場合のみ解析ボタンを表示
    if not st.session_state["is_analyzing"]:
        if st.session_state["uploaded_file_name"] is not None:
            col1, col2 = st.columns([1, 4])
            with col1:
                analyze_button = st.button("🚀 解析を開始", type="primary", use_container_width=True)
            
            with col2:
                st.info(f"📁 選択されたファイル: {st.session_state['uploaded_file_name']}")
            
            if analyze_button:
                # 解析状態を開始
                st.session_state["is_analyzing"] = True
                st.rerun()
    
    # 解析中の処理
    if st.session_state["is_analyzing"]:
        # 解析中表示
        st.markdown("---")
        progress_section = st.container()
        
        with progress_section:
            st.subheader("🔄 解析中...")
            st.info("⏳ 動画の解析を実行しています。しばらくお待ちください。")
            st.warning("⚠️ 解析中はこのページを閉じないでください。")
            
            # 進行状況表示用のコンテナ
            progress_container = st.container()
            
            # アップロードされたファイルを再構築
            if st.session_state["uploaded_file_bytes"] is not None:
                # BytesIO オブジェクトを作成
                uploaded_file_obj = io.BytesIO(st.session_state["uploaded_file_bytes"])
                uploaded_file_obj.name = st.session_state["uploaded_file_name"]
                
                try:
                    # 解析を実行（長時間処理のため、エラーハンドリングを追加）
                    with progress_container:
                        frames, landmarks_list, elbow_angles = _process_video_analysis(
                            uploaded_file_obj,
                            progress_container=progress_container
                        )
                    
                    # 解析完了後の処理
                    if frames is not None and landmarks_list is not None and elbow_angles is not None:
                        # 解析結果をセッション状態に保存
                        analysis_data = {
                            "name": st.session_state["uploaded_file_name"],
                            "frames": frames,
                            "landmarks": landmarks_list,
                            "elbow_angles": elbow_angles,
                            "annotated_overlay_path": None,
                            "annotated_skeleton_path": None,
                        }
                        st.session_state["analysis_results"].append(analysis_data)
                        st.session_state["current_analysis_index"] = len(st.session_state["analysis_results"]) - 1
                        
                        # 解析状態を終了
                        st.session_state["is_analyzing"] = False
                        
                        # 解析完了メッセージ
                        st.success("✅ 解析が完了しました！結果は下に表示されます。")
                        
                        # 画面を更新（解析完了後のみ）
                        st.rerun()
                    else:
                        # 解析失敗
                        st.session_state["is_analyzing"] = False
                        st.error("❌ 解析に失敗しました。もう一度お試しください。")
                        st.rerun()
                except Exception as e:
                    # 予期しないエラー
                    st.session_state["is_analyzing"] = False
                    st.error(f"❌ 解析中にエラーが発生しました: {str(e)}")
                    st.rerun()
            else:
                # ファイルが存在しない場合
                st.session_state["is_analyzing"] = False
                st.error("❌ アップロードされたファイルが見つかりません。")
                st.rerun()
    
    st.markdown("---")
    
    # 解析結果を画面下に追加表示（画面遷移なし）
    if st.session_state["analysis_results"]:
        st.subheader("📊 解析結果")
        
        # 解析結果の選択（複数の解析結果がある場合）
        if len(st.session_state["analysis_results"]) > 1:
            result_names = [f"{i+1}. {result['name']}" for i, result in enumerate(st.session_state["analysis_results"])]
            selected_idx = st.selectbox(
                "表示する解析結果を選択",
                options=range(len(result_names)),
                format_func=lambda x: result_names[x],
                index=st.session_state["current_analysis_index"] if st.session_state["current_analysis_index"] >= 0 else 0
            )
            st.session_state["current_analysis_index"] = selected_idx
        else:
            st.session_state["current_analysis_index"] = 0
        
        # 現在の解析結果を取得
        if st.session_state["current_analysis_index"] >= 0:
            current_result = st.session_state["analysis_results"][st.session_state["current_analysis_index"]]
            frames = current_result["frames"]
            landmarks_list = current_result["landmarks"]
            elbow_angles = current_result["elbow_angles"]
            
            # 解析結果を表示（タブ形式）
            tabs = st.tabs(["📊 解析結果", "📈 グラフ", "🎬 解析動画", "⭐ 評価"])
            
            with tabs[0]:
                _render_analysis_tab(frames, landmarks_list, elbow_angles)
            
            with tabs[1]:
                _render_graph_tab(frames, landmarks_list, elbow_angles)
            
            with tabs[2]:
                _render_video_tab(frames, landmarks_list, current_result)
            
            with tabs[3]:
                _render_evaluation_tab(frames, landmarks_list, elbow_angles)
    elif not st.session_state["is_analyzing"]:
        st.info("💡 動画をアップロードして解析を開始してください")


def run_strobe_ui() -> None:
    """ストロボ解析UI（3種類のストロボ生成機能を集約）"""
    # 共通設定セクション
    st.subheader("📸 ストロボ解析設定")
    
    # 表示モード選択
    display_mode = st.radio(
        "表示モード",
        ["normal", "skeleton"],
        index=0,
        horizontal=True,
        key="strobe_display_mode",
        help="normal: 元動画フレーム + 骨格描画 / skeleton: 骨格線のみ（黒背景）"
    )
    
    # フレーム抽出間隔の設定
    step = st.slider(
        "フレーム抽出間隔",
        min_value=1,
        max_value=20,
        value=5,
        help="この値ごとにフレームを抽出して連結します（小さいほど多くのフレームが含まれます）",
        key="strobe_step_slider"
    )
    
    st.markdown("---")
    
    # 1. 通常ストロボ画像生成セクション
    st.subheader("📸 ストロボ解析画像生成")
    
    # ストロボ画像生成ボタン
    if st.button("📸 ストロボ解析画像生成", type="secondary", use_container_width=True, key="generate_strobe_button"):
        # セッション状態から動画ファイルを取得
        if st.session_state.get("uploaded_file_bytes") is not None:
            # 一時ファイルとして保存
            tmp_dir = "/tmp"
            os.makedirs(tmp_dir, exist_ok=True)
            tmp_video_path = tempfile.mktemp(suffix=".mp4", dir=tmp_dir)
            
            try:
                # セッション状態のバイト列を一時ファイルに書き込み
                with open(tmp_video_path, "wb") as tmp_file:
                    tmp_file.write(st.session_state["uploaded_file_bytes"])
                
                # ストロボ画像を生成
                with st.spinner("ストロボ解析画像を生成中..."):
                    strobe_image = generate_strobe_image(tmp_video_path, step=step, mode=display_mode)
                    
                    if strobe_image is not None:
                        # セッション状態に保存（numpy.ndarray形式、既にリサイズ済み）
                        st.session_state["strobe_image"] = strobe_image
                        # モードも保存
                        st.session_state["strobe_mode"] = display_mode
                        
                        # ダウンロード用にもリサイズ処理を適用（念のため）
                        strobe_image_for_download = _resize_image_if_needed(strobe_image, max_pixels=150000000)
                        
                        # PNG形式のバイト列に変換して保存（ダウンロード用）
                        success, buffer = cv2.imencode('.png', strobe_image_for_download)
                        if success:
                            st.session_state["strobe_image_bytes"] = buffer.tobytes()
                        else:
                            st.session_state["strobe_image_bytes"] = None
                            st.warning("PNG形式への変換に失敗しました")
                        
                        st.success("✅ ストロボ解析画像の生成が完了しました！")
                    else:
                        st.error("ストロボ解析画像の生成に失敗しました")
            except Exception as e:
                st.error(f"ストロボ画像生成中にエラーが発生しました: {e}")
            finally:
                # 一時ファイルを削除
                if os.path.exists(tmp_video_path):
                    try:
                        os.remove(tmp_video_path)
                    except Exception:
                        pass
        else:
            st.error("動画ファイルが見つかりません。先に「解析」タブで動画をアップロードして解析を実行してください。")
    
    # 生成されたストロボ画像を表示
    if st.session_state.get("strobe_image") is not None:
        strobe_image = st.session_state["strobe_image"]
        
        # 表示用にリサイズ（念のため二重チェック）
        strobe_image_display = _resize_image_if_needed(strobe_image, max_pixels=150000000)
        
        # RGB形式に変換して表示
        strobe_image_rgb = cv2.cvtColor(strobe_image_display, cv2.COLOR_BGR2RGB)
        st.image(strobe_image_rgb, use_container_width=True, caption="ストロボ解析画像")
        
        # 画像情報を表示
        height, width = strobe_image_display.shape[:2]
        st.info(f"画像サイズ: {width} x {height} ピクセル")
        
        # ダウンロードボタン
        if st.session_state.get("strobe_image_bytes") is not None:
            # モードに応じたファイル名を生成
            strobe_mode = st.session_state.get("strobe_mode", "normal")
            file_name = f"strobe_{strobe_mode}.png"
            
            st.download_button(
                label="📥 ストロボ画像をダウンロード",
                data=st.session_state["strobe_image_bytes"],
                file_name=file_name,
                mime="image/png",
                use_container_width=True,
                key="download_strobe_button"
            )
        else:
            st.warning("ダウンロード用の画像データが準備できていません")
    
    # 2. フェーズ別ストロボ画像生成セクション
    st.markdown("---")
    st.subheader("📸 フェーズ別ストロボ画像生成")
    
    # フェーズ別ストロボ生成ボタン
    if st.button("📸 フェーズ別ストロボ生成", type="secondary", use_container_width=True, key="generate_phase_strobes_button"):
        # セッション状態から動画ファイルを取得
        if st.session_state.get("uploaded_file_bytes") is not None:
            # 一時ファイルとして保存
            tmp_dir = "/tmp"
            os.makedirs(tmp_dir, exist_ok=True)
            tmp_video_path = tempfile.mktemp(suffix=".mp4", dir=tmp_dir)
            
            try:
                # セッション状態のバイト列を一時ファイルに書き込み
                with open(tmp_video_path, "wb") as tmp_file:
                    tmp_file.write(st.session_state["uploaded_file_bytes"])
                
                # フェーズ別ストロボ画像を生成
                with st.spinner("フェーズ別ストロボ画像を生成中..."):
                    phase_strobes = generate_phase_strobes(tmp_video_path, step=step, mode=display_mode)
                    
                    if phase_strobes:
                        # セッション状態に保存
                        st.session_state["phase_strobes"] = phase_strobes
                        st.session_state["phase_strobes_mode"] = display_mode
                        
                        # 各フェーズの画像をPNG形式のバイト列に変換
                        phase_strobes_bytes = {}
                        for phase_name, phase_image in phase_strobes.items():
                            if phase_image is not None:
                                # リサイズ処理を適用
                                phase_image_resized = _resize_image_if_needed(phase_image, max_pixels=150000000)
                                success, buffer = cv2.imencode('.png', phase_image_resized)
                                if success:
                                    phase_strobes_bytes[phase_name] = buffer.tobytes()
                                else:
                                    phase_strobes_bytes[phase_name] = None
                            else:
                                phase_strobes_bytes[phase_name] = None
                        
                        st.session_state["phase_strobes_bytes"] = phase_strobes_bytes
                        st.success("✅ フェーズ別ストロボ画像の生成が完了しました！")
                    else:
                        st.error("フェーズ別ストロボ画像の生成に失敗しました")
            except Exception as e:
                st.error(f"フェーズ別ストロボ画像生成中にエラーが発生しました: {e}")
            finally:
                # 一時ファイルを削除
                if os.path.exists(tmp_video_path):
                    try:
                        os.remove(tmp_video_path)
                    except Exception:
                        pass
        else:
            st.error("動画ファイルが見つかりません。先に「解析」タブで動画をアップロードして解析を実行してください。")
    
    # 生成されたフェーズ別ストロボ画像を表示
    if st.session_state.get("phase_strobes") is not None:
        phase_strobes = st.session_state["phase_strobes"]
        phase_strobes_bytes = st.session_state.get("phase_strobes_bytes", {})
        
        # フェーズ名の日本語対応
        phase_names_jp = {
            "windup": "ウィンドアップ",
            "stride": "ストライド",
            "foot_plant": "フットプラント",
            "acceleration": "加速",
            "follow_through": "フォロースルー"
        }
        
        st.markdown("### フェーズ別ストロボ画像")
        
        # 各フェーズごとに表示
        for phase_name in ["windup", "stride", "foot_plant", "acceleration", "follow_through"]:
            phase_image = phase_strobes.get(phase_name)
            
            if phase_image is not None:
                phase_name_jp = phase_names_jp.get(phase_name, phase_name)
                st.markdown(f"#### {phase_name_jp}")
                
                # 表示用にリサイズ（念のため二重チェック）
                phase_image_display = _resize_image_if_needed(phase_image, max_pixels=150000000)
                
                # RGB形式に変換して表示
                phase_image_rgb = cv2.cvtColor(phase_image_display, cv2.COLOR_BGR2RGB)
                st.image(phase_image_rgb, use_container_width=True, caption=f"{phase_name_jp}フェーズ")
                
                # 画像情報を表示
                height, width = phase_image_display.shape[:2]
                st.info(f"画像サイズ: {width} x {height} ピクセル")
                
                # ダウンロードボタン
                if phase_name in phase_strobes_bytes and phase_strobes_bytes[phase_name] is not None:
                    file_name = f"strobe_phase_{phase_name}.png"
                    st.download_button(
                        label=f"📥 {phase_name_jp}フェーズ画像をダウンロード",
                        data=phase_strobes_bytes[phase_name],
                        file_name=file_name,
                        mime="image/png",
                        use_container_width=True,
                        key=f"download_phase_strobe_{phase_name}"
                    )
                else:
                    st.warning(f"{phase_name_jp}フェーズのダウンロード用データが準備できていません")
                
                st.markdown("---")
            else:
                phase_name_jp = phase_names_jp.get(phase_name, phase_name)
                st.warning(f"{phase_name_jp}フェーズの画像が生成されませんでした（フレームが不足している可能性があります）")
    
    # 3. フェーズ代表ストロボ画像生成セクション
    st.markdown("---")
    st.subheader("📸 フェーズ代表ストロボ画像生成")
    
    # フェーズ代表ストロボ生成ボタン
    if st.button("📸 フェーズ代表ストロボ生成", type="secondary", use_container_width=True, key="generate_phase_keyframe_strobe_button"):
        # セッション状態から動画ファイルを取得
        if st.session_state.get("uploaded_file_bytes") is not None:
            # 一時ファイルとして保存
            tmp_dir = "/tmp"
            os.makedirs(tmp_dir, exist_ok=True)
            tmp_video_path = tempfile.mktemp(suffix=".mp4", dir=tmp_dir)
            
            try:
                # セッション状態のバイト列を一時ファイルに書き込み
                with open(tmp_video_path, "wb") as tmp_file:
                    tmp_file.write(st.session_state["uploaded_file_bytes"])
                
                # フェーズ代表ストロボ画像を生成
                with st.spinner("フェーズ代表ストロボ画像を生成中..."):
                    keyframe_strobe = generate_phase_keyframe_strobe(tmp_video_path, mode=display_mode)
                    
                    if keyframe_strobe is not None:
                        # セッション状態に保存
                        st.session_state["phase_keyframe_strobe"] = keyframe_strobe
                        st.session_state["phase_keyframe_strobe_mode"] = display_mode
                        
                        # PNG形式のバイト列に変換して保存（ダウンロード用）
                        keyframe_strobe_resized = _resize_image_if_needed(keyframe_strobe, max_pixels=150000000)
                        success, buffer = cv2.imencode('.png', keyframe_strobe_resized)
                        if success:
                            st.session_state["phase_keyframe_strobe_bytes"] = buffer.tobytes()
                        else:
                            st.session_state["phase_keyframe_strobe_bytes"] = None
                            st.warning("PNG形式への変換に失敗しました")
                        
                        st.success("✅ フェーズ代表ストロボ画像の生成が完了しました！")
                    else:
                        st.error("フェーズ代表ストロボ画像の生成に失敗しました")
            except Exception as e:
                st.error(f"フェーズ代表ストロボ画像生成中にエラーが発生しました: {e}")
            finally:
                # 一時ファイルを削除
                if os.path.exists(tmp_video_path):
                    try:
                        os.remove(tmp_video_path)
                    except Exception:
                        pass
        else:
            st.error("動画ファイルが見つかりません。先に「解析」タブで動画をアップロードして解析を実行してください。")
    
    # 生成されたフェーズ代表ストロボ画像を表示
    if st.session_state.get("phase_keyframe_strobe") is not None:
        keyframe_strobe = st.session_state["phase_keyframe_strobe"]
        
        st.markdown("### フェーズ代表ストロボ画像")
        
        # 表示用にリサイズ（念のため二重チェック）
        keyframe_strobe_display = _resize_image_if_needed(keyframe_strobe, max_pixels=150000000)
        
        # RGB形式に変換して表示
        keyframe_strobe_rgb = cv2.cvtColor(keyframe_strobe_display, cv2.COLOR_BGR2RGB)
        st.image(keyframe_strobe_rgb, use_container_width=True, caption="フェーズ代表ストロボ画像（windup → stride → plant → acceleration → follow）")
        
        # 画像情報を表示
        height, width = keyframe_strobe_display.shape[:2]
        st.info(f"画像サイズ: {width} x {height} ピクセル")
        
        # ダウンロードボタン
        if st.session_state.get("phase_keyframe_strobe_bytes") is not None:
            st.download_button(
                label="📥 フェーズ代表ストロボ画像をダウンロード",
                data=st.session_state["phase_keyframe_strobe_bytes"],
                file_name="phase_keyframes.png",
                mime="image/png",
                use_container_width=True,
                key="download_phase_keyframe_strobe_button"
            )
        else:
            st.warning("ダウンロード用の画像データが準備できていません")


def _render_comparison_mode() -> None:
    """
    フォーム比較モード：2つの動画を並列表示し、MediaPipe Pose の骨格を重ねて同時再生
    
    左: お手本動画アップロード
    右: 自分の動画アップロード
    """
    st.title("📊 フォーム比較モード")
    st.markdown("---")
    
    # セッション状態の初期化（比較モード用）
    if "comparison_reference_video" not in st.session_state:
        st.session_state["comparison_reference_video"] = None
    if "comparison_user_video" not in st.session_state:
        st.session_state["comparison_user_video"] = None
    if "comparison_reference_processed" not in st.session_state:
        st.session_state["comparison_reference_processed"] = None
    if "comparison_user_processed" not in st.session_state:
        st.session_state["comparison_user_processed"] = None
    if "comparison_processing" not in st.session_state:
        st.session_state["comparison_processing"] = False
    if "comparison_reference_angles" not in st.session_state:
        st.session_state["comparison_reference_angles"] = None
    if "comparison_user_angles" not in st.session_state:
        st.session_state["comparison_user_angles"] = None
    if "comparison_angles_processing" not in st.session_state:
        st.session_state["comparison_angles_processing"] = False
    if "comparison_reference_peak" not in st.session_state:
        st.session_state["comparison_reference_peak"] = None
    if "comparison_user_peak" not in st.session_state:
        st.session_state["comparison_user_peak"] = None
    if "comparison_use_sync" not in st.session_state:
        st.session_state["comparison_use_sync"] = True
    
    # 2カラムレイアウト
    col1, col2 = st.columns(2)
    
    # 左カラム：お手本動画アップロード
    with col1:
        st.subheader("📹 お手本動画")
        reference_file = st.file_uploader(
            "お手本動画をアップロード",
            type=["mp4", "avi", "mov", "mkv"],
            key="comparison_reference_uploader"
        )
        
        if reference_file is not None:
            # 一時ファイルに保存
            reference_path = _save_uploaded_file_to_temp(reference_file)
            if reference_path:
                st.session_state["comparison_reference_video"] = reference_path
                st.success(f"✅ お手本動画をアップロードしました: {reference_file.name}")
    
    # 右カラム：自分の動画アップロード
    with col2:
        st.subheader("📹 自分の動画")
        user_file = st.file_uploader(
            "自分の動画をアップロード",
            type=["mp4", "avi", "mov", "mkv"],
            key="comparison_user_uploader"
        )
        
        if user_file is not None:
            # 一時ファイルに保存
            user_path = _save_uploaded_file_to_temp(user_file)
            if user_path:
                st.session_state["comparison_user_video"] = user_path
                st.success(f"✅ 自分の動画をアップロードしました: {user_file.name}")
    
    st.markdown("---")
    
    # 両方アップロードされた場合のみ処理を実行
    if (st.session_state["comparison_reference_video"] is not None and 
        st.session_state["comparison_user_video"] is not None):
        
        # 処理ボタン
        if not st.session_state["comparison_processing"]:
            if st.button("🚀 骨格描画動画を生成", type="primary", use_container_width=True):
                st.session_state["comparison_processing"] = True
                st.session_state["comparison_reference_processed"] = None
                st.session_state["comparison_user_processed"] = None
                st.rerun()
        
        # 処理中の表示
        if st.session_state["comparison_processing"]:
            st.info("⏳ 骨格描画動画を生成中...")
            
            # お手本動画の処理
            if st.session_state["comparison_reference_processed"] is None:
                with st.spinner("お手本動画を処理中..."):
                    reference_processed = process_video_with_pose(
                        st.session_state["comparison_reference_video"]
                    )
                    if reference_processed:
                        st.session_state["comparison_reference_processed"] = reference_processed
                    else:
                        st.error("お手本動画の処理に失敗しました")
                        st.session_state["comparison_processing"] = False
                        st.rerun()
            
            # 自分の動画の処理
            if (st.session_state["comparison_reference_processed"] is not None and
                st.session_state["comparison_user_processed"] is None):
                with st.spinner("自分の動画を処理中..."):
                    user_processed = process_video_with_pose(
                        st.session_state["comparison_user_video"]
                    )
                    if user_processed:
                        st.session_state["comparison_user_processed"] = user_processed
                    else:
                        st.error("自分の動画の処理に失敗しました")
                        st.session_state["comparison_processing"] = False
                        st.rerun()
            
            # 両方の処理が完了した場合
            if (st.session_state["comparison_reference_processed"] is not None and
                st.session_state["comparison_user_processed"] is not None):
                st.session_state["comparison_processing"] = False
                st.success("✅ 骨格描画動画の生成が完了しました！")
                st.rerun()
        
        # 処理完了後、動画を表示
        if (st.session_state["comparison_reference_processed"] is not None and
            st.session_state["comparison_user_processed"] is not None):
            
            st.markdown("---")
            st.subheader("🎬 骨格付き動画")
            
            # 2カラムで動画を並列表示
            video_col1, video_col2 = st.columns(2)
            
            with video_col1:
                st.markdown("**📹 お手本動画（骨格付き）**")
                with open(st.session_state["comparison_reference_processed"], "rb") as video_file:
                    video_bytes = video_file.read()
                    st.video(video_bytes)
            
            with video_col2:
                st.markdown("**📹 自分の動画（骨格付き）**")
                with open(st.session_state["comparison_user_processed"], "rb") as video_file:
                    video_bytes = video_file.read()
                    st.video(video_bytes)
            
            # 角度比較機能（STEP2）
            st.markdown("---")
            
            # 角度抽出ボタン
            if not st.session_state["comparison_angles_processing"]:
                if st.button("📐 角度を抽出・比較", type="primary", use_container_width=True):
                    st.session_state["comparison_angles_processing"] = True
                    st.session_state["comparison_reference_angles"] = None
                    st.session_state["comparison_user_angles"] = None
                    st.rerun()
            
            # 角度抽出処理中
            if st.session_state["comparison_angles_processing"]:
                st.info("⏳ 角度を抽出中...")
                
                # お手本動画の角度抽出
                if st.session_state["comparison_reference_angles"] is None:
                    with st.spinner("お手本動画から角度を抽出中..."):
                        try:
                            reference_angles = extract_angles_from_video(
                                st.session_state["comparison_reference_video"]
                            )
                            st.session_state["comparison_reference_angles"] = reference_angles
                        except Exception as e:
                            st.error(f"お手本動画の角度抽出に失敗しました: {e}")
                            st.session_state["comparison_angles_processing"] = False
                            st.rerun()
                
                # 自分の動画の角度抽出
                if (st.session_state["comparison_reference_angles"] is not None and
                    st.session_state["comparison_user_angles"] is None):
                    with st.spinner("自分の動画から角度を抽出中..."):
                        try:
                            user_angles = extract_angles_from_video(
                                st.session_state["comparison_user_video"]
                            )
                            st.session_state["comparison_user_angles"] = user_angles
                        except Exception as e:
                            st.error(f"自分の動画の角度抽出に失敗しました: {e}")
                            st.session_state["comparison_angles_processing"] = False
                            st.rerun()
                
                # 両方の角度抽出が完了した場合、ピーク検出を実行（STEP3）
                if (st.session_state["comparison_reference_angles"] is not None and
                    st.session_state["comparison_user_angles"] is not None):
                    
                    # ピーク検出がまだ実行されていない場合
                    if st.session_state["comparison_reference_peak"] is None:
                        with st.spinner("お手本動画の足上げトップを検出中..."):
                            try:
                                reference_peak = detect_peak_knee_frame(
                                    st.session_state["comparison_reference_video"]
                                )
                                st.session_state["comparison_reference_peak"] = reference_peak
                            except Exception as e:
                                st.warning(f"お手本動画のピーク検出に失敗しました: {e}")
                                st.session_state["comparison_reference_peak"] = None
                    
                    # 自分の動画のピーク検出
                    if (st.session_state["comparison_reference_peak"] is not None and
                        st.session_state["comparison_user_peak"] is None):
                        with st.spinner("自分の動画の足上げトップを検出中..."):
                            try:
                                user_peak = detect_peak_knee_frame(
                                    st.session_state["comparison_user_video"]
                                )
                                st.session_state["comparison_user_peak"] = user_peak
                            except Exception as e:
                                st.warning(f"自分の動画のピーク検出に失敗しました: {e}")
                                st.session_state["comparison_user_peak"] = None
                    
                    # ピーク検出が完了した場合
                    if (st.session_state["comparison_reference_peak"] is not None and
                        st.session_state["comparison_user_peak"] is not None):
                        st.session_state["comparison_angles_processing"] = False
                        st.success("✅ 角度抽出とピーク検出が完了しました！")
                        st.rerun()
                    elif (st.session_state["comparison_reference_peak"] is None or
                          st.session_state["comparison_user_peak"] is None):
                        # ピーク検出に失敗した場合でも処理を続行
                        st.session_state["comparison_angles_processing"] = False
                        st.warning("⚠️ ピーク検出に失敗しましたが、角度比較は続行します")
                        st.rerun()
            
            # 角度比較結果を表示
            if (st.session_state["comparison_reference_angles"] is not None and
                st.session_state["comparison_user_angles"] is not None):
                _render_angle_comparison(
                    st.session_state["comparison_reference_angles"],
                    st.session_state["comparison_user_angles"],
                    st.session_state["comparison_reference_peak"],
                    st.session_state["comparison_user_peak"]
                )
    
    else:
        # 両方アップロードされていない場合
        if (st.session_state["comparison_reference_video"] is None or
            st.session_state["comparison_user_video"] is None):
            st.info("💡 左右両方の動画をアップロードしてください")


def _render_angle_comparison(
    reference_angles: Dict[str, List[Optional[float]]],
    user_angles: Dict[str, List[Optional[float]]],
    reference_peak: Optional[int] = None,
    user_peak: Optional[int] = None
) -> None:
    """
    角度比較グラフと差分を表示する（STEP3: 自動同期機能付き）
    
    Args:
        reference_angles: お手本動画の角度データ
        user_angles: 自分の動画の角度データ
        reference_peak: お手本動画のピークフレーム番号（Noneの場合は同期なし）
        user_peak: 自分の動画のピークフレーム番号（Noneの場合は同期なし）
    """
    st.markdown("---")
    st.subheader("📊 関節角度比較")
    
    # 自動同期のトグル（STEP3）
    use_sync = st.checkbox("自動同期を有効化", value=st.session_state.get("comparison_use_sync", True), key="comparison_sync_toggle")
    st.session_state["comparison_use_sync"] = use_sync
    
    # 同期情報を表示（STEP3）
    if use_sync and reference_peak is not None and user_peak is not None:
        st.markdown("---")
        st.subheader("🔄 自動同期情報")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**お手本動画ピーク:** {reference_peak} frame")
        with col2:
            st.write(f"**自分の動画ピーク:** {user_peak} frame")
        with col3:
            frame_diff = abs(reference_peak - user_peak)
            st.write(f"**補正差分:** {frame_diff} frame")
    
    # 同期処理を適用するかどうか
    if use_sync and reference_peak is not None and user_peak is not None:
        # 同期処理を適用
        reference_angles_synced = {}
        user_angles_synced = {}
        
        for angle_key in ["elbow", "knee", "hip"]:
            ref_seq = reference_angles.get(angle_key, [])
            user_seq = user_angles.get(angle_key, [])
            
            # 同期処理
            ref_synced, user_synced = sync_angle_sequences(
                ref_seq, user_seq, reference_peak, user_peak
            )
            
            reference_angles_synced[angle_key] = ref_synced
            user_angles_synced[angle_key] = user_synced
        
        # 同期後のデータを使用
        display_reference_angles = reference_angles_synced
        display_user_angles = user_angles_synced
    else:
        # 同期なしで元のデータを使用
        display_reference_angles = reference_angles
        display_user_angles = user_angles
    
    # 角度の種類と表示名
    angle_types = {
        "elbow": "右肘角度",
        "knee": "右膝角度",
        "hip": "右股関節角度"
    }
    
    # 各角度についてグラフを表示
    for angle_key, angle_name in angle_types.items():
        # 角度データを取得（同期済みまたは元のデータ）
        ref_angles = display_reference_angles.get(angle_key, [])
        user_angles_list = display_user_angles.get(angle_key, [])
        
        # Noneを除外した有効な角度のみを取得
        ref_valid = [a for a in ref_angles if a is not None]
        user_valid = [a for a in user_angles_list if a is not None]
        
        # 有効なデータがない場合はスキップ
        if not ref_valid and not user_valid:
            st.warning(f"{angle_name}: データが不足しています")
            continue
        
        # グラフを作成
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # フレーム番号（インデックス）
        ref_frames = list(range(len(ref_angles)))
        user_frames = list(range(len(user_angles_list)))
        
        # お手本動画の角度推移をプロット
        if ref_valid:
            ref_plot_frames = [i for i, a in enumerate(ref_angles) if a is not None]
            ref_plot_angles = [a for a in ref_angles if a is not None]
            ax.plot(ref_plot_frames, ref_plot_angles, label="お手本動画", 
                   color="blue", linewidth=2, marker="o", markersize=3)
        
        # 自分の動画の角度推移をプロット
        if user_valid:
            user_plot_frames = [i for i, a in enumerate(user_angles_list) if a is not None]
            user_plot_angles = [a for a in user_angles_list if a is not None]
            ax.plot(user_plot_frames, user_plot_angles, label="自分の動画", 
                   color="red", linewidth=2, marker="s", markersize=3)
        
        # グラフの設定
        ax.set_xlabel("フレーム番号", fontsize=12)
        ax.set_ylabel("角度（度）", fontsize=12)
        ax.set_title(f"{angle_name}の推移比較", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # グラフを表示
        st.pyplot(fig)
        plt.close(fig)
    
    # 角度差分を計算して表示
    st.markdown("---")
    st.subheader("📈 平均角度差分")
    
    # 3カラムで差分を表示
    diff_col1, diff_col2, diff_col3 = st.columns(3)
    
    # 右肘角度の差分
    with diff_col1:
        ref_elbow_valid = [a for a in display_reference_angles.get("elbow", []) if a is not None]
        user_elbow_valid = [a for a in display_user_angles.get("elbow", []) if a is not None]
        if ref_elbow_valid and user_elbow_valid:
            ref_mean = np.mean(ref_elbow_valid)
            user_mean = np.mean(user_elbow_valid)
            diff = abs(ref_mean - user_mean)
            st.metric("肘角度差", f"{diff:.1f}°")
        else:
            st.metric("肘角度差", "データ不足")
    
    # 右膝角度の差分
    with diff_col2:
        ref_knee_valid = [a for a in display_reference_angles.get("knee", []) if a is not None]
        user_knee_valid = [a for a in display_user_angles.get("knee", []) if a is not None]
        if ref_knee_valid and user_knee_valid:
            ref_mean = np.mean(ref_knee_valid)
            user_mean = np.mean(user_knee_valid)
            diff = abs(ref_mean - user_mean)
            st.metric("膝角度差", f"{diff:.1f}°")
        else:
            st.metric("膝角度差", "データ不足")
    
    # 右股関節角度の差分
    with diff_col3:
        ref_hip_valid = [a for a in display_reference_angles.get("hip", []) if a is not None]
        user_hip_valid = [a for a in display_user_angles.get("hip", []) if a is not None]
        if ref_hip_valid and user_hip_valid:
            ref_mean = np.mean(ref_hip_valid)
            user_mean = np.mean(user_hip_valid)
            diff = abs(ref_mean - user_mean)
            st.metric("股関節角度差", f"{diff:.1f}°")
        else:
            st.metric("股関節角度差", "データ不足")


def main() -> None:
    """メインアプリケーション（Cloud Run 対応：解析中に画面がリセットされない）"""
    # セッション状態の初期化
    if "analysis_results" not in st.session_state:
        st.session_state["analysis_results"] = []
    if "current_analysis_index" not in st.session_state:
        st.session_state["current_analysis_index"] = -1
    if "is_analyzing" not in st.session_state:
        st.session_state["is_analyzing"] = False
    if "uploaded_file_name" not in st.session_state:
        st.session_state["uploaded_file_name"] = None
    if "uploaded_file_bytes" not in st.session_state:
        st.session_state["uploaded_file_bytes"] = None
    if "strobe_image" not in st.session_state:
        st.session_state["strobe_image"] = None
    if "strobe_image_bytes" not in st.session_state:
        st.session_state["strobe_image_bytes"] = None
    if "strobe_mode" not in st.session_state:
        st.session_state["strobe_mode"] = "normal"
    if "phase_strobes" not in st.session_state:
        st.session_state["phase_strobes"] = None
    if "phase_strobes_mode" not in st.session_state:
        st.session_state["phase_strobes_mode"] = "normal"
    if "phase_strobes_bytes" not in st.session_state:
        st.session_state["phase_strobes_bytes"] = {}
    if "phase_keyframe_strobe" not in st.session_state:
        st.session_state["phase_keyframe_strobe"] = None
    if "phase_keyframe_strobe_bytes" not in st.session_state:
        st.session_state["phase_keyframe_strobe_bytes"] = None
    if "phase_keyframe_strobe_mode" not in st.session_state:
        st.session_state["phase_keyframe_strobe_mode"] = "normal"
    
    # タブ構成
    tab_analysis, tab_strobe, tab_comparison = st.tabs(["解析", "ストロボ解析", "比較モード"])
    
    with tab_analysis:
        st.title("⚾ 野球フォーム解析アプリ")
        st.markdown("---")
        run_normal_analysis()
    
    with tab_strobe:
        st.title("📸 ストロボ解析")
        st.markdown("---")
        run_strobe_ui()
    
    with tab_comparison:
        _render_comparison_mode()


if __name__ == "__main__":
    main()

