"""Streamlit UI アプリケーション"""

import streamlit as st
import numpy as np
import cv2
from typing import Optional, Dict, List, Tuple, Any
import tempfile
import os
import io
import matplotlib.pyplot as plt

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


def main() -> None:
    """メインアプリケーション（Cloud Run 対応：解析中に画面がリセットされない）"""
    st.title("⚾ 野球フォーム解析アプリ")
    st.markdown("---")
    
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


if __name__ == "__main__":
    main()

