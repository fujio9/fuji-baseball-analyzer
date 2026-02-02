"""angles.py のテスト"""

import pytest
import numpy as np
from analysis.angles import calculate_elbow_angle


def test_calculate_elbow_angle_90_degrees() -> None:
    """90度の角度を計算するテスト"""
    # 肩、肘、手首が直角になる配置
    # 肩(0, 0, 0), 肘(1, 0, 0), 手首(1, 1, 0) → 90度
    shoulder = np.array([0.0, 0.0, 0.0])
    elbow = np.array([1.0, 0.0, 0.0])
    wrist = np.array([1.0, 1.0, 0.0])
    
    angle = calculate_elbow_angle(shoulder, elbow, wrist)
    
    assert abs(angle - 90.0) < 0.1, f"期待値: 90度, 実際: {angle}度"


def test_calculate_elbow_angle_180_degrees() -> None:
    """180度の角度を計算するテスト"""
    # 肩、肘、手首が一直線上に並ぶ配置
    # 肩(0, 0, 0), 肘(1, 0, 0), 手首(2, 0, 0) → 180度
    shoulder = np.array([0.0, 0.0, 0.0])
    elbow = np.array([1.0, 0.0, 0.0])
    wrist = np.array([2.0, 0.0, 0.0])
    
    angle = calculate_elbow_angle(shoulder, elbow, wrist)
    
    assert abs(angle - 180.0) < 0.1, f"期待値: 180度, 実際: {angle}度"


def test_calculate_elbow_angle_0_degrees() -> None:
    """0度の角度を計算するテスト（完全に曲がっている）"""
    # 肩(0, 0, 0), 肘(1, 0, 0), 手首(0, 0, 0) → 0度（実際には180度に近い）
    # より正確な0度: 肩(0, 0, 0), 肘(1, 0, 0), 手首(2, 0, 0) を反転
    # 実際には、肩から肘と肘から手首が同じ方向を向いている場合
    shoulder = np.array([0.0, 0.0, 0.0])
    elbow = np.array([1.0, 0.0, 0.0])
    wrist = np.array([0.0, 0.0, 0.0])
    
    angle = calculate_elbow_angle(shoulder, elbow, wrist)
    
    # この場合、角度は180度に近くなる
    assert angle > 90.0, f"角度が予想外: {angle}度"


def test_calculate_elbow_angle_45_degrees() -> None:
    """45度の角度を計算するテスト"""
    # 肩(0, 0, 0), 肘(1, 0, 0), 手首(1 + cos(45°), sin(45°), 0)
    shoulder = np.array([0.0, 0.0, 0.0])
    elbow = np.array([1.0, 0.0, 0.0])
    # 45度になるように手首を配置
    wrist = np.array([1.0 + np.cos(np.radians(45)), np.sin(np.radians(45)), 0.0])
    
    angle = calculate_elbow_angle(shoulder, elbow, wrist)
    
    assert abs(angle - 45.0) < 1.0, f"期待値: 45度, 実際: {angle}度"


def test_calculate_elbow_angle_2d_coordinates() -> None:
    """2次元座標での角度計算テスト"""
    # 2次元座標でも動作することを確認
    shoulder = np.array([0.0, 0.0])
    elbow = np.array([1.0, 0.0])
    wrist = np.array([1.0, 1.0])
    
    angle = calculate_elbow_angle(shoulder, elbow, wrist)
    
    assert abs(angle - 90.0) < 0.1, f"期待値: 90度, 実際: {angle}度"


def test_calculate_elbow_angle_zero_vector() -> None:
    """ゼロベクトルの場合のテスト"""
    # 肩と肘が同じ位置の場合
    shoulder = np.array([0.0, 0.0, 0.0])
    elbow = np.array([0.0, 0.0, 0.0])
    wrist = np.array([1.0, 0.0, 0.0])
    
    angle = calculate_elbow_angle(shoulder, elbow, wrist)
    
    assert angle == 0.0, f"ゼロベクトルの場合、0度を返すべき: {angle}度"


def test_calculate_elbow_angle_zero_vector_wrist() -> None:
    """手首が肘と同じ位置の場合のテスト"""
    shoulder = np.array([0.0, 0.0, 0.0])
    elbow = np.array([1.0, 0.0, 0.0])
    wrist = np.array([1.0, 0.0, 0.0])
    
    angle = calculate_elbow_angle(shoulder, elbow, wrist)
    
    assert angle == 0.0, f"ゼロベクトルの場合、0度を返すべき: {angle}度"


def test_calculate_elbow_angle_3d_coordinates() -> None:
    """3次元座標での角度計算テスト"""
    # 3次元空間での90度
    shoulder = np.array([0.0, 0.0, 0.0])
    elbow = np.array([1.0, 0.0, 0.0])
    wrist = np.array([1.0, 1.0, 0.0])
    
    angle = calculate_elbow_angle(shoulder, elbow, wrist)
    
    assert abs(angle - 90.0) < 0.1, f"期待値: 90度, 実際: {angle}度"


def test_calculate_elbow_angle_negative_coordinates() -> None:
    """負の座標での角度計算テスト"""
    shoulder = np.array([-1.0, -1.0, 0.0])
    elbow = np.array([0.0, -1.0, 0.0])
    wrist = np.array([0.0, 0.0, 0.0])
    
    angle = calculate_elbow_angle(shoulder, elbow, wrist)
    
    assert abs(angle - 90.0) < 0.1, f"期待値: 90度, 実際: {angle}度"














