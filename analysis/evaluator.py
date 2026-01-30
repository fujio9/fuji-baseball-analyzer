"""投球フォーム評価モジュール"""

from typing import Dict, Any, List, Optional


def _score_from_range(
    value: Optional[float],
    ideal_min: float,
    ideal_max: float,
    tol: float = 10.0,
) -> float:
    """理想範囲に対するスコアを 0-100 で算出

    - ideal_min〜ideal_max を 100 点
    - その外側 tol 度までは線形に減点
    - それ以上離れると 0 点
    """
    if value is None:
        return 0.0

    v = float(value)
    if ideal_min <= v <= ideal_max:
        return 100.0

    if v < ideal_min:
        diff = ideal_min - v
    else:
        diff = v - ideal_max

    if diff >= tol:
        return 0.0

    return max(0.0, 100.0 * (1.0 - diff / tol))


def evaluate_pitching_form(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """投球フォームを総合評価する

    Args:
        metrics: compute_pitching_metrics() の結果

    Returns:
        {
            "score": 総合スコア(0-100),
            "subscores": 各指標ごとのスコア辞書,
            "comments": コメント一覧,
        }
    """

    subscores: Dict[str, float] = {}
    comments: List[str] = []

    # 1. 最大肘角度（過度なしなりを抑制）
    max_elbow = metrics.get("max_elbow_angle")
    subscores["最大肘角度"] = _score_from_range(max_elbow, ideal_min=150.0, ideal_max=180.0, tol=20.0)
    if max_elbow is not None:
        if max_elbow > 180.0:
            comments.append("肘のしなりが大きすぎる可能性があります（最大肘角度が180°を超えています）。")
        elif max_elbow < 140.0:
            comments.append("テイクバックで肘が十分に伸びていない可能性があります。")

    # 2. リリース時肘角度（適度な屈曲）
    release_elbow = metrics.get("release_elbow_angle")
    subscores["リリース時肘角度"] = _score_from_range(release_elbow, ideal_min=80.0, ideal_max=110.0, tol=20.0)
    if release_elbow is not None:
        if release_elbow > 120.0:
            comments.append("リリース時の肘が伸びすぎている可能性があります。")
        elif release_elbow < 70.0:
            comments.append("リリース時の肘の曲げが強く、ボールリリースが遅れているかもしれません。")

    # 3. 体幹傾き（前傾角）
    torso_rel = metrics.get("torso_angle_at_release")
    subscores["体幹傾き"] = _score_from_range(torso_rel, ideal_min=10.0, ideal_max=35.0, tol=20.0)
    if torso_rel is not None:
        if torso_rel < 5.0:
            comments.append("リリース時の体幹前傾が少なく、上半身のエネルギーが十分に伝わっていない可能性があります。")
        elif torso_rel > 40.0:
            comments.append("リリース時の体幹前傾が大きく、バランスを崩している可能性があります。")

    # 4. 肩ライン角度（開き具合）
    shoulder_rel = metrics.get("shoulder_angle_at_release")
    subscores["肩の開き"] = _score_from_range(shoulder_rel, ideal_min=-10.0, ideal_max=20.0, tol=30.0)
    if shoulder_rel is not None:
        if shoulder_rel > 25.0:
            comments.append("リリース時の肩の開きが早く、ボールが抜けやすいフォームかもしれません。")
        elif shoulder_rel < -15.0:
            comments.append("肩の開きが抑えられすぎており、腕の振りが窮屈になっている可能性があります。")

    # 5. 骨盤角度（骨盤の開き）
    hip_rel = metrics.get("hip_angle_at_release")
    subscores["骨盤角度"] = _score_from_range(hip_rel, ideal_min=0.0, ideal_max=30.0, tol=30.0)
    if hip_rel is not None:
        if hip_rel > 35.0:
            comments.append("骨盤の開きが大きく、下半身の回転が先行しすぎている可能性があります。")
        elif hip_rel < -5.0:
            comments.append("骨盤の回転が不足しており、下半身主導の動きが弱いかもしれません。")

    # 総合スコア：各サブスコアの平均
    if subscores:
        total_score = sum(subscores.values()) / len(subscores)
    else:
        total_score = 0.0

    # コメントが1つもない場合はポジティブなコメントを追加
    if not comments:
        comments.append("全体としてバランスの良いフォームです。引き続きこのフォームを維持しましょう。")

    return {
        "score": int(round(total_score)),
        "subscores": subscores,
        "comments": comments,
    }

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