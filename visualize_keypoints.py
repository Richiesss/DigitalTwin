# -*- coding: utf-8 -*-
"""
キーポイント座標の可視化スクリプト
CSVファイルから特定のtracking_idのキーポイント座標をプロットします
"""

import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path


# COCOキーポイントの定義
COCO_KEYPOINTS = [
    "nose",
    "left-eye",
    "right-eye",
    "left-ear",
    "right-ear",
    "left-shoulder",
    "right-shoulder",
    "left-elbow",
    "right-elbow",
    "left-wrist",
    "right-wrist",
    "left-hip",
    "right-hip",
    "left-knee",
    "right-knee",
    "left-ankle",
    "right-ankle",
]


def parse_args():
    """コマンドライン引数のパース"""
    parser = argparse.ArgumentParser(description="キーポイント座標の可視化")
    parser.add_argument("-c", "--csv", type=str, default="pose_output.csv",
                        help="入力CSVファイルのパス (デフォルト: pose_output.csv)")
    parser.add_argument("-t", "--tracking-id", type=int, default=None,
                        help="プロットするtracking ID")
    parser.add_argument("-k", "--keypoint", type=str, default="nose",
                        choices=COCO_KEYPOINTS,
                        help="プロットするキーポイント (デフォルト: nose)")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="プロット画像の保存先 (指定しない場合は画面表示)")
    parser.add_argument("--list-ids", action="store_true",
                        help="利用可能なtracking IDのリストを表示して終了")
    return parser.parse_args()


def list_tracking_ids(df):
    """利用可能なtracking IDのリストを表示"""
    id_list = df["tracking_id"].unique()
    id_list = id_list[~pd.isna(id_list)].astype(int).tolist()

    print("利用可能なtracking ID:")
    print("-" * 50)
    for i, id_val in enumerate(id_list):
        if (i + 1) % 10 == 0:
            print(f"{id_val}")
        else:
            print(f"{id_val} ", end="")
    print("\n" + "-" * 50)
    return id_list


def plot_keypoint(df, tracking_id, keypoint_name, output_path=None):
    """指定されたtracking_idとキーポイントの座標をプロット"""

    # tracking_idでフィルタリング
    df_filtered = df[df["tracking_id"] == tracking_id].copy()

    if df_filtered.empty:
        print(f"エラー: tracking_id {tracking_id} のデータが見つかりません")
        return False

    # frameとキーポイントの座標を抽出
    df_keypoints = df_filtered.loc[:, ["frame", keypoint_name + "_x", keypoint_name + "_y"]]

    # プロット
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_keypoints["frame"], df_keypoints[keypoint_name + "_x"],
            label=f"{keypoint_name}_x", marker='o', markersize=3)
    ax.plot(df_keypoints["frame"], df_keypoints[keypoint_name + "_y"],
            label=f"{keypoint_name}_y", marker='s', markersize=3)

    ax.set_xlabel("Frame", fontsize=12)
    ax.set_ylabel("Coordinate (pixels)", fontsize=12)
    ax.set_title(f"Keypoint Trajectory: {keypoint_name} (tracking_id={tracking_id})",
                 fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"プロットを保存しました: {output_path}")
    else:
        plt.show()

    return True


def plot_all_keypoints(df, tracking_id, output_dir=None):
    """指定されたtracking_idのすべてのキーポイントをプロット"""

    # tracking_idでフィルタリング
    df_filtered = df[df["tracking_id"] == tracking_id].copy()

    if df_filtered.empty:
        print(f"エラー: tracking_id {tracking_id} のデータが見つかりません")
        return False

    # すべてのキーポイントのX座標をプロット
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    for keypoint in COCO_KEYPOINTS:
        ax1.plot(df_filtered["frame"], df_filtered[keypoint + "_x"],
                 label=keypoint, alpha=0.7, linewidth=1)

    ax1.set_xlabel("Frame", fontsize=12)
    ax1.set_ylabel("X Coordinate (pixels)", fontsize=12)
    ax1.set_title(f"All Keypoints X Coordinates (tracking_id={tracking_id})",
                  fontsize=14)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # すべてのキーポイントのY座標をプロット
    for keypoint in COCO_KEYPOINTS:
        ax2.plot(df_filtered["frame"], df_filtered[keypoint + "_y"],
                 label=keypoint, alpha=0.7, linewidth=1)

    ax2.set_xlabel("Frame", fontsize=12)
    ax2.set_ylabel("Y Coordinate (pixels)", fontsize=12)
    ax2.set_title(f"All Keypoints Y Coordinates (tracking_id={tracking_id})",
                  fontsize=14)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_dir:
        output_path = Path(output_dir) / f"all_keypoints_tracking_id_{tracking_id}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"プロットを保存しました: {output_path}")
    else:
        plt.show()

    return True


def main():
    """メイン処理"""
    args = parse_args()

    # CSVファイルの存在確認
    if not Path(args.csv).exists():
        print(f"エラー: CSVファイル '{args.csv}' が見つかりません")
        return

    # CSVファイルの読み込み
    print(f"CSVファイルを読み込んでいます: {args.csv}")
    df = pd.read_csv(args.csv).dropna(how="any")

    if df.empty:
        print("エラー: CSVファイルにデータがありません")
        return

    # tracking IDのリスト表示モード
    if args.list_ids:
        list_tracking_ids(df)
        return

    # tracking IDのリストを取得
    id_list = df["tracking_id"].unique()
    id_list = id_list[~pd.isna(id_list)].astype(int).tolist()

    # tracking IDが指定されていない場合、リストを表示して終了
    if args.tracking_id is None:
        print("tracking IDが指定されていません。")
        list_tracking_ids(df)
        print("\n使用方法:")
        print(f"  python {Path(__file__).name} --tracking-id <ID> --keypoint <keypoint_name>")
        print(f"\n例:")
        print(f"  python {Path(__file__).name} --tracking-id 1 --keypoint nose")
        return

    # tracking IDの検証
    if args.tracking_id not in id_list:
        print(f"エラー: tracking_id {args.tracking_id} は存在しません")
        list_tracking_ids(df)
        return

    # プロットの実行
    print(f"\ntracking_id={args.tracking_id}, keypoint={args.keypoint} をプロットします")
    success = plot_keypoint(df, args.tracking_id, args.keypoint, args.output)

    if success:
        print("プロットが完了しました")


if __name__ == "__main__":
    main()