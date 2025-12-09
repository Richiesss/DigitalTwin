# -*- coding: utf-8 -*-
"""
YOLO11 Pose Detection and Tracking with Norfair (Selected IDs)
指定したTracking IDのみを出力する姿勢検出・トラッキングプログラム
"""

import numpy as np
import csv
import os
import argparse

from norfair import Detection, Tracker, Video, draw_tracked_objects
from norfair.distances import create_keypoints_voting_distance
from ultralytics import YOLO


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
    parser = argparse.ArgumentParser(description="YOLO11 Pose Detection and Tracking (Selected IDs)")
    parser.add_argument("input_video", type=str, help="入力動画ファイルのパス")
    parser.add_argument("-i", "--ids", type=str, required=True,
                        help="出力対象のTracking ID (カンマ区切り、例: 1,3,5)")
    parser.add_argument("-o", "--output", type=str, default="output_selected.mp4",
                        help="出力動画ファイルのパス (デフォルト: output_selected.mp4)")
    parser.add_argument("-c", "--csv", type=str, default="pose_output_selected.csv",
                        help="出力CSVファイルのパス (デフォルト: pose_output_selected.csv)")
    parser.add_argument("-m", "--model", type=str, default="yolo11s-pose.pt",
                        help="YOLOモデルのパス (デフォルト: yolo11s-pose.pt)")
    parser.add_argument("--detection-threshold", type=float, default=0.1,
                        help="検出閾値 (デフォルト: 0.1)")
    parser.add_argument("--distance-threshold", type=float, default=0.4,
                        help="距離閾値 (デフォルト: 0.4)")
    parser.add_argument("--initialization-delay", type=int, default=4,
                        help="初期化遅延 (デフォルト: 4)")
    parser.add_argument("--hit-counter-max", type=int, default=30,
                        help="ヒットカウンター最大値 (デフォルト: 30)")
    parser.add_argument("--pointwise-hit-counter-max", type=int, default=10,
                        help="ポイント毎のヒットカウンター最大値 (デフォルト: 10)")
    return parser.parse_args()


def parse_tracking_ids(ids_str):
    """カンマ区切りのTracking ID文字列をセットに変換"""
    try:
        ids = set(int(id_str.strip()) for id_str in ids_str.split(","))
        return ids
    except ValueError:
        raise ValueError("Tracking IDは整数をカンマ区切りで指定してください (例: 1,3,5)")


def create_csv_header():
    """CSV出力用のヘッダーを作成"""
    fieldnames_list = ["frame"]
    fieldnames_list += [keypoints + coords for keypoints in COCO_KEYPOINTS
                        for coords in ["_x", "_y"]]
    fieldnames_list += ["frame_height", "frame_width", "tracking_id",
                        "dist_ear_nose", "look_down"]
    return fieldnames_list


def write_empty_frame(writer, frame_num, video_h, video_w):
    """人物が検出されなかった場合の空フレームをCSVに書き込み"""
    csv_dict = {"frame": frame_num}
    for field in COCO_KEYPOINTS:
        csv_dict[field + "_x"] = None
        csv_dict[field + "_y"] = None
    csv_dict["frame_height"] = video_h
    csv_dict["frame_width"] = video_w
    csv_dict["tracking_id"] = None
    csv_dict["dist_ear_nose"] = None
    csv_dict["look_down"] = None
    writer.writerow(csv_dict)


def write_tracked_object(writer, frame_num, video_h, video_w, obj):
    """トラッキングされたオブジェクトの情報をCSVに書き込み"""
    csv_dict = {"frame": frame_num}
    for field, coords in zip(COCO_KEYPOINTS, obj.estimate):
        csv_dict[field + "_x"] = coords[0]
        csv_dict[field + "_y"] = coords[1]
    csv_dict["frame_height"] = video_h
    csv_dict["frame_width"] = video_w
    csv_dict["tracking_id"] = obj.id

    # うつむきの判定
    dist_ear_nose = obj.estimate[0, 1] - max(obj.estimate[3, 1], obj.estimate[4, 1])
    csv_dict["dist_ear_nose"] = dist_ear_nose
    csv_dict["look_down"] = dist_ear_nose >= 0

    writer.writerow(csv_dict)


def main():
    """メイン処理"""
    args = parse_args()

    # Tracking IDのパース
    try:
        selected_ids = parse_tracking_ids(args.ids)
        print(f"選択されたTracking ID: {sorted(selected_ids)}")
    except ValueError as e:
        print(f"エラー: {e}")
        return

    # 入力動画の存在確認
    if not os.path.exists(args.input_video):
        print(f"エラー: 入力動画ファイル '{args.input_video}' が見つかりません")
        return

    print(f"入力動画: {args.input_video}")
    print(f"出力動画: {args.output}")
    print(f"出力CSV: {args.csv}")
    print(f"モデル: {args.model}")

    # YOLOモデルの読み込み
    print("YOLOモデルを読み込んでいます...")
    model = YOLO(args.model)

    # 動画の読み込み
    print("動画を読み込んでいます...")
    video = Video(input_path=args.input_video, output_path=args.output)
    video_h = video.input_height
    video_w = video.input_width

    print(f"動画サイズ: {video_w}x{video_h}")

    # トラッカーの設定
    keypoint_dist_threshold = video_h / 40
    tracker = Tracker(
        distance_function=create_keypoints_voting_distance(
            keypoint_distance_threshold=keypoint_dist_threshold,
            detection_threshold=args.detection_threshold,
        ),
        distance_threshold=args.distance_threshold,
        detection_threshold=args.detection_threshold,
        initialization_delay=args.initialization_delay,
        hit_counter_max=args.hit_counter_max,
        pointwise_hit_counter_max=args.pointwise_hit_counter_max,
    )

    # CSV出力の準備
    fieldnames_list = create_csv_header()

    print("処理を開始します...")
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames_list)
        writer.writeheader()

        for i, frame in enumerate(video):
            if i % 30 == 0:  # 30フレームごとに進捗を表示
                print(f"処理中: フレーム {i}")

            # キーポイントの検出
            results = model(frame, save=False, verbose=False)[0]

            # 人物が検出されない場合はスキップ
            if results.keypoints.conf is None:
                write_empty_frame(writer, i, video_h, video_w)
                video.write(frame)
                tracked_objects = tracker.update()
                continue

            # トラッキングの実行
            detections = [
                Detection(np.array(p), scores=s)
                for (p, s) in zip(results.keypoints.xy.cpu().numpy(),
                                  results.keypoints.conf.cpu().numpy())
            ]
            tracked_objects = tracker.update(detections=detections)

            # 選択されたIDのみをフィルタリング
            selected_objects = [obj for obj in tracked_objects if obj.id in selected_ids]

            # 選択されたIDのみを描画
            draw_tracked_objects(frame, selected_objects)

            if not selected_objects:
                write_empty_frame(writer, i, video_h, video_w)
            else:
                for obj in selected_objects:
                    # うつむきの判定
                    dist_ear_nose = obj.estimate[0, 1] - max(obj.estimate[3, 1],
                                                              obj.estimate[4, 1])
                    look_down = dist_ear_nose >= 0

                    # CSVに書き込み
                    write_tracked_object(writer, i, video_h, video_w, obj)

            # 動画の出力
            video.write(frame)

    print(f"\n処理が完了しました！")
    print(f"出力動画: {args.output}")
    print(f"出力CSV: {args.csv}")
    print(f"選択されたTracking ID: {sorted(selected_ids)}")


if __name__ == "__main__":
    main()
