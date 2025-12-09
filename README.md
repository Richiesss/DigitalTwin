# YOLO11 姿勢検出・トラッキングプロジェクト

YOLO11とNorfairを使用した人物の姿勢検出、トラッキング、うつむき判定を行うプロジェクトです。

## 機能

- **姿勢検出**: YOLO11を使用して人物の17個のキーポイント（鼻、目、耳、肩、肘、手首、腰、膝、足首）を検出
- **トラッキング**: Norfairを使用して複数の人物を追跡し、一意のIDを割り当て
- **うつむき判定**: 鼻と耳の位置関係からうつむき状態を判定
- **CSV出力**: フレームごとのキーポイント座標とトラッキング情報を保存
- **動画出力**: トラッキング結果とうつむき判定を描画した動画を出力
- **可視化**: 特定の人物のキーポイント軌跡をグラフ化

## セットアップ

### 1. 仮想環境の作成（推奨）

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# または
.venv\Scripts\activate  # Windows
```

### 2. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

必要なパッケージ:
- norfair
- ultralytics==8.3.47
- opencv-python==4.6.0.66
- pandas
- matplotlib
- numpy

## 使い方

### 1. 姿勢検出とトラッキングの実行

基本的な使い方:

```bash
python pose_detection.py <入力動画ファイル>
```

オプションを指定:

```bash
python pose_detection.py input_video.mp4 \
    --output output.mp4 \
    --csv pose_output.csv \
    --model yolo11s-pose.pt \
    --detection-threshold 0.1 \
    --distance-threshold 0.4
```

#### オプション

| オプション | 説明 | デフォルト値 |
|-----------|------|-------------|
| `input_video` | 入力動画ファイルのパス（必須） | - |
| `-o, --output` | 出力動画ファイルのパス | `output.mp4` |
| `-c, --csv` | 出力CSVファイルのパス | `pose_output.csv` |
| `-m, --model` | YOLOモデルのパス | `yolo11s-pose.pt` |
| `--detection-threshold` | 検出閾値 | `0.1` |
| `--distance-threshold` | トラッキングの距離閾値 | `0.4` |
| `--initialization-delay` | トラッキング初期化遅延 | `4` |
| `--hit-counter-max` | ヒットカウンター最大値 | `30` |
| `--pointwise-hit-counter-max` | ポイント毎のヒットカウンター最大値 | `10` |

### 2. キーポイントの可視化

#### tracking IDのリストを表示

```bash
python visualize_keypoints.py --csv pose_output.csv --list-ids
```

#### 特定のキーポイントをプロット

```bash
python visualize_keypoints.py \
    --csv pose_output.csv \
    --tracking-id 1 \
    --keypoint nose \
    --output plot.png
```

#### オプション

| オプション | 説明 | デフォルト値 |
|-----------|------|-------------|
| `-c, --csv` | 入力CSVファイルのパス | `pose_output.csv` |
| `-t, --tracking-id` | プロットするtracking ID | - |
| `-k, --keypoint` | プロットするキーポイント | `nose` |
| `-o, --output` | プロット画像の保存先（指定しない場合は画面表示） | - |
| `--list-ids` | 利用可能なtracking IDのリストを表示 | - |

#### 利用可能なキーポイント

- `nose` (鼻)
- `left-eye` (左目)
- `right-eye` (右目)
- `left-ear` (左耳)
- `right-ear` (右耳)
- `left-shoulder` (左肩)
- `right-shoulder` (右肩)
- `left-elbow` (左肘)
- `right-elbow` (右肘)
- `left-wrist` (左手首)
- `right-wrist` (右手首)
- `left-hip` (左腰)
- `right-hip` (右腰)
- `left-knee` (左膝)
- `right-knee` (右膝)
- `left-ankle` (左足首)
- `right-ankle` (右足首)

## 出力ファイル

### 動画ファイル (output.mp4)

- トラッキングされた人物のキーポイントとスケルトンが描画されます
- 各人物にトラッキングIDが表示されます
- うつむき判定の結果が表示されます（`look_down: True/False`）

### CSVファイル (pose_output.csv)

各フレームごとに以下の情報が記録されます:

| カラム | 説明 |
|--------|------|
| `frame` | フレーム番号 |
| `nose_x, nose_y` | 鼻の座標 |
| `left-eye_x, left-eye_y` | 左目の座標 |
| ... | その他のキーポイント座標 |
| `frame_height` | フレームの高さ |
| `frame_width` | フレームの幅 |
| `tracking_id` | トラッキングID |
| `dist_ear_nose` | 耳と鼻の距離（うつむき判定用） |
| `look_down` | うつむき判定（True/False） |

## うつむき判定のロジック

うつむき判定は以下の計算式で行われます:

```python
dist_ear_nose = nose_y - max(left_ear_y, right_ear_y)
look_down = (dist_ear_nose >= 0)
```

- `dist_ear_nose >= 0`: 鼻が耳より下にある場合、うつむいていると判定
- `dist_ear_nose < 0`: 鼻が耳より上にある場合、うつむいていないと判定

## 使用例

### 例1: 基本的な使い方

```bash
# 1. 動画を処理
python pose_detection.py sample_video.mp4

# 2. tracking IDのリストを確認
python visualize_keypoints.py --list-ids

# 3. 特定の人物の鼻の軌跡をプロット
python visualize_keypoints.py --tracking-id 1 --keypoint nose
```

### 例2: カスタム設定

```bash
# 検出閾値を上げてより確実な検出のみを使用
python pose_detection.py video.mp4 \
    --detection-threshold 0.3 \
    --output high_confidence.mp4 \
    --csv high_confidence.csv
```

### 例3: 複数のキーポイントを可視化

```bash
# 左目をプロット
python visualize_keypoints.py -t 1 -k left-eye -o left_eye.png

# 右目をプロット
python visualize_keypoints.py -t 1 -k right-eye -o right_eye.png

# 鼻をプロット
python visualize_keypoints.py -t 1 -k nose -o nose.png
```

### 3. 集中度分析

トラッキング結果のCSVファイルから、各人物の集中度（うつむいていない時間の割合）を分析し、タイムラインをグラフ化します。

```bash
python concentration_analysis.py
```

#### 注意事項

- デフォルトでは `pose_output.csv` を読み込みます
- 別のCSVファイルを使用する場合は、スクリプト内の7行目を編集してください

```python
df = pd.read_csv("pose_output.csv")  # ここを変更
```

#### 出力

1. **コンソール出力**: 各トラッキングIDの統計情報

```
Concentration Statistics:
   Tracking ID  Total Frames  Look Down Frames  Concentration Rate (%)
0            1          1234               234                    81.0
1            2           987               123                    87.5
...
```

統計情報の説明:
- **Tracking ID**: 人物のID
- **Total Frames**: その人物が検出されたフレーム数
- **Look Down Frames**: うつむいていたフレーム数
- **Concentration Rate (%)**: 集中度（うつむいていない時間の割合）

2. **グラフ画像** (`concentration_analysis_final.png`): 各人物の集中状態のタイムライン
   - X軸: フレーム番号
   - Y軸: トラッキングID（Student ID）
   - 緑色: 集中している状態（うつむいていない）
   - 赤色: うつむいている状態（集中していない可能性）

#### スクリプトの処理内容

1. CSVファイルを読み込み、有効なトラッキングIDのみをフィルタリング
2. 無効な座標（鼻の座標が0のデータなど）を除去
3. 各トラッキングIDごとに集中度を計算
4. タイムラインをグラフ化し、PNG画像として保存

## トラブルシューティング

### モデルのダウンロードエラー

初回実行時、YOLOモデル（`yolo11s-pose.pt`）が自動的にダウンロードされます。ネットワーク環境によってはダウンロードに時間がかかる場合があります。

### メモリ不足

長時間の動画や高解像度の動画を処理する場合、メモリ不足になる可能性があります。その場合は:
- より軽量なモデル（yolo11n-pose.pt）を使用
- 動画を分割して処理
- 解像度を下げる

### トラッキングの精度が低い

トラッキングパラメータを調整してください:
- `--distance-threshold`: 値を小さくすると、より厳格なトラッキングになります
- `--hit-counter-max`: 値を大きくすると、一時的に見えなくなった人物のIDを保持しやすくなります

## ライセンス

このプロジェクトで使用しているライブラリのライセンス:
- Ultralytics YOLO: AGPL-3.0
- Norfair: Apache-2.0
- OpenCV: Apache-2.0

## 参考

- [Ultralytics YOLO11](https://docs.ultralytics.com/)
- [Norfair](https://github.com/tryolabs/norfair)
- [COCO Keypoints](https://cocodataset.org/#keypoints-2020)