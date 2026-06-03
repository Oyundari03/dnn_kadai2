# 課題4 Detection: Faster R-CNN / SSD

## 実行ファイル

- `04_detection_fasterrcnn_ssd.ipynb`

## 内容

CNN による物体検出を pretrained model で実行します。
Faster R-CNN と SSD を同じ COCO 画像に対して動かし、設計思想と出力の違いを確認します。
学習は行いません。

## コードの説明

1. `/export/data/dataset/COCO` の annotation を使い、`person` が写っている画像と `bus` が写っている画像を1枚ずつ選びます。
2. torchvision の pretrained Faster R-CNN を読み込みます。
3. 同じ画像に対して pretrained SSD を読み込みます。
4. 両方の detector が返す `boxes`, `labels`, `scores` を確認します。
5. score 0.5 以上の bounding box を描画します。
6. Faster R-CNN と SSD の結果を並べて `out/` に保存します。

## 保存先

- 選択画像: `out/selected_detection_images.png`
- 比較結果画像: `out/detection_fasterrcnn_vs_ssd.png`
- 検出結果JSON: `out/detection_summary.json`
- checkpoint等を手動保存する場合: `model/`
- 使用データ: `/export/data/dataset/COCO`

## 設計思想の違い

- Faster R-CNN: two-stage detector。候補領域を作ってから分類と位置補正を行うため、精度重視。
- SSD: single-stage detector。候補領域生成を別段階に分けず、default box から直接カテゴリと位置を予測するため、高速化しやすい。

