# 課題12 CLIP Zero-shot Image Classification

## 実行ファイル

- `12_clip_zero_shot_coco.ipynb`

## 内容

CLIP を使って COCO 画像の zero-shot 分類を行います。
画像エンコーダとテキストエンコーダが同じ embedding 空間に特徴量を写像するため、`a photo of a person` のようなテキスト候補と画像の類似度を比べることで、学習なしで分類できます。

## モデル準備

外部ネットワークが使えない場合は、Hugging Face から `openai/clip-vit-base-patch32` を手動で取得し、以下に置きます。

```text
kadai12-clip_zero_shot/model/clip-vit-base-patch32/
```

notebook はこのローカルモデルを優先して読みます。

## 保存先

- 結果画像: `out/clip_zero_shot_results.png`
- 結果CSV: `out/clip_zero_shot_results.csv`
- 使用データ: `data/COCO -> /export/data/dataset/COCO`

