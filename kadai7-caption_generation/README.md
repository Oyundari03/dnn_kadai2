# 課題7 Sentence Generation: Image Captioning

## 実行ファイル

- `07_caption_generation_blip.ipynb`

## 内容

COCO 画像から caption を生成します。
`data/COCO` から `/export/data/dataset/COCO` へのシンボリックリンクを作って使います。

古典的な caption generation は CNN で画像特徴を抽出し、LSTM で文章を生成する流れです。
この notebook では、オープンソースの pretrained BLIP model を使い、学習なしで caption generation を体験します。

## モデル準備

外部ネットワークが使えない場合は、Hugging Face から `Salesforce/blip-image-captioning-base` を手動で取得し、以下に置きます。

```text
kadai7-caption_generation/model/blip-image-captioning-base/
```

notebook はこのローカルモデルを優先して読みます。

## 保存先

- 結果画像: `out/caption_generation_results.png`
- 結果JSON: `out/caption_results.json`
- 結果Markdown: `out/caption_results.md`
- 使用データ: `data/COCO -> /export/data/dataset/COCO`

