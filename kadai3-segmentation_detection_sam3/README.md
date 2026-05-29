# 課題3 Segmentation: FCN / SAM 3

## 実行ファイル

- `03_segmentation_detection_sam3.ipynb`

## 内容

学校の COCO 画像を入力として、torchvision の pretrained model を使い、FCN による semantic segmentation を実行します。
SAM 3 / SAM 3.1 は checkpoint と依存関係を準備した場合だけ `RUN_SAM3=True` にして実行します。

## SAM 3 checkpoint の準備

SAM 3 の checkpoint は自動ダウンロードせず、手動でダウンロードしてこの課題フォルダの `model/` に置きます。

配置例:

```text
kadai3-segmentation_detection_sam3/
|-- model/
|   `-- sam3.pt
```

notebook では `model/sam3.pt` を参照する想定です。
checkpoint を配置したあと、SAM 3 のセルで `RUN_SAM3=True` に変更して実行します。

依存関係が入っていない場合は、JupyterHub の環境で以下を実行します。

```bash
pip install -U ultralytics
```

## コードの説明

1. `/export/data/dataset/COCO` の annotation を使い、`person` が写っている画像と `bus`  が写っている画像を1枚ずつ選びます。
2. Segmentation が AutoEncoder と同じ Encoder-Decoder 型の考え方を持つことを確認します。
3. 2枚の画像に対して FCN-ResNet50 で pixel-wise な semantic segmentation label map を推定します。
4. FCN の出力 feature map の shape と label map を確認します。
5. 推論した segmentation map を `out/` に保存します。
6. SAM 3 は checkpoint 準備後に `RUN_SAM3=True` にして、1枚目は `person`、2枚目は `bus`  の text prompt segmentation を試します。

## 保存先

- 推論結果画像: `out/`
- checkpoint 等を保存する場合: `model/`
- 使用データ: `/export/data/dataset/COCO`
