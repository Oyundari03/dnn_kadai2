# 課題5 Vision Transformer

## 実行ファイル

- `05_vision_transformer_cifar10.ipynb`

## 内容

`/export/data/dataset/CIFAR` の CIFAR-10 を使い、ImageNet 事前学習済み ViT-B/16 を小規模 subset で fine-tuning します。
ViT は画像を patch に分割し、各 patch を token として Transformer に入力するモデルです。

## コードの説明

1. `/export/data/dataset/CIFAR` から `cifar-10-batches-py` を探します。
2. `download=False` で CIFAR-10 を読み込み、新たなダウンロードは行いません。
3. ViT-B/16 の入力に合わせて 224x224 にリサイズし、ImageNet mean/std で正規化します。
4. ImageNet 事前学習済み ViT の最後の分類 head を CIFAR-10 用の10クラスに置き換えます。
5. 小規模 subset で fine-tuning し、loss と accuracy を記録します。
6. 学習済みモデルを `model/` に保存します。

## 保存先

- 学習済みモデル: `model/vit_cifar10_subset.pt`
- 学習履歴: `out/vit_history.csv`
- 使用データ: `/export/data/dataset/CIFAR`
