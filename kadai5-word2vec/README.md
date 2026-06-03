# 課題5 Word2Vec

## 実行ファイル

- `05_word2vec.ipynb`

## 内容

研究室共通データセット内のローカル word vector を優先して使い、単語ベクトルの演算を5通り以上試します。
例: `king - man + woman` のような意味的演算を確認します。

## コードの説明

1. `/export/data/dataset/google/GoogleNews-vectors-negative300.bin` などのローカル word2vec モデルを探します。
2. 見つかった場合は `KeyedVectors.load_word2vec_format()` で読み込みます。
3. 見つからない場合は、`ALLOW_DOWNLOAD=True` にしたときだけ gensim downloader で軽量 GloVe を取得します。
4. `most_similar(positive=..., negative=...)` を使って、単語ベクトルの足し算・引き算を行います。
5. 結果を notebook に表示し、`out/word2vec_results.txt` に保存します。

## 保存先

- 結果: `out/word2vec_results.txt`
- 使用候補データ: `/export/data/dataset/google/`
