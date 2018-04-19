# GAN-NMT

Yangらの
[Improving Neural Machine Translation with Conditional Sequence Generative Adversarial Nets](https://arxiv.org/abs/1703.04887v1)
におけるRNNモデルを参考にした実装です。

## 1. 環境設定
はじめに、以下の環境設定が必要（バージョンは推奨）です。このうち、chainea及びcupyは記載されたバージョンであることを強く推奨します。
- Python 3.5.1
- chainer (ver 4.0.0)
- numpy (ver 1.14.2)
- cupy (ver 4.0.0)
- h5py (ver 2.7.1)
- gensim (ver 2.2.0)

## 2. 実験設定
実験の設定はconfigファイルにて行います。/sample/sample\_gan.configが設定例です。
なお、nmt-chainerで事前学習したGenerator（NMT）モデルが必要です。


- **pre_model** : 事前学習したモデルの名前を指定してください。
- **model** : 保存するモデルの名前を指定してください。
- **source_train** : 学習用ソースファイルのパスを指定してください。
- **target_train** : 学習用ターゲットファイルのパスを指定してください。
- **output_train** : 事前学習用Generator出力サンプルファイルのパスを指定してください。
- **source_dev** : 開発用ソースファイルのパスを指定してください。
- **source_test** : 評価用ソースファイルのパスを指定してください。
- **use_gpu** : GPUを使用する場合はTrue、CPUを使用する場合はFalseを指定してください。
- **gpu_device** : 使用するGPUの番号を整数で指定してください。
- **use_word2vec** : gensimのword2vecを用いてEncoderおよびDecoderのword embeddingを初期化することができます。新しくデータを作成して用いる場合は"Make"、すでに作成済みのデータを用いる場合は"Load"、word2vecを用いずにランダムな初期化を行う場合は"None"を指定してください。 
- **source_word2vec_file** : 作成済みのword2vecを用いる場合のソース言語ファイルのパスを指定してください。
- **target_word2vec_file** : 作成済みのword2vecを用いる場合のターゲット言語ファイルのパスを指定してください。
- **epoch** : 学習時のエポック数を整数で指定してください。
- **generator_optimizer** : 学習時のGeneratorの最適化手法を"SGD"、"Adam"、"AdaDelta"、"AdaGrad"の中から指定してください。
- **discriminator_optimizer** : 学習時Discriminatorの最適化手法を"SGD"、"Adam"、"AdaDelta"、"AdaGrad"の中から指定してください。
- **learning_rate** : 学習時の最適化手法における初期学習率を小数で指定してください。ただし、最適化手法が"SGD"および"AdaGrad"の時のみ有効です。
- **use_dropout** : 学習時にdropoutを適用する場合はTrue、適用しない場合はFalseを指定してください。
- **dropout_rate** : 学習時のdropoutを適用する場合の適用率を小数で指定してください。
- **source_vocabulary_size** : ソース言語側の語彙サイズを整数で指定してください。
- **target_vocabulary_size** : ターゲット言語側の語彙サイズを整数で指定してください。
- **embed_size** : EncoderおよびDecoderのword embeddingの次元数を整数で指定してください。
- **hidden_size** : EncoderおよびDecoderの隠れ層の次元数を整数で指定してください。
- **batch_size** : 学習時のミニバッチサイズを整数で指定してください。
- **pooling** : 学習時にはコーパス全体の文をランダムに入れ替えた後、batch\_size×pooling文のブロック内で文がソートされ、ミニバッチを作成します。この数は整数で指定してください。
- **generation_limit** : 開発時および評価時の生成単語制限数を整数で指定してください。この機能は無限に単語を生成し続けることを防ぐためのものです。指定した制限数の単語をすでに生成している場合、仮に翻訳の途中であっても文が終了してしまうことに注意してください。
- **use_beamsearch** : 評価時にbeam searchを行う場合はTrue、行わない場合はFalseを指定してください。
- **beam_size** : 評価時にbeam searchを行う場合のbeam sizeを整数で指定してください。

## 3. 実行方法
プログラムを実行するには、モデルを保存したい（保存してある）ディレクトリで以下のコマンドを実行してください。

**Discriminatorの事前学習**
```
python src/rnn_csgan_pretrain.py [MODE] [CONFIG_PATH] [BEST_EPOCH (only testing)]
```

**全体学習**
```
python src/rnn_csgan.py [MODE] [CONFIG_PATH] [BEST_PRE_EPOCH (on training) / BEST_EPOCH (on testing)]
```

- **MODE** : "train"、"dev"、"test"のいずれかを指定してください。ただし、"train"済みのモデルが存在しない場合は"dev"、"test"モードは正しく実行されません。
- **CONFIG_PATH** : 実験設定を記述したconfigファイルのパスを指定してください。
