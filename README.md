# GAN-NMT

## English

This is an implementation of GAN-NMT based on RNN-CSGAN in [Improving Neural Machine Translation with Conditional Sequence Generative Adversarial Nets](https://arxiv.org/abs/1703.04887v1).

### 1. Environmental Settings
You have to install these module. The written virsion is recommended.
- Python 3.5.1
- chainer (ver 4.0.0)
- numpy (ver 1.14.2)
- cupy (ver 4.0.0)
- h5py (ver 2.7.1)
- gensim (ver 2.2.0)

### 2. Experimental Settings
You have to write experimental settings in the configration file. You can see the sample configration file [sample\_gan.config](https://github.com/yukio326/GAN-NMT/blob/master/sample/sample_gan.config).
GAN-NMT needs NMT (Generator) model pre-trained by [nmt-chainer](https://github.com/yukio326/nmt-chainer).


- **model** : Model name.
- **pre_model** : Pre-trained Generator model name.
- **pre_best_epoch** : The best epoch number of the pre-trained Generator model.
- **source_train** : The path to source train file.
- **target_train** : The path to target train file.
- **output_train** : The path to output train file on pre-training of discriminator.
- **source_dev** : The path to source development file.
- **source_test** : The path to source test file.
- **use_gpu** : True / False
- **gpu_device** : The GPU number.
- **use_word2vec** : "Make" / "Load" / "None" 
- **source_word2vec_file** : The path to source word2vec file.
- **target_word2vec_file** : The path to target word2vec file.
- **epoch** : The epoch number on training.
- **generator_optimizer** : "SGD" / "Adam" / "AdaDelta" / "AdaGrad"
- **discriminator_optimizer** : "SGD" / "Adam" / "AdaDelta" / "AdaGrad"
- **learning_rate** : The initial learning rate.
- **use_dropout** : True / False
- **dropout_rate** : The dropout rate.
- **source_vocabulary_size** : The source vocabulary size.
- **target_vocabulary_size** : The target vocabulary size.
- **embed_size** : The embedding size.
- **hidden_size** : The hidden size.
- **batch_size** : The minibatch size.
- **pooling** : The minibatch pooling size.
- **generation_limit** : The generation limit number on testing.
- **use_beamsearch** : True / False
- **beam_size** : The beam size on testing.

### 3. Execution


**Pre-Training of Discriminator**
```
python src/rnn_csgan_pretrain.py [MODE] [CONFIG_PATH] [BEST_EPOCH (only testing)]
```

**Adversarial Training**
```
python src/rnn_csgan.py [MODE] [CONFIG_PATH] [BEST_PRE_EPOCH (on training) / BEST_EPOCH (on testing)]
```


- **MODE** : "train" / "dev" / "test"
- **CONFIG_PATH** : The path to configration file.
- **BEST_PRE_EPOCH** : The best epoch number of the pre-trained Discriminator model on adversarial training.
- **BEST_EPOCH** : The epoch number of model using on testing.


## 日本語

Yangらの
[Improving Neural Machine Translation with Conditional Sequence Generative Adversarial Nets](https://arxiv.org/abs/1703.04887v1)
におけるRNNモデルを参考にした実装です。

### 1. 環境設定
はじめに、以下の環境設定が必要（バージョンは推奨）です。このうち、chainer及びcupyは記載されたバージョンであることを強く推奨します。
- Python 3.5.1
- chainer (ver 4.0.0)
- numpy (ver 1.14.2)
- cupy (ver 4.0.0)
- h5py (ver 2.7.1)
- gensim (ver 2.2.0)

### 2. 実験設定
実験の設定はconfigファイルにて行います。/sample/sample\_gan.configが設定例です。
なお、nmt-chainerで事前学習したGenerator（NMT）モデルが必要です。


- **model** : 保存するモデルの名前を指定してください。
- **pre_model** : 事前学習したGeneratorモデルの名前を指定してください。
- **pre_best_epoch** : 事前学習したモデルにおける最良モデルのエポック数を整数で指定してください。
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

### 3. 実行方法
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
- **BEST_PRE_EPOCH** : 全体学習時に用いる、事前学習したDiscriminatorモデルにおける最良モデルのエポック数を整数で指定してください。
- **BEST_EPOCH** : "test"モードのときのみ、使用するモデルのエポック数を整数で指定してください。
