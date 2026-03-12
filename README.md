# mc_research_model2

## 概要
このプロジェクトは、ニューラルネットワークを用いて原子核物理学における **IBM2 (Interacting Boson Model 2)** のハミルトニアンパラメータ ($\epsilon, \kappa, \chi_\nu$) を推定するための研究用コードです。

Hartree-Fock-Bogoliubov (HFB) 法などで計算されたポテンシャルエネルギー曲面 (PES) や実験データから、最適なIBM2パラメータを逆推定することを目的としています。また、推定されたパラメータを用いて `NPBOS` コードを実行し、エネルギースペクトルを計算して実験値と比較評価する機能も備えています。

## ディレクトリ構造

```
mc_research_model2/
├── config.yml          # プロジェクト全体の設定ファイル
├── requirements.txt    # Python依存ライブラリ一覧
├── src/                # Pythonソースコード
│   ├── train.py        # モデル学習用スクリプト
│   ├── eval.py         # モデル評価用スクリプト
│   ├── model.py        # ニューラルネットワークモデル定義
│   ├── data.py         # データ読み込み・前処理
│   ├── physics.py      # 物理計算（PESなど）
│   └── ...
├── NPBOS/              # IBM2計算用Fortranコード (T. Otsuka氏による)
│   ├── compile.sh      # コンパイル用スクリプト
│   └── ...
├── dataset/            # データセットディレクトリ
│   ├── raw/            # 生データ
│   └── processed/      # 前処理済みデータ
└── results/            # 出力ディレクトリ（モデル、ログ、評価結果）
```

## 必要要件

*   **OS**: Linux (推奨)
*   **Python**: 3.8 以上
*   **Fortranコンパイラ**: `gfortran` (NPBOSのコンパイルに必要)

### Python ライブラリ
*   numpy
*   scipy
*   pandas
*   matplotlib
*   torch (PyTorch)
*   pyyaml

## セットアップ

### 1. Python環境の構築

仮想環境を作成し、依存ライブラリをインストールすることをお勧めします。

```bash
# 仮想環境の作成と有効化 (例)
python -m venv venv
source venv/bin/activate

# ライブラリのインストール
pip install -r requirements.txt
```

### 2. NPBOS (Fortranコード) のコンパイル

評価パートでスペクトル計算を行うために、`NPBOS` ディレクトリ内のFortranプログラムをコンパイルし、必要なデータファイルを生成する必要があります。

```bash
cd NPBOS
chmod +x compile.sh
./compile.sh
cd ..
```
※ `compile.sh` は `gfortran` を使用して複数のソースコードをコンパイルし、初期化プログラムを実行します。

## 使用方法

**注意**: 以下のコマンドはすべて、プロジェクトのルートディレクトリ（`project1/`）で実行してください。

### 設定 (Configuration)

`config.yml` ファイルで、データパス、対象とする原子核の範囲、ニューラルネットワークの構造、学習パラメータなどを設定できます。

```yaml
nuclei:
  p_min: 62
  p_max: 62
  ...
nn:
  input_dim: 3
  output_dim: 3
  ...
training:
  batch_size: 32
  num_epochs: 500
  ...
```

### 学習 (Training)

ニューラルネットワークの学習を実行します。

```bash
python -m src.train
```
学習済みモデルは `results/training/` 以下に保存されます。複数の隠れ層構成（パターン）を並列で学習する機能があります。

### 評価 (Evaluation)

学習済みモデルを用いてパラメータを推定し、NPBOSを実行してスペクトルを計算・評価します。

```bash
python -m src.eval
```
評価結果は `results/evaluation/` にCSV形式で保存されます。

## データセットについて

*   **Raw Data**: `dataset/raw/{Z}/` ディレクトリに、各陽子数 $Z$ (例: 62) ごとのデータが配置されます。
    *   `{N}.csv`: 中性子数 $N$ ごとのHFB計算データなど。
    *   `expt.csv`: 実験値スペクトルデータ。
*   **Processed Data**: 学習に使用する `numpy` 配列などが `dataset/processed/` に保存されます。

## クレジット

このプロジェクトに含まれる `NPBOS` ディレクトリ内のFortranコードは、**T. Otsuka** 氏によって開発されたものです。
ニューラルネットワーク部分およびPythonラッパーコードは本プロジェクト独自の実装です。
