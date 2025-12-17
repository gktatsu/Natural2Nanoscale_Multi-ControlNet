# From Natural to Nanoscale: 希少なFIB-SEMデータを用いたControlNetの訓練によるセマンティックセグメンテーションデータの拡張
[![Project](https://img.shields.io/badge/Project-Webpage-blue.svg)](https://viscom.uni-ulm.de/publications/from-natural-to-nanoscale-training-controlnet-on-scarce-fib-sem-data-for-augmenting-semantic-segmentation-data/)
[![ICCVW](https://img.shields.io/badge/ICCVW-2025-green.svg)]()
[![Paper](https://img.shields.io/badge/Paper-PDF-red.svg)](source/paper.pdf)

このリポジトリは以下の論文のコードを含んでいます：

```
From Natural to Nanoscale: Training ControlNet on Scarce FIB-SEM Data for Augmenting Semantic Segmentation Data
Hannah Kniesel*, Pascal Rapp*, Pedro Hermosilla, Timo Ropinski
ICCVW BIC
```

本コードは Lvmin Zhang と Maneesh Agrawala による公式 ControlNet [1] リポジトリをベースに拡張しています。基礎的な研究に感謝いたします。

オリジナル ControlNet リポジトリ: [https://github.com/lllyasviel/ControlNet](https://github.com/lllyasviel/ControlNet)


![Teaser](source/image.png)
*データセットからの代表的なサンプル。ControlNetの条件付けに使用されるカラーコード化されたセグメンテーションマスクと、
生成された合成画像を並べて表示しています。サンプル間（列）の視覚的な一貫性は、データセット内の変動性の低さを示しています。
実画像（1行目）と合成画像（3行目）の間には視覚的な違いがありますが、定量的な実験により、U-Netは合成サンプルから
有用な画像特徴を抽出できることが示されています。*

---
[1] Zhang, Lvmin, Anyi Rao, and Maneesh Agrawala. "Adding conditional control to text-to-image diffusion models." Proceedings of the IEEE/CVF international conference on computer vision. 2023.

## セットアップ

### ✅ 必要要件

- Python: `3.10.13`
- CUDA: `12.1`
- cuDNN: `8.9`
- PyTorch: `2.1.1`
- torchvision: `0.16.1`

---

### 🔧 オプション1: Docker

Dockerイメージを直接ダウンロードできます：
```bash
docker run --gpus all -v $PWD:/workspace -it hannahkniesel/natural2nanoscale:latest bash
```

または自分でビルド：
```bash
docker build -t natural2nanoscale .
docker run --gpus all -v $PWD:/workspace -it natural2nanoscale bash
```

Docker ≥ 20.10 と NVIDIA Container Toolkit がインストールされていることを確認してください。

---

### ⚙️ オプション2: Virtualenv + pip

```bash
python3 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r req.txt
```

---

## モデル重みとデータ

すべてのモデル重みは[こちら](https://viscom.datasets.uni-ulm.de/Natural2Nanoscale/Weights.zip)からダウンロードできます。

事前学習済みControlNetのみをダウンロードしたい場合は[こちら](https://viscom.datasets.uni-ulm.de/Natural2Nanoscale/ControlNet-Weights.zip)

生成画像と対応するマスクは[こちら](https://viscom.datasets.uni-ulm.de/Natural2Nanoscale/Generated.zip)からダウンロードできます。

実画像は Devan et al [2] の Dataset 1 から取得しています。データは[こちら](https://data.mendeley.com/datasets/9rdmnn2x4x/1)にあります。

---
*[2] Shaga Devan, Kavitha, et al. "Weighted average ensemble-based semantic segmentation in biological electron microscopy images." Histochemistry and Cell Biology 158.5 (2022): 447-462.*

## ControlNet の訓練

まず初期の sd1.5 チェックポイントをダウンロードする必要があります。2つの方法があります：
1. [こちら](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main)から `v1-5-pruned.ckpt` をダウンロードし、`models` ディレクトリに移動します。その後、`python tool_add_control.py ./models/v1-5-pruned.ckpt ./models/control_sd15_ini.ckpt` を実行してControlNetアーキテクチャ用にチェックポイントを準備します。

2. または、ControlNetモデル重み（上記参照）をダウンロードした場合、`ControlNet-Weights/control_sd15_ini.ckpt` のチェックポイントを使用できます。

次に、論文と同様のControlNetを訓練したい場合は、以下を実行します：
```bash
python train.py \
    --batch_size 2 \
    --learning_rate 1e-5 \
    --image_path /path/to/my/images \
    --mask_path /path/to/my/masks \
    --resume_path ./models/control_sd15_ini.ckpt \
    --gpus 1 \
    --precision 32 \
    --wandb_api_key YOUR_WANDB_KEY_HERE
```

*注意：現在のコードは3クラスのエンコーディングのみをサポートしています。*
モデル重みとログ画像はデフォルトで `./models/<timestamp>_...` に保存されます。`--output_root /custom/path` を渡すと、異なるベースディレクトリに保存できます。

統合型 RGBA ControlNet（マスク + Cannyエッジを1つのテンソルに）を訓練するには、まず条件付けテンソルを事前計算し（次のセクション参照）、`condition_type` を切り替えます：

```bash
python train.py \
    --condition_type rgba \
    --image_path /path/to/my/images \
    --rgba_path /path/to/my_rgba/train \
    --val_rgba_path /path/to/my_rgba/val \
    --resume_path ./models/control_sd15_ini.ckpt \
    --gpus 1
```

`condition_type=rgba` の場合、トレーナーは自動的に4チャンネルを期待するヒントブランチを持つControlNetをインスタンス化します。

---

## RGBA 条件付けの事前計算

クラスタジョブを起動する前に、`utils/build_rgba_dataset.py` を使用してマスクと新しく計算されたCannyエッジを `(H, W, 4)` テンソルに融合します：

```bash
python utils/build_rgba_dataset.py \
    --img_dir data/images/train \
    --mask_dir data/masks/train \
    --dest_dir data_rgba/train \
    --fmt npz \
    --canny-low 100 --canny-high 200 --beta-edge 1.0 \
    --preview-max 32
```

### RGBA ビルドオプション

| オプション | デフォルト | 説明 |
|---|---|---|
| `--img_dir` | (必須) | ソース画像を含むディレクトリ |
| `--mask_dir` | (必須) | マスク画像（グレースケールラベル）を含むディレクトリ |
| `--dest_dir` | (必須) | 出力 RGBA テンソルのベースディレクトリ |
| `--fmt` | `npz` | 出力フォーマット: `npz`、`png`、または両方 |
| `--preview_dir` | `<dest_dir>/preview` | プレビューパネルのディレクトリ |
| `--preview-max` | `32` | 保存するプレビューパネルの最大数 |
| `--num-mask-classes` | `3` | ワンホットエンコードするマスククラス数 |
| `--canny-low` | `100` | Cannyの下限ヒステリシス閾値 |
| `--canny-high` | `200` | Cannyの上限ヒステリシス閾値 |
| `--beta-edge` | `1.0` | 正規化前のエッジチャンネル乗数 |
| `--overwrite` | `False` | 既存の出力を上書き |

スクリプトは圧縮された `.npz` テンソル（R/G/BチャンネルにマスクRGB、Aチャンネルにエッジ）とオプションの `preview/*.png` トリプティッチ（オリジナル/マスク/エッジ）を保存し、訓練前にアライメントを視覚的に確認できます。`--dest_dir` はベースフォルダとして機能し、出力は `<dest_dir>/<canny_low>_<canny_high>` に書き込まれるため、各閾値ペアが自動的に分離されます。

---

## 画像生成

事前学習済みControlNetで画像を生成するには：
```bash
python generate.py \
    --config_yaml_path ./models/cldm_v15.yaml \
    --model_weights_path ./models/EM_best_results.ckpt \
    --mask_dir /path/to/my/segmentation/masks \
    --output_base_dir ./my_synth_data \
    --n_augmentations_per_mask 1 \
    --batch_size_per_inference 1 
```

シングルブランチ RGBA モデルの場合、事前計算された RGBA テンソルを指定し、新しいモードを有効にします：

```bash
python generate.py \
    --generation_mode rgba \
    --rgba_dir data_rgba/val \
    --config_yaml_path ./models/cldm_v15.yaml \
    --rgba_model_path ./models/rgba_controlnet.ckpt \
    --output_base_dir ./my_synth_data_rgba
```

訓練に使用した同じ RGBA テンソルを推論時に再利用でき、訓練/テストパイプライン間の一貫性が保証されます。

---

## 訓練オプションリファレンス

`train.py` スクリプトは多くのコマンドラインオプションをサポートしています：

### コア訓練パラメータ

| オプション | デフォルト | 説明 |
|---|---|---|
| `--batch_size` | `2` | 訓練用バッチサイズ |
| `--learning_rate` | `1e-5` | オプティマイザの学習率 |
| `--logger_freq` | `300` | WandBへの画像ログ頻度（バッチ単位） |
| `--sd_locked` | `True` | 訓練中にStable Diffusionバックボーンをロック |
| `--only_mid_control` | `False` | ControlNetでミッドブロック制御のみを使用 |
| `--gpus` | `1` | GPU数（0=CPU、-1=すべて利用可能） |
| `--precision` | `32` | 浮動小数点精度（16または32） |
| `--num_workers` | `0` | データ読み込みワーカー数 |
| `--output_root` | `./models` | チェックポイントとログの親ディレクトリ |

### データパスオプション

| オプション | デフォルト | 説明 |
|---|---|---|
| `--image_path` | `data/EM-Dataset/train_images` | 訓練画像のパス |
| `--mask_path` | `data/EM-Dataset/train_masks` | 訓練マスクのパス |
| `--edge_path` | `None` | 訓練エッジマップのパス |
| `--rgba_path` | `None` | 訓練 RGBA テンソルのパス |
| `--val_image_path` | `None` | 検証画像のパス |
| `--val_mask_path` | `None` | 検証マスクのパス |
| `--val_edge_path` | `None` | 検証エッジマップのパス |
| `--val_rgba_path` | `None` | 検証 RGBA テンソルのパス |
| `--rgba_alpha_scale` | `1.0` | アルファ（エッジ）チャンネルのスケーリング係数 |

### モデル設定

| オプション | デフォルト | 説明 |
|---|---|---|
| `--resume_path` | `./models/control_sd15_ini.ckpt` | 再開するチェックポイント |
| `--cldm_config_path` | `./models/cldm_v15.yaml` | ControlNetモデル設定YAML |
| `--condition_type` | `segmentation` | 条件モダリティ: `segmentation`、`edge`、または `rgba` |

### WandB 設定

| オプション | デフォルト | 説明 |
|---|---|---|
| `--wandb_project` | `EM-ControlNet` | WandBプロジェクト名 |
| `--wandb_api_key` | `INSERT KEY` | WandB APIキー（または `WANDB_API_KEY` 環境変数を使用） |

### 検証 FID オプション

`--enable_val_fid` で訓練中の CEM FID 計算を有効にします：

| オプション | デフォルト | 説明 |
|---|---|---|
| `--enable_val_fid` | `False` | 各エポック後に検証スプリットでCEM FIDを計算 |
| `--fid_batch_size` | `2` | 検証生成とFID特徴抽出のバッチサイズ |
| `--fid_num_workers` | `0` | 検証データローダーのワーカー数 |
| `--fid_ddim_steps` | `50` | 検証画像生成のDDIMステップ数 |
| `--fid_guidance_scale` | `9.0` | 検証用の分類器フリーガイダンススケール |
| `--fid_eta` | `0.0` | 検証生成のDDIM eta値 |
| `--fid_control_strength` | `1.0` | 検証生成のコントロール強度乗数 |
| `--fid_backbone` | `cem500k` | CEM FIDのバックボーン（`cem500k`または`cem1.5m`） |
| `--fid_image_size` | `512` | CEMバックボーンが期待する入力サイズ |
| `--fid_device` | `cuda` | CEM FID特徴抽出のデバイス |
| `--fid_weights_path` | `None` | 事前ダウンロードしたCEM重みへのローカルパス（オプション） |
| `--fid_download_dir` | `None` | ダウンロードしたCEM重みをキャッシュするディレクトリ（オプション） |
| `--fid_seed` | `1234` | 検証プロンプトサンプリングの乱数シード |

---

## 生成オプションリファレンス

`generate.py` スクリプトは画像生成の詳細な制御のための豊富なオプションを提供します：

### モデルパスオプション

| オプション | デフォルト | 説明 |
|---|---|---|
| `--config_yaml_path` | `./models/cldm_v15.yaml` | ControlNetモデル設定YAMLへのパス |
| `--model_weights_path` | `./models/EM_best_results.ckpt` | ControlNetモデル重みチェックポイントへのパス |
| `--mask_model_path` | `None` | マスク条件付きControlNetへのパス（`--model_weights_path`がデフォルト） |
| `--edge_model_path` | `None` | 単一エッジControlNetチェックポイント（非推奨、`--edge_model_paths`を使用） |
| `--edge_model_paths` | `[]` | エッジControlNetチェックポイント（`<name>=/path/to/model.ckpt`形式） |
| `--rgba_model_path` | `None` | RGBA ControlNetのチェックポイント（`--model_weights_path`がデフォルト） |

### 条件ディレクトリオプション

| オプション | デフォルト | 説明 |
|---|---|---|
| `--mask_dir` | (マスクモードで必須) | 入力マスク画像へのパス |
| `--edge_dir` | `None` | 事前計算されたエッジ画像へのパス |
| `--edge_dirs` | `[]` | エッジディレクトリ（`<name>=/path/to/edges`形式） |
| `--rgba_dir` | `None` | RGBA npz/png コントロールテンソルを含むディレクトリ |

### サンプリングパラメータ

| オプション | デフォルト | 説明 |
|---|---|---|
| `--ddim_steps` | `70` | DDIMサンプリングステップ数 |
| `--strength` | `2.0` | ControlNet条件付け強度 |
| `--scale` | `9.0` | 分類器フリーガイダンススケール |
| `--seed` | `-1` | 乱数シード（`-1`で毎回ランダム） |
| `--eta` | `1.0` | 確率性のためのDDIM etaパラメータ |
| `--guess_mode` | `False` | ゲスモードを有効化（より緩い条件付け） |

### マルチ条件強度制御

| オプション | デフォルト | 説明 |
|---|---|---|
| `--mask_strength` | `1.0` | マスクControlNetブランチの相対強度 |
| `--edge_strengths` | `[]` | エッジブランチの相対強度（`<name>=<float>`形式） |
| `--skip_missing_edges` | `False` | エッジファイルが見つからないサンプルをエラーではなくスキップ |

### 生成モード

`--generation_mode` で使用する条件ブランチを選択：

| モード | 説明 |
|---|---|
| `mask_only` | マスク条件付けのみを使用 |
| `edge_only` | エッジ条件付けのみを使用 |
| `mask_and_edge` | マスクとエッジ両方の条件付けを使用（マルチ条件ControlNet） |
| `rgba` | 統合RGBA条件付けを使用（単一4チャンネルブランチ） |

### 出力オプション

| オプション | デフォルト | 説明 |
|---|---|---|
| `--output_base_dir` | `my_synth_data` | 生成画像とマスクのベースディレクトリ |
| `--n_augmentations_per_mask` | `1` | 入力マスクあたりの合成画像数 |
| `--batch_size_per_inference` | `1` | 推論呼び出しあたりのサンプル数 |

---

## マルチ条件 ControlNet

`generate.py` スクリプトには `MultiConditionControlNet` クラスが含まれており、複数のモダリティ（例：セグメンテーションマスクとCannyエッジ）で同時に条件付けできます。これは公式ControlNetのマルチ条件付け設計に従っています。

### 例：マスク + エッジ生成

```bash
python generate.py \
    --generation_mode mask_and_edge \
    --config_yaml_path ./models/cldm_v15.yaml \
    --mask_model_path ./models/mask_controlnet.ckpt \
    --edge_model_paths canny=./models/edge_controlnet.ckpt \
    --mask_dir /path/to/masks \
    --edge_dirs canny=/path/to/canny_edges \
    --mask_strength 1.0 \
    --edge_strengths canny=1.0 \
    --output_base_dir ./my_synth_data_multi \
    --ddim_steps 50 \
    --scale 9.0
```

これはマスクとエッジ用の別々のControlNetブランチをロードし、生成中にそれらのコントロール信号を集約します。

---

## FID / KID 計算

FIDとKIDメトリクスの計算に関する詳細なドキュメントは [fid/README_ja.md](fid/README_ja.md) を参照してください。

### クイックスタート

```bash
# EM画像用のCEM FID
python fid/compute_cem_fid.py /path/to/real /path/to/generated --backbone cem500k

# 標準的なInception FID
python fid/compute_normal_fid.py /path/to/real /path/to/generated
```

---

## プロジェクト構造

```
Natural2Nanoscale/
├── train.py              # ControlNetのメイン訓練スクリプト
├── generate.py           # マルチ条件サポート付き画像生成スクリプト
├── tool_add_control.py   # SDチェックポイントをControlNet用に準備するユーティリティ
├── config.py             # グローバル設定（例：save_memoryフラグ）
├── dataset.py            # 訓練用PyTorch Dataset実装
├── share.py              # 共有インポートとセットアップ
│
├── annotator/            # 条件抽出器
│   ├── canny/            # Cannyエッジ検出器
│   ├── hed/              # HEDエッジ検出器
│   ├── midas/            # MiDaS深度推定器
│   ├── mlsd/             # M-LSD線検出器
│   ├── openpose/         # OpenPoseボディキーポイント検出器
│   └── uniformer/        # Uniformerセマンティックセグメンテーション
│
├── cldm/                 # ControlNetモデル実装
│   ├── cldm.py           # ControlNetアーキテクチャ
│   ├── ddim_hacked.py    # ControlNet用に修正されたDDIMサンプラー
│   ├── hack.py           # モデルハッキングユーティリティ
│   ├── logger.py         # 訓練用画像ロギング
│   └── model.py          # モデル作成とロードユーティリティ
│
├── ldm/                  # Latent Diffusion Modelベースコード
│   ├── data/             # データユーティリティ
│   ├── models/           # Diffusionモデルアーキテクチャ
│   └── modules/          # ニューラルネットワークモジュール（attention、encodersなど）
│
├── fid/                  # FID/KID計算ツール
│   ├── compute_cem_fid.py    # EM画像用CEM ResNet50ベースFID
│   ├── compute_normal_fid.py # ImageNet Inception v3ベースFID
│   ├── pretraining/      # CEM事前学習ユーティリティ（MoCo v2、SwAV）
│   └── README_ja.md      # 詳細なFIDツールドキュメント
│
├── utils/                # ユーティリティスクリプト
│   └── build_rgba_dataset.py  # RGBA条件付けテンソルの事前計算
│
├── models/               # モデルチェックポイントと設定
│   ├── cldm_v15.yaml     # ControlNetモデル設定
│   ├── control_sd15_ini.ckpt  # ControlNet用初期SD1.5チェックポイント
│   └── saved/            # 保存された訓練実行
│
├── demo/                 # デモ画像（エッジ、マスク）
└── my_synth_data/        # 生成画像のデフォルト出力ディレクトリ
```

### 主要ファイル

| ファイル | 説明 |
|---|---|
| `config.py` | 低VRAMh環境用の `save_memory` フラグを含むグローバル設定 |
| `dataset.py` | 拡張付きで画像/マスク/エッジ/RGBAペアを読み込むカスタムPyTorch Dataset |
| `share.py` | スクリプト間で共有される共通インポートと初期化 |
| `cldm/model.py` | RGBA（4チャンネル）ControlNet用の設定オーバーライド付きモデル作成 |

---

## 引用

この研究でこのコードを使用する場合は、以下を引用してください：

```bibtex
@inproceedings{kniesel2025natural,
  title={From Natural to Nanoscale: Training ControlNet on Scarce FIB-SEM Data for Augmenting Semantic Segmentation Data},
  author={Kniesel, Hannah and Rapp, Pascal and Hermosilla, Pedro and Ropinski, Timo},
  booktitle={ICCVW BIC},
  year={2025}
}
```

---

## ライセンス

このプロジェクトは [LICENSE](LICENSE) ファイルの条項に基づいてライセンスされています。
