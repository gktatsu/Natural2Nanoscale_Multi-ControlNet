# FID / KID 計算ツールの使い方

このディレクトリには FID（Fréchet Inception Distance）および KID（Kernel Inception Distance）を計算するためのユーティリティスクリプトが含まれます。主に次の 2 つのスクリプトを提供しています。

| スクリプト | 特徴抽出器 | 用途 |
|---|---|---|
| `compute_cem_fid.py` | CEM 事前学習済み ResNet50（CEM500K / CEM1.5M） | EM 画像向け FID/KID 計算 |
| `compute_normal_fid.py` | ImageNet 学習済み Inception v3 | 一般的な FID/KID 計算 |

---

## 共通の前提（依存ライブラリ）

両スクリプトで必要な Python パッケージ:

- `torch`
- `torchvision`
- `numpy`
- `scipy`
- `tqdm`
- `Pillow`

未インストールの場合は、仮想環境を有効にした上で次を実行してください。

```bash
pip install torch torchvision numpy scipy tqdm Pillow
```

---

## compute_cem_fid.py（CEM ResNet50 を使う）

### 概要

`compute_cem_fid.py` は CEM500K (MoCo v2) または CEM1.5M (SwAV) の事前学習済み ResNet50 を特徴抽出器として用い、2 つの EM 画像フォルダ間の FID を算出します。オプションで KID も推定できます。

#### 処理フロー

1. グレースケールの EM 画像を 3 チャンネル（同一値の複製）に変換
2. 指定解像度（デフォルト 224×224）にリサイズ
3. CEM 事前学習時と同じ正規化（`cem500k`: mean=0.573, std=0.127 / `cem1.5m`: mean=0.576, std=0.128）を適用
4. ResNet50 のグローバル平均プーリング層から 2048 次元の特徴ベクトルを抽出
5. 特徴ベクトルから平均・共分散行列を計算し FID を算出

### 基本的な使い方

```bash
python fid/compute_cem_fid.py REAL_DIR GEN_DIR [オプション]
```

- `REAL_DIR`: 実画像が入ったディレクトリ
- `GEN_DIR`: 生成画像が入ったディレクトリ

### 全オプション一覧

| オプション | 既定値 | 説明 |
|---|---|---|
| `--backbone {cem500k, cem1.5m}` | `cem500k` | 使用する CEM 事前学習モデル（MoCo v2 または SwAV） |
| `--batch-size INT` | `32` | 特徴抽出時のバッチサイズ |
| `--num-workers INT` | `4` | DataLoader のワーカープロセス数 |
| `--device` | 自動 (`cuda` / `cpu`) | 推論デバイス |
| `--image-size INT` | `224` | 入力をリサイズするサイズ |
| `--weights-path PATH` | なし | 手動でダウンロードした重みファイルのパス |
| `--download-dir PATH` | なし | 重みのキャッシュ先ディレクトリ |
| `--output-json PATH` | `cem_fid.json` | 結果保存先（タイムスタンプが自動付与） |
| `--data-volume STR` | なし | 実行環境メモ（Docker マウント情報等）を記録 |
| `--compute-kid` | 無効 | 指定すると KID も計算 |
| `--kid-subset-size INT` | `1000` | KID サブセットあたりのサンプル数 |
| `--kid-subset-count INT` | `100` | KID のサブセット試行回数 |
| `--seed INT` | `42` | KID 用乱数シード |

### 実行例

```bash
# 基本的な使用例
python fid/compute_cem_fid.py /data/real /data/generated --backbone cem500k

# KID も計算する場合
python fid/compute_cem_fid.py /data/real /data/generated \
    --backbone cem1.5m \
    --compute-kid \
    --batch-size 64 \
    --output-json results/cem_fid_result.json

# ローカルの重みファイルを使用する場合
python fid/compute_cem_fid.py /data/real /data/generated \
    --weights-path ./fid/weights/cem500k_mocov2_resnet50_200ep.pth.tar
```

### 出力

- **コンソール出力**: FID 値、使用したバックボーン、画像数、重みのソース等を表示
- **JSON ファイル**: 以下の情報を保存
  - `fid`: FID 値
  - `backbone`: 使用したバックボーン名
  - `weights`: 重みファイルのソース（URL またはローカルパス）
  - `num_real`, `num_generated`: 評価した画像数
  - `image_size`: 入力解像度
  - `normalization_mean`, `normalization_std`: 正規化パラメータ
  - `timestamp_utc`: 実行日時（UTC）
  - `real_dir`, `gen_dir`: 入力ディレクトリの絶対パス
  - `kid`, `kid_std`: KID の平均と標準誤差（`--compute-kid` 指定時のみ）

### 重みファイルのダウンロード

重みファイルは初回実行時に Zenodo から自動ダウンロードされます。オフライン環境では以下から手動でダウンロードしてください。

| バックボーン | ダウンロード URL |
|---|---|
| CEM500K (MoCo v2) | https://zenodo.org/record/6453140/files/cem500k_mocov2_resnet50_200ep.pth.tar |
| CEM1.5M (SwAV) | https://zenodo.org/record/6453160/files/cem15m_swav_resnet50_200ep.pth.tar |

ダウンロード後、`--weights-path` オプションでファイルパスを指定してください。

---

## compute_normal_fid.py（ImageNet Inception v3 を使う）

### 概要

`compute_normal_fid.py` は torchvision の ImageNet 学習済み Inception v3 (`IMAGENET1K_V1`) を用いて、実画像群と生成画像群の FID（およびオプションで KID）を算出します。標準的な Inception ベースの FID 評価を行います。

#### 処理フロー

1. 画像を RGB に変換（グレースケール画像は自動変換）
2. 299×299 にリサイズ
3. ImageNet 正規化（mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]）を適用
4. Inception v3 の fc 層直前から 2048 次元の特徴ベクトルを抽出
5. 特徴ベクトルから平均・共分散行列を計算し FID を算出

### 基本的な使い方

```bash
python fid/compute_normal_fid.py REAL_DIR GEN_DIR [オプション]
```

### 全オプション一覧

| オプション | 既定値 | 説明 |
|---|---|---|
| `--batch-size INT` | `32` | 特徴抽出時のバッチサイズ |
| `--num-workers INT` | `4` | DataLoader のワーカープロセス数 |
| `--device` | 自動 (`cuda` / `cpu`) | 推論デバイス |
| `--image-size INT` | `299` | Inception v3 が期待する入力解像度 |
| `--output-json PATH` | `inception_fid.json` | 結果保存先（タイムスタンプが自動付与） |
| `--data-volume STR` | なし | 実行環境メモ（Docker マウント情報等）を記録 |
| `--compute-kid` | 無効 | 指定すると KID も計算 |
| `--kid-subset-size INT` | `1000` | KID サブセットあたりのサンプル数 |
| `--kid-subset-count INT` | `100` | KID のサブセット試行回数 |
| `--seed INT` | `42` | KID 用乱数シード |

### 実行例

```bash
# 基本的な使用例
python fid/compute_normal_fid.py /data/real /data/generated

# KID も計算する場合
python fid/compute_normal_fid.py /data/real /data/generated \
    --compute-kid \
    --batch-size 64 \
    --output-json results/inception_fid_result.json
```

---

## 前処理の違い（両スクリプトの比較）

| 項目 | compute_cem_fid.py | compute_normal_fid.py |
|---|---|---|
| **バックボーン** | CEM ResNet50 | ImageNet Inception v3 |
| **入力チャンネル** | グレースケール→3ch 複製 | RGB（グレースケールは自動変換） |
| **入力解像度** | 224×224 | 299×299 |
| **正規化** | CEM 事前学習時の値 | ImageNet 標準値 |
| **特徴次元** | 2048 | 2048 |
| **推奨用途** | EM / 電子顕微鏡画像 | 一般的な自然画像 |

---

## 訓練時の自動 FID 評価

`train.py` スクリプトでは `--enable_val_fid` オプションを指定することで、訓練中に各エポック終了時に自動で CEM FID を計算できます。

### 関連オプション

| オプション | 既定値 | 説明 |
|---|---|---|
| `--enable_val_fid` | `False` | 検証時に CEM FID を計算 |
| `--fid_batch_size` | `2` | FID 計算時のバッチサイズ |
| `--fid_num_workers` | `0` | FID 計算時のワーカー数 |
| `--fid_ddim_steps` | `50` | 検証用画像生成の DDIM ステップ数 |
| `--fid_guidance_scale` | `9.0` | 検証用画像生成のガイダンススケール |
| `--fid_eta` | `0.0` | 検証用 DDIM eta 値 |
| `--fid_control_strength` | `1.0` | 検証用 ControlNet 強度 |
| `--fid_backbone` | `cem500k` | CEM バックボーン（`cem500k` / `cem1.5m`） |
| `--fid_image_size` | `512` | CEM バックボーンの入力サイズ |
| `--fid_device` | `cuda` | FID 計算デバイス |
| `--fid_weights_path` | なし | CEM 重みのローカルパス |
| `--fid_download_dir` | なし | CEM 重みのキャッシュ先 |
| `--fid_seed` | `1234` | 検証時の乱数シード |

### 使用例

```bash
python train.py \
    --batch_size 2 \
    --learning_rate 1e-5 \
    --image_path /data/images/train \
    --mask_path /data/masks/train \
    --val_image_path /data/images/val \
    --val_mask_path /data/masks/val \
    --enable_val_fid \
    --fid_backbone cem500k \
    --resume_path ./models/control_sd15_ini.ckpt \
    --gpus 1
```

---

## ベストプラクティス

1. **画像ディレクトリの準備**: 評価対象の画像のみを含むディレクトリを指定してください（スクリプトは再帰的に画像を探索します）
2. **対応フォーマット**: `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.bmp` がサポートされています
3. **GPU メモリ**: `--batch-size` を増やすと高速化できますが、メモリ不足に注意してください
4. **KID 計算**: `--compute-kid` を有効にすると全特徴をメモリに保持するため、大規模データセットでは注意が必要です
5. **再現性**: `--seed` オプションで KID のサンプリングを固定できます

---

## Docker 利用例

```bash
# CEM FID の計算
docker run --gpus all --rm \
  -v /path/to/data:/data \
  -v /path/to/weights:/weights \
  -v /path/to/results:/results \
  hannahkniesel/natural2nanoscale:latest \
  python fid/compute_cem_fid.py \
    /data/real /data/generated \
    --backbone cem500k \
    --weights-path /weights/cem500k_mocov2_resnet50_200ep.pth.tar \
    --output-json /results/cem_fid.json \
    --data-volume /path/to/data:/data
```

---

## 事前学習ディレクトリ（pretraining/）

`pretraining/` ディレクトリには CEM バックボーンの事前学習用コードが含まれています。

| サブディレクトリ | 内容 |
|---|---|
| `mocov2/` | MoCo v2 による ResNet50 の事前学習（CEM500K 用） |
| `swav/` | SwAV による ResNet50 の事前学習（CEM1.5M 用） |

これらは FID 計算に必要な ResNet50 アーキテクチャの定義を提供しています。

---

## トラブルシューティング

### 重みのダウンロードに失敗する

オフライン環境や Zenodo へのアクセスが制限されている場合：

1. 上記の URL から手動で重みファイルをダウンロード
2. `fid/weights/` ディレクトリに配置（スクリプトが自動検出）
3. または `--weights-path` オプションで明示的に指定

### CUDA out of memory

- `--batch-size` を小さくする（例: 16 → 8）
- `--device cpu` で CPU モードを使用

### 画像が見つからない

- 対応拡張子（`.png`, `.jpg` 等）を確認
- ディレクトリパスが正しいか確認
- サブディレクトリも再帰的に探索されます

---

## 参考文献

- CEM 事前学習: [Zenodo - CEM500K](https://zenodo.org/record/6453140), [Zenodo - CEM1.5M](https://zenodo.org/record/6453160)
- FID 論文: Heusel et al., "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium", NeurIPS 2017
- KID 論文: Bińkowski et al., "Demystifying MMD GANs", ICLR 2018
