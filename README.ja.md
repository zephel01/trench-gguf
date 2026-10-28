<div align="center">

# trench-gguf

**GGUF の i-quant（IQ1_S、IQ1_M、IQ2_XXS、IQ2_S）を GPU で量子化する。結果は llama-quantize とバイト単位で同じ。**

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Platform](https://img.shields.io/badge/platform-Linux-lightgrey)
![ROCm](https://img.shields.io/badge/ROCm-checked-ED1C24?logo=amd&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-builds%20(unchecked)-76B900?logo=nvidia&logoColor=white)
![GPU](https://img.shields.io/badge/tested-AI%20PRO%20R9700-ED1C24)
![llama.cpp](https://img.shields.io/badge/llama.cpp-ac8e1694d-4D6BFE)

[English](README.md) · **日本語**

[はじめかた](#-はじめかた) · [実例](#-実例-deepseek-v41-flash-を一から作る) · [使い方](#️-使い方) · [速さ](#-速さ) · [仕組み](#-仕組み) · [制限](#️-制限) ·
[謝辞](#-謝辞)

</div>

---

llama.cpp は、i-quant の imatrix 付き量子化を CPU でしか行いません。そのうえ時間がかかります。
- 重み 8 個ごとに、符号帳の点（256〜2,048 個）を総当たりで比べます。
- それを、尺度を何通りも変えて繰り返します。

大きな MoE モデルのエキスパートを量子化し直すと、32 スレッドで約 1 時間かかります。DeepSeek V4.1 Flash なら、エキスパートの重みは 5,440 億個あります。層ごとの配分を変えて試すたびに、この時間がかかります。

trench-gguf は、この量子化をそのまま GPU で行います。
- GPU の処理は、ggml の `quantize_row_*_impl` を 1 行ずつ写したものです。
- 探索に使う表も、ggml 自身が書き出したものを使います。
- そのため、出力は **`llama-quantize` が書くものとバイト単位で同じ**です。探索を手抜きしていないので、品質も変わりません。

DeepSeek V4.1 のエキスパート全部を、AI PRO R9700 で 15 分半で量子化し直しました。

> [!NOTE]
> trench-gguf は [Abyss](https://github.com/zephel01/Abyss) から生まれました。Abyss は、DeepSeek V4 / V4.1 Flash を家庭用の GPU 1 枚で動かすエンジンで、そこで層ごとの配分を探すのに使っています。trench-gguf を使うのに Abyss は要りません。

## ✨ 特長

- **llama.cpp とバイト単位で同じ**
  - IQ2_XXS / IQ2_S / IQ1_S / IQ1_M のすべてのブロックが、`ggml_quantize_chunk` の出力と一致します。
  - DeepSeek V4.1 のエキスパートと、乱数の重みで確かめました。
  - 1 層を量子化し直した GGUF も、`cmp` でまったく同じでした。
- **CPU 32 スレッドの 5 倍、1 スレッドの 50〜100 倍**
  - エキスパート 384 個のテンソル 1 本が、27 秒から約 5 秒になりました。
- **エキスパートだけ量子化し直す（`tools/requant_experts.py`）**
  - 元にできるのは、ggml が読める GGUF なら何でもかまいません（MXFP4、i-quant、k-quant）。
  - 正規表現の規則ファイルに従い、imatrix を使って MoE のエキスパートだけを量子化し直します。
  - それ以外のテンソルはそのままコピーします。
- **2 つの GGUF を層ごとに組み合わせる（`tools/merge_experts.py`）**
  - 層ごとに、どちらかの GGUF のエキスパートをバイト単位でそのまま使います。
  - 公開するファイルが、測った組み合わせとまったく同じになります。
- **セットアップは 1 本**
  - `./setup.sh` だけで準備が終わります。
  - llama.cpp の取得、`libggml-base` だけのビルド、GPU の判別、ライブラリのビルド、ggml との一致の自己テストまで行います。
- **ROCm と CUDA を同じソースから**ビルドできます。

## 🚀 はじめかた

```bash
cd ~
git clone https://github.com/zephel01/trench-gguf
cd trench-gguf
./setup.sh                 # llama.cpp、libggml-base、GPU 用ライブラリ、Python の venv、自己テスト
source ~/trench-gguf/trench.env    # 新しいターミナルを開くたびに
```

`setup.sh` の最後に自己テストが走ります。`OK: byte-identical to ggml` と出れば、GPU の出力があなたの llama.cpp と一致しています。

オプション：

| オプション | 内容 |
| --- | --- |
| `--llama <フォルダ>`、`--llama-build <フォルダ>` | 手元にある llama.cpp を使う |
| `--backend hip\|cuda`、`--arch gfx1201\|sm_120` | GPU の自動判別を上書きする |
| `--ref <git の ref>` | llama.cpp の版を固定する |

### 必要なもの

| 必要なもの | 確かめた版 |
| --- | --- |
| Linux x86-64 | Ubuntu、glibc 2.39 |
| git、cmake、C コンパイラ、venv が使える Python 3 | Ubuntu 標準のもの |
| ROCm（`hipcc`、`rocminfo`）**または** CUDA（`nvcc`、`nvidia-smi`） | HIP 7.15（ROCm 10 / TheRock）。CUDA 13.3 はビルドのみ |
| GPU | AI PRO R9700（gfx1201）。RTX 50 系（sm_120）は CUDA 12.8 以降 |
| GPU のメモリ | 数百 MB |

### マシンに必要なもの

ツール自体は小さいですが、扱うモデルは大きいです。[手順の実例](#-実例-deepseek-v41-flash-を一から作る)で必要なのは次のとおりです。

| 必要なもの | 量 | 理由 |
| --- | --- | --- |
| ディスクの空き | **約 650 GB**（手順 6 もやるなら +350 GB） | 元の GGUF 501 GB + 出力 141 GB |
| ディスクの種類 | NVMe SSD（HDD でも動くが遅い） | 290 GB を読み、141 GB を書く |
| メモリ（RAM） | 32 GB 以上（目安） | GGUF をメモリに割り当てて読む。CPU のスレッドごとに、エキスパート 1 個分の float（47 MB）を持つ |
| CPU | x86-64 なら何でも。コアが多いほど速い | 元の重みを float に戻す処理を、全スレッドで行う |
| GPU | ROCm か CUDA の GPU 1 枚。空きメモリ 1 GB | 量子化そのもの。確かめたのは AI PRO R9700 |
| 回線 | 速いほど良い | 501 GB は、1 Gbps で約 1 時間 10 分、100 Mbps で約 11 時間 |
| ツール本体 | 約 750 MB | trench-gguf + llama.cpp + Python の venv |

確かめたマシン：Ryzen 16 コア / 32 スレッド、メモリ 94 GB、AI PRO R9700、Gen4 の NVMe SSD。

## 📘 実例: DeepSeek V4.1 Flash を一から作る

下のコマンドは、どれもそのままコピペで動きます。

作るもの：DeepSeek V4.1 Flash（1017）で、**すべてのエキスパートを unsloth の UD-IQ2_XXS より 1 段低い型**にしたものです。
- 元にするのは unsloth の UD-Q4_K_XL です。エキスパートが配布元の MXFP4 のまま入っています。
- imatrix は unsloth のものを使います。

**手順 0：ツールを用意する**（最初の 1 回だけ。[はじめかた](#-はじめかた)を済ませていれば不要）

```bash
cd ~
git clone https://github.com/zephel01/trench-gguf
cd trench-gguf
./setup.sh
```

**手順 1：作業フォルダを作る。** 空きが約 650 GB あるディスクに作ります。`df -h .` の `Avail`（空き）で確かめます。

```bash
source ~/trench-gguf/trench.env
mkdir -p ~/gguf-work/DeepSeek-V4.1
cd ~/gguf-work/DeepSeek-V4.1
df -h .
```

大きいディスクが別の場所（たとえば `/mnt/data`）にあるなら、代わりにこちらを使います。
`mkdir -p /mnt/data/gguf-work/DeepSeek-V4.1 && cd /mnt/data/gguf-work/DeepSeek-V4.1`

**手順 2：元の GGUF（501 GB）と imatrix をダウンロードする。**
- `hf` コマンドは `setup.sh` が入れています。
- 途中で止まったら、同じコマンドをもう一度流すと続きから再開します。

```bash
hf download unsloth/DeepSeek-V4.1-Flash-1017-GGUF --revision 85f81fb970add013033f25e94006713ca2c0aaf2 \
  --local-dir . --include "UD-Q4_K_XL/*" --include "imatrix_unsloth.gguf_file"
```

**手順 3：規則ファイルを作る。** エキスパートのテンソルごとに、どの型にするかを書きます（0〜9 層の gate/up は IQ1_S、など）。

```bash
cat > rules-down1.txt <<'RULES'
blk\.[0-9]\.ffn_(gate|up)_exps\.weight              iq1_s
blk\.[0-9]\.ffn_down_exps\.weight                   iq1_m
blk\.(1[0-9]|2[0-4])\.ffn_(gate|up)_exps\.weight    iq1_m
blk\.1[0-9]\.ffn_down_exps\.weight                  iq2_xxs
blk\.2[0-4]\.ffn_down_exps\.weight                  iq2_s
blk\.(2[5-9]|3[0-9])\.ffn_(gate|up)_exps\.weight    iq2_xxs
blk\.(2[5-9]|3[0-4])\.ffn_down_exps\.weight         iq2_s
blk\.3[5-9]\.ffn_down_exps\.weight                  iq2_xxs
RULES
```

**手順 4：試しに流す**（何も書き出しません）。最後の行が `re-quantized 268.95 GiB -> 122.61 GiB (check only)` なら正しく準備できています。

```bash
python ~/trench-gguf/tools/requant_experts.py \
  UD-Q4_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q4_K_XL-00001-of-00009.gguf down1 rules-down1.txt \
  --imatrix=imatrix_unsloth.gguf_file --check
```

**手順 5：本番。** 確かめたマシンで 15 分半でした。エキスパートのテンソル 1 本ごとに、行末が `(gpu)` の行が出ます。

```bash
python ~/trench-gguf/tools/requant_experts.py \
  UD-Q4_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q4_K_XL-00001-of-00009.gguf down1 rules-down1.txt \
  --imatrix=imatrix_unsloth.gguf_file --gpu
ls -lh down1
```

`down1/` に、元と同じ名前のシャードが 9 個できます（シャード 8 を除いて 141 GB）。
- シャード 8 は Engram の表（203 GB）で、エキスパートを含みません。
- これは元のファイルへのハードリンクなので、同じディスクなら容量を使いません。

**手順 6（任意）：層ごとに組み合わせる。** unsloth の UD-IQ2_XXS（301 GB）を土台にして、0〜3 層だけを `down1` から取ります。

```bash
hf download unsloth/DeepSeek-V4.1-Flash-1017-GGUF --revision 85f81fb970add013033f25e94006713ca2c0aaf2 \
  --local-dir . --include "UD-IQ2_XXS/*"
python ~/trench-gguf/tools/merge_experts.py \
  UD-IQ2_XXS/DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS-00001-of-00006.gguf \
  down1/DeepSeek-V4.1-Flash-1017-UD-Q4_K_XL-00001-of-00009.gguf \
  mix DeepSeek-V4.1-Flash-1017-mix 0-3
ls -lh mix
```

書き直されるのは、0〜3 層を含むシャード 2（約 49 GB）だけです。ほかのシャードはハードリンクです。

**うまくいかないとき**

| 出たメッセージ | どうするか |
| --- | --- |
| `No space left on device` | 空きを作るか、作業フォルダを大きいディスクに移す（手順 1） |
| `hf: command not found`、`python: ... numpy` | そのターミナルで `source ~/trench-gguf/trench.env` を流す |
| `gpuq_init failed` | GPU が見えていない。`rocminfo` か `nvidia-smi` に GPU が出るか確かめ、`./setup.sh` をもう一度流す |
| `setup.sh` の自己テストで `MISMATCH` | `--gpu` を使わない。`setup-test.log` を付けて issue を立てる |

## 🧩 対応するモデル

trench-gguf が量子化し直すのは、**MoE モデルのエキスパート**です。llama.cpp の MoE の GGUF にある、3 次元の `ffn_gate_exps` / `ffn_up_exps` / `ffn_down_exps`（または、まとまった `ffn_gate_up_exps`）が対象です。元の型は、ggml が読めるものなら何でも構いません（BF16、Q8_0、MXFP4、k-quant、i-quant）。2026-10-29 に、手元の GGUF（約 300 ファイルの見出しと、下の実行）で確かめました。

| モデル（llama.cpp のアーキテクチャ） | エキスパート（行の長さ ne0） | unsloth の imatrix | 確かめたこと | 状態 |
| --- | --- | --- | --- | --- |
| DeepSeek V4.1 Flash 1017（`deepseek41`） | gate/up 5120、down 2304、384 個 | あり | 通しで実行、15 分半 | ◎ |
| Qwen3.8-Flash-Next（`qwen4exp`） | gate/up 2560、**down 640**、512 個 | あり | 通しで実行（119.5 → 40.4 GiB）、llama.cpp で読み込んで答えることを確認、GPU の一致 100% | ◎（down は IQ4_NL） |
| GLM-5.3-Flash（`glm5-next`） | gate/up 4096、down 2048、288 個 | あり | `--check`、GPU の一致 100% | ○ |
| Ornith-1.5-122B、Qwen3.5-122B（`qwen35moe`） | gate/up 3072、down 1024、256 個 | 自作 | `--check`、GPU の一致 100% | ○ |
| DeepSeek V4 Flash 0731（`deepseek4`） | gate/up 4096、down 2048、256 個 | **なし** | 見出し | ○（imatrix が必要） |
| MiniMax M3.1 Flash（`minimax-m3`） | gate/up 6144、down 3072、128 個 | — | 見出し | ○ |
| 35B-A3B 系（`qwen35moe`：Ornith-1.0/1.5-35B、Qwen-AgentWorld、Holo4、Tiel-Coder など）、Laguna-S-2.1 | すべて 256 の倍数 | — | 見出し | ○ |
| Qwen3 MoE（`qwen3moe`：llm-jp-4.1-32b-a3b など） | gate/up 2048、**down 768 / 960** | — | 見出し | △（down は IQ4_NL） |
| Gemma 4 26B-A4B（`gemma4`） | まとまった gate_up 2816、**down 704** | — | 見出し | △（down は IQ4_NL。`ffn_gate_up_exps` の規則を書く） |

◎ 通しで実行済み、○ 形が合う（表にあるものは `--check` と GPU の一致も確認）、△ 一部のテンソルは代わりの型になる。

**一部のテンソルが指定の型にならない理由**

| 場合 | 例 | 原因 | trench-gguf の動き |
| --- | --- | --- | --- |
| 行の長さが 256 の倍数でない | Qwen3.8 の down（640）、Qwen3 MoE の down（768 / 960）、Gemma 4 の down（704） | IQ1/IQ2/IQ3 系と k-quant は、重み 256 個を 1 ブロックにまとめる型です。256 で割り切れない行は保存できません。ツールではなく、型そのものの制限です | llama-quantize と同じ代わりの型にします：IQ1/IQ2/IQ3 系と IQ4_XS → IQ4_NL、Q2_K/Q3_K → Q4_0、Q4_K → Q5_0、Q5_K → Q5_1、Q6_K → Q8_0（それも合わなければ F16）。unsloth や ISTA の Qwen3.8 も、ここは IQ4_NL です |
| そのテンソルの imatrix のデータがない | 予測用の追加の層（NextN / MTP）：Ornith-1.5 の `blk.48`、GLM-5.3-Flash の `blk.45` | imatrix を作るときに、その層を通っていません。IQ1_S/IQ1_M/IQ2_XXS/IQ2_XS/IQ2_S は imatrix が必須です（llama-quantize も「結果がでたらめになる」と止まります） | 元のまま残します。imatrix が要らない型の規則を、ほかの規則より上に書いてください（例：`blk\.48\.ffn_\w+_exps\.weight  iq4_xs`） |
| gate と up が 1 本にまとまっている | Gemma 4 の `ffn_gate_up_exps` | テンソルが 2 本ではなく 1 本です | 量子化できます。規則に `ffn_gate_up_exps` を書く必要があります |

**対象外のもの**

| 場合 | 例 | 原因 |
| --- | --- | --- |
| 密なモデル | Qwen3.8-27B、Gemma 4 12B / 31B、MiniCPM5 | エキスパートのテンソルがありません。trench-gguf は MoE のエキスパートだけを書き直します（密なモデルへの対応は今後の予定） |
| フォーク独自のテンソルの型 | ROCmFP4 の版（型 101）、Ternary Bonsai の PQ2_0 / PTQ1_0（型 142 / 143） | 本家の llama.cpp にない型です。gguf-py が見出しを読めず、ggml も元の重みに戻せません |
| そのモデルの imatrix がない | DeepSeek V4 Flash 0731（unsloth は公開していない） | IQ1/IQ2 系には imatrix が必須です。`llama-imatrix` で作るか、配布元のものを使ってください |

## 🛠️ 使い方

### モデルのエキスパートを量子化し直す

一般的な書き方です（具体的な値を入れたものは[実例](#-実例-deepseek-v41-flash-を一から作る)にあります）。

```bash
python tools/requant_experts.py <元のシャード 1> <出力フォルダ> <規則ファイル> --imatrix=<imatrix.gguf> --gpu
```

規則ファイルは、1 行に「テンソル名の正規表現」と「型」を書きます。上から見て、最初に当てはまった行が使われます。

```text
blk\.[0-9]\.ffn_(gate|up)_exps\.weight              iq1_s
blk\.(2[5-9]|3[0-9])\.ffn_(gate|up)_exps\.weight    iq2_xxs
blk\.\d+\.ffn_down_exps\.weight                     iq2_s
```

- **元のファイルはなるべく精度の高いものを**
  - 元のファイルの丸め誤差に、新しい量子化の誤差が重なります。
  - たとえば DeepSeek V4.1 なら、unsloth の UD-Q4_K_XL のエキスパートは配布元の MXFP4 そのままです。量子化し直す前に失われるものがありません。
- **`--gpu`**
  - IQ2_XXS / IQ2_S / IQ1_S / IQ1_M を GPU で計算します。2 枚目の GPU を使うときは `--gpu=1` です。
  - それ以外の型は CPU で計算します。
  - ログの行末に `(gpu)` と付いたものが、GPU で計算したテンソルです。
- **`--check`**：書き出さずに、計画と大きさだけを表示します。

### 2 つの GGUF を層ごとに組み合わせる

```bash
python tools/merge_experts.py <土台のシャード 1> <差し替え元のシャード 1> <出力フォルダ> <出力名> 0-9,12,35-39
```

- 指定した層の `ffn_{gate,up,down}_exps` を差し替え元から取り、それ以外はすべて土台から取ります。
- 量子化し直しはしません。
- 差し替えるテンソルがないシャードは、そのままコピーします。

### 一致を自分で確かめる

```bash
python gpuq/test_gpuq.py --selftest
python gpuq/test_gpuq.py <GGUF のシャード> <imatrix.gguf> --experts=3 --tensor='blk\.30\.ffn_down_exps'
```

次の場合は、もう一度確かめてください。
- llama.cpp を更新したとき。先に `./setup.sh` を流すと、表も書き出し直されます。
- 新しい GPU で使うとき。

## 📊 速さ

AI PRO R9700（OCuLink Gen4 x4）、16 コアの Ryzen（32 スレッド）、DeepSeek V4.1 Flash のエキスパートで測りました（2026-10-29）。

| | CPU（ggml） | GPU（trench-gguf） |
| --- | --- | --- |
| エキスパート 1 個（1,180 万個の重み） | 1 スレッドで 0.5〜2.5 秒 | 0.010〜0.028 秒 |
| テンソル 1 本（エキスパート 384 個） | 32 スレッドで約 27 秒 | 約 5 秒 |
| 1 層（gate / up / down）を GGUF に | 1 分 54 秒 | 29 秒、`cmp` で同じ |
| エキスパート全部（120 本、MXFP4 の 289 GB → 132 GB） | 約 1 時間（見積もり） | 15 分半 |

GPU の時間の約半分は、float の重みを PCIe で送る時間です。

## 🧠 仕組み

- **同じコード**：GPU の 1 スレッドが、256 個の重みのまとまり 1 つを担当します。中身は ggml の `quantize_row_*_impl` を 1 行ずつ写したものです。
- **同じ表**：`gpuq/gpuq_tables.c` は `ggml-quants.c` そのものを取り込み、`iq2xs_init_impl` が作る grid、map、近傍の表を書き出します。
- **同じ計算**：
  - llama.cpp の `libggml-base` は FMA を使わずにビルドされています。
  - そこで GPU 側も FMA へのまとめを止めています（`-ffp-contract=off`、nvcc は `-fmad=false`）。
  - 割り算と平方根も、正しく丸める設定にしています。
- **同じ順番**：
  - IQ1_S / IQ1_M は、重みを並べ替えてから探索します。
  - glibc 2.39 の `qsort` は、この大きさでは安定なマージソートです。
  - GPU も安定な並べ替えを使い、同じ値の順番をそろえています。
- **同じ fp16 への丸め**：ggml の `ggml_compute_fp32_to_fp16` をそのまま写しています。

## ⚠️ 制限

- **型**：GPU で計算するのは IQ2_XXS、IQ2_S、IQ1_S、IQ1_M です。IQ2_XS と IQ3_XXS が次の予定です。それ以外の型（代わりの型の IQ4_NL も）は、ggml を通して CPU で計算します。
- **モデル**：MoE のエキスパートだけが対象です。代わりの型になるテンソルとその理由は、[対応するモデル](#-対応するモデル)を見てください。
- **CUDA**：ビルドはできますが、NVIDIA の GPU での一致はまだ確かめていません。
- **環境**：確かめたのは x86-64 の Linux だけです。
  - ARM では gcc がふつうに FMA を使うので、ggml の CPU の出力そのものが変わります。
  - Windows では、MSVC の `qsort` が同じ値の順番を変える可能性があります。
  - どちらも、自己テストで分かります。
- **llama.cpp の版**：
  - bed0a8566 から写し、ac8e1694d でも確かめ直しました。
  - それより古い llama.cpp は、IQ1_M の探索のやり方が違います。
  - 更新したら、自己テストを流してください。
- **imatrix が必須**です（llama.cpp でも、IQ1/IQ2 系では必須です）。imatrix にデータのないテンソルは、imatrix が要らない型なら imatrix なしで作り、IQ1/IQ2 系なら元のまま残します。

## 🗺️ 今後の予定

- IQ2_XS と IQ3_XXS の GPU 版
- RTX 5090 での CUDA 版の一致の確認
- PCIe の転送と GPU の計算を重ねる
- GPU がない人向けに、ggml と一致したまま CPU 版を速くする（符号帳の探索を SIMD のレーンで並べる）

## 🙏 謝辞

- [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp)：i-quant と、その量子化の処理。trench-gguf はこれを移植しています。
- [unsloth](https://huggingface.co/unsloth)：確認に使った GGUF と imatrix。

## 📄 ライセンス

MIT です。詳しくは [LICENSE](LICENSE) を見てください。移植した ggml のコードには、ggml の作者の著作権表示を残しています。
