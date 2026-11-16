# SGMSE+ ONNX 生成

这是录音模式最后一段 Diffusion 精修用的 **SGMSE+**（sp-uhh / speechbrain 的 16 kHz
VoiceBank-DEMAND 模型）。模型约 262 MB，不随仓库/APK 分发，需要在本机生成一次：

## 1. 拉代码和权重（hf-mirror）

```bash
# SGMSE 的 NCSN++ 代码（MIT）
git clone https://hf-mirror.com/sp-uhh/speech-enhancement-sgmse tools/sgmse/sgmse_src

# SpeechBrain 的 SGMSE+ 权重（约 262 MB）
curl -L -o tools/sgmse/score_model_ema.ckpt \
  https://hf-mirror.com/speechbrain/sgmse-voicebank/resolve/main/score_model_ema.ckpt
```

只用到 `sgmse_src/sgmse/backbones/` 等推理代码，不需要装 SpeechBrain / Lightning / onnx。

## 2. 生成 ONNX（PyTorch + rvc-onnx-web 序列化器，不装 Python onnx）

```bash
# 256 帧 = 2.048 s，和训练时的 num_frames 一致
python tools/sgmse/build_sgmse.py \
  --frames 256 \
  --checkpoint tools/sgmse/score_model_ema.ckpt \
  --out-json tools/sgmse/sgmse.json \
  --out-weights tools/sgmse/sgmse.bin

# 用 rvc-onnx-web 的 dependency-free serializer 写 ONNX
RVC_ONNX_WEB=$HOME/rvc-onnx-web/dist/onnx-serializer.js \
node tools/sgmse/serialize_sgmse.mjs \
  tools/sgmse/sgmse.json tools/sgmse/sgmse.bin tools/sgmse/sgmse.onnx
```

`build_sgmse.py` 直接遍历 PyTorch 模块权重、手写 NCSN++ 图（Conv/GroupNorm/SiLU/
注意力/FIR 上采样），固定 `[1,4,256,256]` 输入；已和 PyTorch 前向对齐到 `4e-5`（max abs）。

## 3. 安装到手机

```bash
mkdir -p /sdcard/models/mm
cp tools/sgmse/sgmse.onnx /sdcard/models/mm/sgmse.onnx
```

App 的 `Diffusion.modelFile()` 会先找 `cfg.modelDir/sgmse.onnx`，再回退到
`/sdcard/models/mm/sgmse.onnx`。启动录音模式 → 打开「Diffusion 精修」即可；
默认 8 步（Fast），20/30 步更慢。

## 注意

- `frames` 固定为 256；Java 端按 256 帧分块（重叠 64 帧）处理长音频。
- ONNX 输入 `features[1,4,256,256]`（state.real/imag、noisy.real/imag）和 `time[1]`，
  输出 `score[1,2,256,256]`（已经取负）。
- STFT：`n_fft=510`、`hop=128`、periodic Hann、reflect padding；
  频谱变换 `abs^0.5 * exp(i*angle) * 0.15`。
- 采样器：OUVE SDE，theta=1.5、sigma 0.05→0.5、eps=0.03、corrector snr=0.5。
- 手机 CPU 上很慢，默认关；只适合离线短语音。
