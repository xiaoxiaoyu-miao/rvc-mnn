# RVC-MNN (Android)

hubert + 合成器走 **MNN（Vulkan/CPU）**，f0 提取走 **ONNX Runtime（fcpe）** 的实时/离线 RVC 变声 App。

## 构建

```bash
# Termux 下 aapt2 需要指向 SDK 自带的可执行文件（Maven 版是给 glibc 的）
./gradlew assembleDebug \
  -Pandroid.aapt2FromMavenOverride=$PREFIX/opt/android-sdk/build-tools/35.0.0/aapt2
# 产物: app/build/outputs/apk/debug/app-debug.apk
```

首次启动会申请「所有文件访问权限」，模型放在 `/sdcard/models/mm/`。

> 仓库只带源码和 DFN 降噪模型。RVC 的四个权重（`hubert_fp16.mnn` 181 MB、`ct_static.mnn` 105 MB、
> `fcpe_audio.onnx` 43 MB、`emb_pitch.bin`）超过 GitHub 单文件限制，没有放进仓库：
> 要么放到 `/sdcard/models/mm/`（App 优先用这个目录），要么放进 `app/src/main/assets/models/` 随 APK 打包。
> Diffusion 模型同样没有内置，见录音模式一节。

## 功能

| 区块 | 说明 |
| --- | --- |
| ① 音色预设 | 3 个预设（与 MeowRVC 演示一致）：男→女 / 女→男 / 同性转换。原理是缩放 fcpe 输出的 f0（`k = 2^(半音/12)`，对应 RVC 的 `f0_up_key`），再重新取音高包络，**不改变音色模型**。滑块可再微调 ±24 半音。 |
| ② 引擎与模型 | 3 个引擎预设：`极速(CPU)` / `均衡(Vulkan)` / `高质量(Vulkan+fp32)`，联动后端与两个模型的计算精度；模型目录可改，自动扫描 `*.mnn`（hubert / 合成器）、`*fcpe*.onnx`、`*emb*.bin` 并分四个下拉手动选择；配置可导出 JSON、可一键重置。 |
| ③ WAV 转换 | 可选音频文件（SAF 自动拷到 Download）、自动生成输出名、离线转换、试听结果。 |
| ④ 实时变声 | 低延迟流水线（采集 → hubert+fcpe → 合成 → 播放）。**回环/自激专门处理**：进通话音频模式让系统 AEC 真正工作、输出标成语音通信用途、逐帧自适应反馈抑制门控切断"扬声器→麦克风→再合成"的循环，队列只保留最新块（不越播越旧）。输出路由三选一：**外放 / 听筒 / 不外放**；可把输出录成 wav。 |
| ⑤ 录屏 | **App 内录屏**：屏幕 H.264 + 实时变声音轨 AAC 合成 mp4。因为声音直接取自实时管线、根本不经过扬声器，所以能做到**手机不出声、视频里有声音**。 |
| ⑥ 录音模式 | 独立的「录完即停 · 不实时 · 不外放」页面：录音**或直接挑一个现成音频文件** → RVC 离线推理 → DFN3 降噪 → SGMSE+ Diffusion 精修 → 转 SILK → 用 root/Shizuku 写进 QQ 语音草稿（可一键串到底）。见下方专节。 |
| 日志 | 屏幕日志 + `/sdcard/Download/rvc_progress.log`，支持清空/导出。 |

所有配置（模型路径、预设、变调、增益、门限、反馈抑制、输出路由、录屏捕获开关、输入输出路径）都写在 SharedPreferences
（`rvc_mnn_cfg`），重启后自动恢复；`导出配置` 会在 `/sdcard/Download/rvc_config.json` 生成一份可读 JSON。

## 预设对照（音色）

| 预设 | 半音 | f0 倍率 |
| --- | --- | --- |
| 男→女 | +12 | 2.000 |
| 女→男 | −12 | 0.500 |
| 同性转换 | 0 | 1.000 |

## 录音模式：录一段，或者直接挑一个文件

`RecordActivity` 是独立页面，跟实时链路互不相干：**不建 AudioTrack、全程不外放**。

两条输入路径：

| 入口 | 说明 |
| --- | --- |
| **开始录音 / 停止并变声** | AudioRecord 只往 WAV 里灌数据，录完才把整段丢给离线变声。 |
| **选择音频文件** | SAF 挑任意音频（wav / mp3 / m4a / aac / amr / flac…系统能解的都行）。`AudioImport` 用 `MediaExtractor + MediaCodec` 解成单声道 WAV（本身是 wav 就直接原样拷贝，零损失），之后**和录音产物走同一条链路**。 |

产物处理：`转 SILK`（码率可切）→ `写 QQ 草稿`（root/Shizuku 覆写 QQ 的 ptt/*.slk）。

**转换后直接发 QQ（默认开）**：选完文件就自动跑完
`导入 → RVC 推理 → DFN3 降噪 → SGMSE+（有模型才跑）→ SILK @16k → 覆写最近一条 QQ 语音草稿`，
最后弹一个「去 QQ」按钮跳过去——只剩在 QQ 里按一下发送。
QQ 的"发送"动作在它自己进程里，没有公开接口，所以能自动到"草稿已就绪"这一步。

### 推理后处理：DFN3 降噪 → SGMSE+ Diffusion

降噪**不是在推理前**，而是接在 RVC 输出后面：

```
录音/导入 → OfflineConvert（16k 入，40/48k 出） → DeepFilterNet3 → SGMSE+ → SILK/QQ
```

- **DFN3 降噪**（`DeepFilter.java`）：官方 DeepFilterNet3 v0.5.6 的 ONNX 神经图，模型内置在
  `assets/dfn/`，不占外置模型目录。STFT 960/480、Vorbis 窗、32 个 ERB 带、order=5 的深层滤波
  全部在 Java 里实现（960 点混合基 FFT，不依赖 FFTW）。RVC 输出是什么采样率就还回什么采样率
  （本机合成器 40 kHz），不会再压成 16 kHz。失败会自动跳过，保留推理结果。
- **SGMSE+ 精修**（`Sgmse.java` / `Diffusion.java`）：链路最后一段，默认**关**。这是 16 kHz
  复数 STFT 域扩散（65M 参数，OUVE predictor-corrector），Java 里实现了
  `n_fft=510 / hop=128` 的 STFT、reflect padding、8/20/30 步采样和分块重叠拼接。
  - 模型不随 APK 分发（约 262 MB）。把 `sgmse.onnx` 放到 `/sdcard/models/mm/`，
    或把 `Config.diffusionModel` 改成绝对路径；找不到模型时自动跳过。
  - 生成方法见 `tools/sgmse/README.md`：从 hf-mirror 拉官方 checkpoint，用 PyTorch 权重直接
    生成 ONNX（不依赖 Python `onnx`，序列化用 `rvc-onnx-web` 的 `onnx-serializer`）。
  - 采样率固定 16 kHz：先把 RVC 输出降到 16k，增强后再升回原采样率，并用
    `orig + lowpass(enh - orig)` 保留 RVC 的高频（`diffusionKeepHigh`）。
  - **很慢**：默认 8 步（Fast）在手机上每 2 秒音频约 1 分钟量级；20/30 步更慢。只建议离线
    处理短语音，默认关。

> DFN3 模型来自 [Rikorose/DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) v0.5.6
> （MIT OR Apache-2.0），ONNX 导出、哈希与数值校验见 `docs/dfn/export_manifest.json`。
> SGMSE+ 来自 [sp-uhh/sgmse](https://github.com/sp-uhh/sgmse)（MIT），权重由
> [speechbrain/sgmse-voicebank](https://hf-mirror.com/speechbrain/sgmse-voicebank) 提供。

命令行（有 root 时跑得通，方便批量/自动化）：

```bash
adb shell su -c 'am start -n com.rvc.mnn/.RecordActivity --es in /sdcard/Download/a.mp3 --ez auto true'
# 关降噪 / 不自动写 QQ / 开 SGMSE+：
#   --ez dn false  --ez send false  --ez diff true
```

## 「说两个字轻轻顿一下」：块间接缝

### 现象与根因

实时和离线都会有：正常说话里，大约每半秒会轻轻顿一下，顿的那一下还带着一点很小、不太正常的声音。

原因不在推理速度，而在**切块方式**：合成器是逐块独立跑前向的，窗口长 `SEG = 49×160 = 7840` 样本（0.49s）。
卷积网络在窗口边缘拿不到上下文，于是**每块首尾各有一小段输出是歪的**；把这种块首尾直接拼起来，
波形在每个 0.49s 边界上会出现一次跳变。

实测（同一段输入，40 kHz 输出，量「跨块那一步的样本差 vs 块内典型步长」）：

| 版本 | 跨界步长中位 | 均值 | 最大 | 相对块内 | 明显跳变(>0.1)的缝 |
| --- | --- | --- | --- | --- | --- |
| 原来（块首尾直拼） | 0.149 | 0.191 | 0.765 | **25.4×** | **14 / 23** |
| 重叠 5 帧 + 交叉淡化 | 0.011 | 0.012 | 0.044 | 1.7× | **0 / 25** |

### 修法：重叠窗口 + 交叉淡化（`RvcStitch`）

- 窗口长度仍是 `SEG`，但**每次只前进 `HOP_SEG = SEG − 5×HOP = 7040`**（相邻窗口重叠 50 ms）；
- 重叠那段在两块里各被合成过一次，用线性交叉淡化混合，各自边缘的瑕疵互相盖掉；
- 每块只吐 `segOut − ovOut` 个样本，所以**输出总时长/时间轴和原来完全一致**；
- 代价：块数 +11.4%（离线 1.18x → 约 1.05x realtime，仍比实时快；Vulkan 实时链路本来就只在 60~80% 占空比）；
- 静音块（实时输入门限）走 `pushSilence`：把上一块尾巴**淡出**再补零，不硬切。

> 为什么不做"给模型喂更长的上下文"：合成器是静态 `[1,49,768]`，喂不进更长的窗口；
> 而重叠+淡化不需要改模型 I/O。

## 实时变声起步的那几秒

「刚开实时变声前几秒会顿、过一会儿就好了」是另一码事：实时链路刚起来时是最冷的
——Vulkan 第一次遇到某个卷积形状才编译 pipeline、ORT 还没分配 arena、CPU 频率也没拉起来，
这些成本全摊在**前几块**上。现在按顺序处理：

1. **开采集之前先预热**：`EngineWarmup.warmForce()` 把 `hubert → fcpe → 合成器` 整链连跑 2 遍
   （`RealtimeService` 自己调，不会被"实时在跑就跳过预热"那条规则挡掉）；
2. **拿到第一块合成结果再 `AudioTrack.play()`**：一上来就按实时节奏灌静音，开头必然是空洞；
3. **欠载只补 50 ms 静音**（不再补一整块 0.44s）：真正的空洞从 ~440 ms 缩到 ~50 ms；
4. **播放/节拍线程提到 `URGENT_AUDIO` 优先级**：它原来跑在默认优先级，会被两个计算线程挤掉，
   一迟到就是一次 AudioTrack 欠载。

实测 60 秒实时运行（Vulkan 均衡预设）：只在开播瞬间出现过 1 次等待，之后 **0 次**；
`丢弃 0`；RT 稳定在 0.98~0.99x。

## 实时性能：两级流水线

实时链路原本是**单线程串行**跑「hubert+fcpe → 合成器」，两者互相等待：

```
qIn → [hub(CPU) → fcpe(ORT) → synth(Vulkan)] → qOut     单块 = hub+fc+syn
```

但 hubert/fcpe 在 **CPU**、合成器在 **Vulkan/GPU**，天然可以重叠。拆成两级后：

```
qIn → [rt-feat: hub + fcpe] → qFeat → [rt-syn: 合成器] → qOut
       单块耗时 = max(hub+fc, syn)
```

实测（本机，OnePlus PLQ110）：

| 预设 | 改造前 | 改造后 |
| --- | --- | --- |
| 均衡 (Vulkan) | 0.49x RT | **0.98~1.00x RT** |
| 极速 (CPU+fp16) | 0.27x RT | **0.98~1.00x RT** |

两边都到了 **1.0x RT 的实时上限**——播放端 `AudioTrack.write()` 本身就是按实时节奏消费，
处理再快也不会体现在 RT 数字上，只会变成队列余量（抗降频/抗卡顿）。

实现要点：

- `Feat` 特征包 + 对象池（`featPool`）复用大数组，避免每块 GC 抖动；
- `qFeat` 只保留最新一块（`offer` 失败就丢旧的），延迟有界；
- 静音块也走 `qFeat`（`silent` 标记），保证输出节奏与音轨时间轴不错位；
- 采集线程 `THREAD_PRIORITY_URGENT_AUDIO`，两级计算线程 `THREAD_PRIORITY_AUDIO`，
  减少被大小核调度挤压导致的欠载。

再往下想更快，只剩**降延迟**（减小块长，`SEG_FR×HOP` 现在是 49×160=7840 样本 ≈0.49s）
这一条路，但块长减半会让 GPU 单次工作量变小、效率下降，需要重新权衡。

## 录屏：不外放但有声音

需求是「屏幕录制时手机别出声，但录出来的视频里要有变声」。这在 Android 上**没有系统级方案**，原因是：

- 系统/第三方录屏靠 `AudioPlaybackCapture`（MediaProjection 内部音频），它复制的是**正在播放**的音频，
  而播放就必然走扬声器；
- 想绕开扬声器需要把 AudioTrack 路由到 `REMOTE_SUBMIX` 虚拟输出，但本机 `AudioPolicyManager`
  只暴露 `earpiece / speaker / telephony_tx` 三个输出设备，**没有 REMOTE_SUBMIX**（`r_submix` HAL 虽然
  被 policy include，但输出端口没 attached，App 层无法连上）；
- 把音量调 0 也不行——`AudioPlaybackCapture` 取的是混音后（音量之后）的信号，副本同样是静音。

所以本 App 直接自己录屏：

1. 「实时输出路由」选 **不外放** —— 播放链路整个不建 AudioTrack，声音一点不出去；
2. 点 **开始录屏** → 系统弹一次「开始录制或投放」授权 → 服务里建 `VirtualDisplay`（Android 14+ 要求
   前台服务带 `mediaProjection` 类型）；
3. 屏幕画面进 H.264（Surface 输入，长边上限 1920，6Mbps / 30fps），实时 PCM 重采样
   40000→48000 后进 AAC-LC（单声道 128kbps），最后由 `MediaMuxer` 合成一个 mp4；
4. 输出：`/sdcard/Download/rvc_rec_<时间戳>.mp4`。

细节：

- 实时变声没开时会自动一起启动（同一个前台服务里跑）；
- 静音块也会喂给编码器，音轨时间轴不会因为门限静音而跳；
- 停止实时变声 = 同时收尾录屏（`ScreenRecorder.stop()`），mp4 正常封口。

### 想用系统/第三方录屏的话

勾「让系统录屏能录到」：输出切到 `USAGE_MEDIA` + `ALLOW_CAPTURE_BY_ALL`，ColorOS 录屏里选
「系统声音/内部音频」就能录到变声。**代价**：media 路径下系统 AEC 不再把这份输出当参考信号，
所以这个开关只适合录屏/直播，实时通话场景要关掉。而且这条路**声音仍然会外放**。

## 实时回环（啸叫）处理

物理回环没法纯软件 100% 消除（扬声器声音必然进麦克风），所以按 MeowRVC 的做法叠三层：

1. **通话音频路径**：`AudioManager.MODE_IN_COMMUNICATION` + `setSpeakerphoneOn()`（可切免提/听筒），
   采集用 `VOICE_COMMUNICATION`，这样系统 AEC/NS 才会真正挂在采集流上（否则只建 Effect 对象不生效）；
2. **输出参考信号**：AudioTrack 使用 `USAGE_VOICE_COMMUNICATION` + `CONTENT_TYPE_SPEECH`，
   AEC 才知道要抵消什么；
3. **逐帧自适应噪声门**（`applyFeedbackGate`，移植 MeowRVC 的 `adaptiveNoiseGate`）：
   合成输出按 20ms 帧做能量门控，低于门限（滑块 0~60dB，默认 25dB）压到 0.1 倍，
   掐掉回声尾巴，打断自激循环——这是解决"回环一直响"的关键。

另外队列积压时**只保留最新块**（`clear()+offer()`），避免处理跟不上时越播越滞后、像在重放旧音频；
退出实时模式会复位 `MODE_NORMAL` 并关免提。**外放仍建议戴耳机**；把反馈抑制调到 0 等于放弃这层保护。

## 引擎预设对照

| 预设 | 后端 | hubert 精度 | 合成器精度 | 偏好的 hubert 权重 |
| --- | --- | --- | --- | --- |
| 极速 (CPU+fp16) | CPU | fp16 (arm82) | fp16 (arm82) | hubert_fp16.mnn |
| 均衡 (Vulkan) | Vulkan | normal (fp32) | normal (fp32) | hubert_fp16.mnn |
| 高质量 (Vulkan+fp32) | Vulkan | fp32 | fp32 | hubert_static.mnn（缺失则保留当前权重并提示） |

### 精度/线程实测（本机 OnePlus PLQ110，骁龙 8 系，2×4.32GHz 大核 + 6×3.53GHz 中核）

合成器单块耗时（约 0.49s 音频）：

| 配置 | synth | 结论 |
| --- | --- | --- |
| CPU + fp32(Normal) | ~550ms | 基线 |
| **CPU + fp16(Low)** | **~317ms** | **快 42%，所以「极速」用 fp16** |
| Vulkan + fp32(Normal) | ~270ms | 最快，默认预设 |
| Vulkan + fp16(Low) | ~340ms | 反而更慢，不采用 |

线程数（CPU+fp16）：2→256ms、4→317ms、6→261ms、**8→382ms**（8 线程受大小核调度拖累明显变慢）。
默认按 **大核数 × 2**（本机 = 4）自动取，可用系统属性现场覆盖，重启 App 生效：

```bash
su -c 'setprop debug.rvc.threads 6'   # 0/空 = 自动
```

NEON：MNN arm64 默认启用，库内可见 KleidiAI 的 `neon_dotprod` / `neon_i8mm` kernel；设备 `asimddp`/`i8mm`/`bf16` 均支持，无需额外开关。

> CPU 后端整体仍比 Vulkan 慢（0.3~0.5x RT vs 0.98x RT），它的价值在不占 GPU、功耗低；
> 实时变声想要跟得上，默认的「均衡 (Vulkan)」才是正确选择。

## 文件

- `app/src/main/java/com/rvc/mnn/MainActivity.java` — UI、实时启动、WAV IO、配置联动（③ 的转换调用 `OfflineConvert`）
- `app/src/main/java/com/rvc/mnn/RecordActivity.java` — 录音模式：录音 / 选文件 → RVC 推理 → DFN3 降噪 → SGMSE+ → SILK → 写 QQ 草稿
- `app/src/main/java/com/rvc/mnn/DeepFilter.java` — DeepFilterNet3 降噪：ONNX 神经图 + STFT/ERB/深层滤波/重采样
- `app/src/main/java/com/rvc/mnn/Sgmse.java` — SGMSE+ 16k 复数 STFT 扩散采样器
- `app/src/main/java/com/rvc/mnn/Diffusion.java` — Diffusion 阶段的采样率转换/高频保留/模型查找
- `app/src/main/java/com/rvc/mnn/RvcStitch.java` — 块间接缝：重叠窗口 + 交叉淡化（实时与离线共用）
- `app/src/main/java/com/rvc/mnn/AudioImport.java` — 任意音频文件 → 单声道 WAV（MediaExtractor + MediaCodec）
- `app/src/main/java/com/rvc/mnn/OfflineConvert.java` — 离线整段变声（③ 文件转换与录音模式共用）
- `app/src/main/java/com/rvc/mnn/ScreenRecorder.java` — App 内录屏：MediaProjection + H.264/AAC + MediaMuxer
- `app/src/main/java/com/rvc/mnn/Config.java` — 预设表 + SharedPreferences/JSON 持久化
- `app/src/main/cpp/rvc_jni.cpp` — MNN JNI：`init(paths, vulkan, hubPrec, ctPrec)`、`hubert`、`synth`、`audioLen`、`backendInfo`、`release`（配置变化时自动重建 session）
- `app/src/main/assets/dfn/` — 内置 DeepFilterNet3 ONNX 模型与 ERB/Vorbis 常量（约 8.7 MB）
- `docs/dfn/` — DFN3 模型卡、config 与导出校验清单（来源与哈希）
- `tools/sgmse/` — SGMSE+ 权重 → ONNX 的生成脚本与说明（模型 262 MB，不进仓库）

## 命令行入口（可选）

```bash
# ③ 文件转换：可显式指定输入/输出
adb shell am start -n com.rvc.mnn/.MainActivity \
  --es in /sdcard/Download/a.wav --es out /sdcard/Download/a_rvc.wav --ez auto true

# ⑥ 录音模式：把外部音频喂进「导入 → RVC 推理 → DFN3 降噪 → SGMSE+ → SILK → 写 QQ 草稿」整条链
adb shell su -c 'am start -n com.rvc.mnn/.RecordActivity --es in /sdcard/Download/a.mp3 --ez auto true'
```

## 后台保活（实时变声退后台/息屏继续）

实时流水线跑在 **前台服务** `RealtimeService`（`foregroundServiceType=microphone|mediaPlayback`）里：

- 通知栏常驻 + 「停止」按钮；
- `PARTIAL_WAKE_LOCK` 防 CPU 休眠断流，退出时复位 `MODE_NORMAL`；
- App 内「关电池优化」按钮（`ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS`）；
- root/ADB 侧已执行：`cmd deviceidle whitelist +com.rvc.mnn`、`appops RUN_IN_BACKGROUND allow`、`appops WAKE_LOCK allow`；
- 实测：Home 键退到桌面、息屏 15s 后仍在出块（日志 `实时 N 块` 持续增长）。

> 注意：实时流水线本身 ≈0.3–0.6× 实时（本机合成器 0.3–0.5s / 0.49s 块），处理跟不上时队列只保留最新块。
> 内存吃紧时系统可能直接杀进程（日志 `has died: prcp FGS`）；保活三件套已覆盖正常情况。

