package com.rvc.mnn;

import android.content.Context;
import android.content.SharedPreferences;

import org.json.JSONObject;

/**
 * RVC-MNN 的可持久化配置 + 预设表。
 *
 * <p>分两部分：
 * <ul>
 *   <li>音色预设：只改音高（半音），作用在 fcpe 输出的 f0 上，再送进重采样音高编码器；
 *       对应 RVC 的 f0_up_key。</li>
 *   <li>引擎预设：改后端（CPU/Vulkan）与两个模型的计算精度，影响速度/显存/音质。</li>
 * </ul>
 * 所有值都落 SharedPreferences，也可以通过 toJson/fromJson 导出/导入成一份 JSON 配置。
 */
public class Config {
    public static final String PREFS = "rvc_mnn_cfg";
    public static final String DEF_MODEL_DIR = "/sdcard/models/mm";

    // ---------------- 音色预设（半音） ----------------
    // 直接照搬 MeowRVC 演示里的三个一键预设（男→女 / 女→男 / 同性转换）。
    // 作用点：fcpe 输出的 f0 乘 2^(半音/12)，等价 RVC 的 f0_up_key。
    public static final String[] VOICE_NAMES = {"男→女", "女→男", "同性转换"};
    public static final float[] VOICE_SEMI = {12f, -12f, 0f};

    // ---------------- 引擎预设 ----------------
    // 本机 CPU 支持 ARMv8.2 FP16（/proc/cpuinfo: asimdhp asimdfhm i8mm bf16 asimddp），
    // 所以「极速」走 CPU + Precision_Low，MNN 会选中 arm82 后端做 fp16 卷积；
    // Vulkan 两个预设保留 Normal/High，避免个别 GPU 的 fp16 精度问题影响音质。
    public static final String[] ENGINE_NAMES = {"极速 (CPU+fp16)", "均衡 (Vulkan)", "高质量 (Vulkan · ct fp32)"};
    /** 每项 = {vulkan?, hubPrec, ctPrec}；prec: 0=Low(fp16) 1=Normal 2=High(fp32) */
    public static final int[][] ENGINE_CFG = {
            {0, 0, 0},     // CPU + fp16（arm82，实测比 fp32 快 42%）
            {1, 1, 1},     // Vulkan + Normal(fp32)——实测比 Vulkan fp16 更快
            {1, 2, 2},     // Vulkan + High(fp32)
    };
    /**
     * 引擎预设使用的 hubert 权重。
     *
     * <p>注意：内置 assets 只带 <b>hubert_fp16.mnn</b> 这一份。
     * 原来「高质量」预设引用 hubert_static.mnn，但该文件从未打进 APK，
     * 导致 {@link #applyEnginePreset(int)} 永远返回 false、静默回落到当前权重，
     * 预设名和实际行为不符。这里统一指向实际存在的文件；
     * 「高质量」的差异只体现在 ct 合成器走 fp32（见 {@link #ENGINE_CFG}）。
     */
    public static final String[] ENGINE_HUBERT = {"hubert_fp16.mnn", "hubert_fp16.mnn", "hubert_fp16.mnn"};

    // ---------------- 运行配置 ----------------
    public String modelDir = DEF_MODEL_DIR;
    public String hubert = "hubert_fp16.mnn";
    public String synth = "ct_static.mnn";
    public String fcpe = "fcpe_audio.onnx";
    public String emb = "emb_pitch.bin";

    public int enginePreset = 1;
    public int voicePreset = 2;       // 默认「同性转换」= 不改音高
    public float pitchSemi = 0f;      // 自定义变调（滑块可再微调，会被预设覆盖）
    public float gain = 1.0f;          // 输出增益
    public float gate = 0.0005f;       // 输入静音门限 RMS（低于它就按静音处理）

    /**
     * 实时模式的输出**反馈抑制门限**（dB，0=关闭，60=最狠）。
     * 逐帧自适应门控：合成后能量低于门限就衰减，切断"扬声器→麦克风→再合成"的循环尾巴，
     * 对应 MeowRVC 的 noiseGateDb + adaptiveNoiseGate。
     */
    public float feedbackGateDb = 25f;

    // ---------------- 实时输出路由 ----------------
    /** 免提外放：走扬声器。 */
    public static final int OUT_SPEAKER = 0;
    /** 听筒：声音小、更少自激。 */
    public static final int OUT_EARPIECE = 1;
    /**
     * 不外放：输出直接丢弃，不经过扬声器（也不写 AudioTrack）。
     * 配合「App 内录屏」（音轨直接取实时 PCM）就能做到「手机无声、视频有声」。
     */
    public static final int OUT_MUTE = 2;
    public static final String[] OUTPUT_NAMES = {
            "外放（免提扬声器）",
            "听筒（小声，少自激）",
            "不外放（静音，录屏时用）",
    };
    public static final String[] OUTPUT_HINTS = {
            "实时通话/面对面用，需要 AEC",
            "不想吵到别人时用，仍有轻微外放",
            "手机完全不出声；声音只进「App 内录屏」的 mp4",
    };

    /**
     * MNN CPU 推理线程数；0 = 自动（按 detected 大核数推算，上限 8）。
     * 命令行工具走 RVC_THREADS 环境变量，App 走这个配置。
     */
    public int numThreads = 0;

    /** 实时输出路由（见 OUTPUT_NAMES）。 */
    public int outputMode = OUT_SPEAKER;

    /**
     * 媒体用途输出：AudioTrack 用 USAGE_MEDIA + 允许被系统录屏捕获，
     * 这样 ColorOS/第三方录屏勾「内部声音」时能录到本 App 的声音。
     * 副作用：走 media 路径后系统 AEC 不再把这份输出当参考信号，实时通话场景应关闭。
     */
    public boolean captureFriendly = false;

    public String inWav = "/sdcard/Download/test10s.wav";
    public String outWav = "/sdcard/Download/out_mnn.wav";
    public boolean saveRealtime = false;

    /** 变调倍率 k = 2^(半音/12)。 */
    public float pitchRatio() {
        return (float) Math.pow(2.0, pitchSemi / 12.0);
    }

    /** 不外放模式：不创建/不驱动 AudioTrack，输出只进内部管线。 */
    public boolean isMuted() {
        return outputMode == OUT_MUTE;
    }

    /** 是否走 media 用途（可被系统录屏捕获）。不外放时必然走 media，否则无输出路径。 */
    public boolean useMediaUsage() {
        return captureFriendly || outputMode == OUT_MUTE;
    }

    /** 当前输出路由名。 */
    public String outputTag() {
        return OUTPUT_NAMES[clampOutputMode(outputMode)];
    }

    private static int clampOutputMode(int v) {
        return Math.max(0, Math.min(OUTPUT_NAMES.length - 1, v));
    }

    /** 当前音色预设名（用于文件名/日志，箭头换成连字符）。 */
    public String voiceTag() {
        int i = Math.max(0, Math.min(VOICE_NAMES.length - 1, voicePreset));
        return VOICE_NAMES[i].replace("→", "-");
    }

    /**
     * 应用引擎预设：切换后端/精度，并在权重文件存在时顺带换 hubert 变体。
     *
     * @return 若偏好的 hubert 文件不存在则返回 false（调用方提示），配置仍已应用。
     */
    public boolean applyEnginePreset(int idx) {
        if (idx < 0 || idx >= ENGINE_CFG.length) idx = 1;
        enginePreset = idx;
        String want = ENGINE_HUBERT[idx];
        if (new java.io.File(modelDir, want).isFile()) {
            hubert = want;
            return true;
        }
        return false;
    }

    public int vulkan() {
        return ENGINE_CFG[enginePreset][0];
    }

    public int hubPrec() {
        return ENGINE_CFG[enginePreset][1];
    }

    public int ctPrec() {
        return ENGINE_CFG[enginePreset][2];
    }

    // ---------------- 持久化 ----------------
    public void save(Context c) {
        SharedPreferences p = c.getSharedPreferences(PREFS, Context.MODE_PRIVATE);
        p.edit()
                .putString("modelDir", modelDir)
                .putString("hubert", hubert)
                .putString("synth", synth)
                .putString("fcpe", fcpe)
                .putString("emb", emb)
                .putInt("enginePreset", enginePreset)
                .putInt("voicePreset", voicePreset)
                .putFloat("pitchSemi", pitchSemi)
                .putFloat("gain", gain)
                .putFloat("gate", gate)
                                                                .putFloat("feedbackGateDb", feedbackGateDb)
                .putInt("numThreads", numThreads)
                .putInt("outputMode", outputMode)
                .putBoolean("captureFriendly", captureFriendly)
                .putString("inWav", inWav)
                .putString("outWav", outWav)
                .putBoolean("saveRealtime", saveRealtime)
                .apply();
    }

    public void load(Context c) {
        SharedPreferences p = c.getSharedPreferences(PREFS, Context.MODE_PRIVATE);
        modelDir = p.getString("modelDir", modelDir);
        hubert = p.getString("hubert", hubert);
        synth = p.getString("synth", synth);
        fcpe = p.getString("fcpe", fcpe);
        emb = p.getString("emb", emb);
        enginePreset = p.getInt("enginePreset", enginePreset);
        voicePreset = p.getInt("voicePreset", voicePreset);
        pitchSemi = p.getFloat("pitchSemi", pitchSemi);
        gain = p.getFloat("gain", gain);
        gate = p.getFloat("gate", gate);
        feedbackGateDb = p.getFloat("feedbackGateDb", feedbackGateDb);
        // 兼容旧配置：老版本只有 speakerphone 布尔
        int legacyOut = p.getBoolean("speakerphone", true) ? OUT_SPEAKER : OUT_EARPIECE;
        numThreads = p.getInt("numThreads", numThreads);
        outputMode = p.getInt("outputMode", legacyOut);
        captureFriendly = p.getBoolean("captureFriendly", captureFriendly);
        inWav = p.getString("inWav", inWav);
        outWav = p.getString("outWav", outWav);
        saveRealtime = p.getBoolean("saveRealtime", saveRealtime);
        clamp();
    }

    /** 防止手工改过 SharedPreferences / 导入 JSON 后下标越界。 */
    public void clamp() {
        enginePreset = Math.max(0, Math.min(ENGINE_CFG.length - 1, enginePreset));
        voicePreset = Math.max(0, Math.min(VOICE_SEMI.length - 1, voicePreset));
        pitchSemi = Math.max(-24f, Math.min(24f, pitchSemi));
        gain = Math.max(0.25f, Math.min(3f, gain));
        gate = Math.max(0f, Math.min(0.02f, gate));
        feedbackGateDb = Math.max(0f, Math.min(60f, feedbackGateDb));
        outputMode = clampOutputMode(outputMode);
        numThreads = Math.max(0, Math.min(8, numThreads));
    }

    public JSONObject toJson() {
        try {
            JSONObject o = new JSONObject();
            o.put("modelDir", modelDir);
            o.put("hubert", hubert);
            o.put("synth", synth);
            o.put("fcpe", fcpe);
            o.put("emb", emb);
            o.put("enginePreset", enginePreset);
            o.put("voicePreset", voicePreset);
            o.put("pitchSemi", pitchSemi);
            o.put("gain", gain);
            o.put("gate", gate);
            o.put("feedbackGateDb", feedbackGateDb);
            o.put("numThreads", numThreads);
            o.put("outputMode", outputMode);
            o.put("captureFriendly", captureFriendly);
            o.put("inWav", inWav);
            o.put("outWav", outWav);
            o.put("saveRealtime", saveRealtime);
            return o;
        } catch (Exception e) {
            return new JSONObject();
        }
    }

    /** 从 JSON 读入；缺字段保持原值。 */
    public void fromJson(JSONObject o) {
        if (o == null) return;
        modelDir = o.optString("modelDir", modelDir);
        hubert = o.optString("hubert", hubert);
        synth = o.optString("synth", synth);
        fcpe = o.optString("fcpe", fcpe);
        emb = o.optString("emb", emb);
        enginePreset = o.optInt("enginePreset", enginePreset);
        voicePreset = o.optInt("voicePreset", voicePreset);
        pitchSemi = (float) o.optDouble("pitchSemi", pitchSemi);
        gain = (float) o.optDouble("gain", gain);
        gate = (float) o.optDouble("gate", gate);
        feedbackGateDb = (float) o.optDouble("feedbackGateDb", feedbackGateDb);
        numThreads = o.optInt("numThreads", numThreads);
        outputMode = o.optInt("outputMode", o.optBoolean("speakerphone", true) ? OUT_SPEAKER : OUT_EARPIECE);
        captureFriendly = o.optBoolean("captureFriendly", captureFriendly);
        inWav = o.optString("inWav", inWav);
        outWav = o.optString("outWav", outWav);
        saveRealtime = o.optBoolean("saveRealtime", saveRealtime);
        clamp();
    }
}
