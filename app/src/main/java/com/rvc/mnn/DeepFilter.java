package com.rvc.mnn;

import android.content.Context;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Optional;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

/**
 * DeepFilterNet3 神经网络降噪（官方 48 kHz checkpoint 的 ONNX 神经图 + 本类里手写的 DSP）。
 *
 * <p>模型只包含网络：输入是归一化后的 ERB / 复数谱特征，输出是 ERB mask 与 5 阶深层滤波系数。
 * 前后处理（STFT、ERB 归一化、mask 反变换、深层滤波、OLA）全部在这里实现，严格按
 * {@code rvc-train/dfn} 里的参考实现：
 * <ul>
 *   <li>48 kHz、960 点 FFT、hop 480、Vorbis 窗；</li>
 *   <li>32 个不相交 ERB 带（aux 文件里的 481×32 矩阵）；</li>
 *   <li>归一化 alpha=0.99：ERB 均值状态从 −60→−90 dB，复数单元状态从 0.001→0.0001；</li>
 *   <li>深层滤波 order=5 / lookahead=2，只作用于前 96 个频点，其余频点用 ERB 掩码。</li>
 * </ul>
 *
 * <p><b>和 RVC 的衔接</b>：RVC 的入口是 16 kHz，模型固定 48 kHz。
 * 现在降噪放在<b>推理之后</b>（处理 RVC 输出的 40/48 kHz 音频），所以输出采样率由调用方指定：
 * 40 kHz 的合成器输出就还原成 40 kHz，不再统一压到 16 kHz。
 */
public final class DeepFilter {

    /** 模型/DSP 工作采样率。 */
    public static final int SR = 48000;

    private static final int NFFT = 960;
    private static final int HOP = 480;
    private static final int BINS = NFFT / 2 + 1;      // 481
    private static final int NB_ERB = 32;
    private static final int NB_DF = 96;
    private static final int DF_ORDER = 5;
    private static final float ALPHA = 0.99f;

    /** 一次喂给 ONNX 的最大帧数（约 24 s）；长音频分块，避免输出张量把内存吃穿。 */
    private static final int CHUNK_FRAMES = 2400;
    /** 每块两侧多喂的上下文帧（GRU 预热 + 卷积 lookahead）。 */
    private static final int CTX_FRAMES = 192;

    private static final int UP_R = 12;                 // 3× 插值每个相位 25 抽头
    private static final int DN_R = 31;                 // 3× 抽取 63 抽头
    private static final double KAISER_BETA = 8.0;

    private static final String ASSET_DIR = "dfn";
    private static final String MODEL_ASSET = "deepfilter.onnx";
    private static final String AUX_ASSET = "deepfilter-auxiliary.bin";

    /** 进度回调（0..100）。 */
    public interface Progress {
        void onProgress(int percent);
    }

    // ------------------------------------------------------------------ 模型单例
    private static final Object LOCK = new Object();
    private static OrtEnvironment env;
    private static OrtSession session;
    private static float[] win;
    private static int[] bandOf;       // 每个 FFT 频点属于哪个 ERB 带
    private static float[] bandWeight; // 该频点在 ERB 带内的权重（1/带宽）

    private static final Fft FFT = new Fft(NFFT);
    private static final float[][] UP = new float[3][2 * UP_R + 1];
    private static final float[] DOWN = new float[2 * DN_R + 1];

    static {
        buildResamplers();
    }

    private DeepFilter() {
    }

    /** 模型是否已加载。 */
    public static synchronized boolean isReady() {
        return session != null;
    }

    /** 懒加载 assets 里的模型与 DSP 常量。 */
    public static void ensure(Context ctx) throws Exception {
        synchronized (LOCK) {
            if (session != null) return;
            long t0 = System.currentTimeMillis();
            RvcLog.i("RvcDFN", "加载 DeepFilterNet3（assets/" + ASSET_DIR + "）…");
            env = OrtEnvironment.getEnvironment();

            byte[] model = readAll(ctx.getAssets().open(ASSET_DIR + "/" + MODEL_ASSET));
            OrtSession.SessionOptions so = new OrtSession.SessionOptions();
            so.setIntraOpNumThreads(3);
            so.setInterOpNumThreads(1);
            session = env.createSession(model, so);
            model = null;

            byte[] aux = readAll(ctx.getAssets().open(ASSET_DIR + "/" + AUX_ASSET));
            if (aux.length < 4 * (BINS * NB_ERB * 2 + NFFT)) {
                throw new Exception("aux 文件长度不对: " + aux.length);
            }
            FloatBuffer fb = ByteBuffer.wrap(aux).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
            float[] fwd = new float[BINS * NB_ERB];
            fb.get(fwd);
            // 逆 ERB 矩阵：不相交频带时每个频点只属于一个带，mask@inv 等价于按带取值，
            // 所以这里只保留「频点 → 带」的映射，省掉每帧 1.5 万次乘加。
            bandOf = new int[BINS];
            bandWeight = new float[BINS];
            for (int k = 0; k < BINS; k++) {
                int best = 0;
                float bestV = 0f;
                for (int e = 0; e < NB_ERB; e++) {
                    float v = fwd[k * NB_ERB + e];
                    if (v > bestV) {
                        bestV = v;
                        best = e;
                    }
                }
                bandOf[k] = best;
                bandWeight[k] = bestV;
            }
            // 跳过逆 ERB 矩阵（32×481），下一段才是 960 点 Vorbis 窗
            fb.position(fb.position() + NB_ERB * BINS);
            win = new float[NFFT];
            fb.get(win);

            RvcLog.i("RvcDFN", "就绪：" + (System.currentTimeMillis() - t0) + "ms, 输入 "
                    + session.getInputNames() + ", 输出 " + session.getOutputNames());
        }
    }

    private static byte[] readAll(InputStream in) throws Exception {
        try (InputStream i = in) {
            ByteArrayOutputStream bo = new ByteArrayOutputStream(1 << 20);
            byte[] buf = new byte[1 << 16];
            int n;
            while ((n = i.read(buf)) > 0) bo.write(buf, 0, n);
            return bo.toByteArray();
        }
    }

    /** 神经图执行器：生产环境走 ORT，单元测试可以喂 dump 出来的 mask/coefs。 */
    interface MaskRunner {
        void run(float[] erbIn, float[] specIn, int nf, float[] maskOut, float[] coefsOut)
                throws Exception;
    }

    private static final MaskRunner ORT_RUNNER = (erbIn, specIn, nf, maskOut, coefsOut) -> {
        Map<String, OnnxTensor> inputs = new LinkedHashMap<>();
        OnnxTensor tErb = OnnxTensor.createTensor(env, FloatBuffer.wrap(erbIn),
                new long[]{1, 1, nf, NB_ERB});
        OnnxTensor tSpec = OnnxTensor.createTensor(env, FloatBuffer.wrap(specIn),
                new long[]{1, 2, nf, NB_DF});
        inputs.put("feat_erb", tErb);
        inputs.put("feat_spec", tSpec);
        try (OrtSession.Result res = session.run(inputs)) {
            Optional<OnnxValue> mo = res.get("erb_mask");
            Optional<OnnxValue> co = res.get("df_coefs");
            if (!mo.isPresent() || !co.isPresent()) {
                throw new Exception("模型输出名不符: " + session.getOutputNames());
            }
            FloatBuffer mb = ((OnnxTensor) mo.get()).getFloatBuffer();
            FloatBuffer cb = ((OnnxTensor) co.get()).getFloatBuffer();
            mb.get(maskOut);
            cb.get(coefsOut);
        } finally {
            tErb.close();
            tSpec.close();
        }
    };

    // ------------------------------------------------------------------ 对外入口
    /**
     * 对任意 WAV 做一次 DFN3 降噪，输出 16 kHz 单声道 WAV。
     *
     * @return 输出时长（秒）
     */
    /**
     * 对任意 WAV 做一次 DFN3 降噪，输出到指定采样率（16-bit 单声道）。
     *
     * <p>典型用法：<b>推理后降噪</b>。RVC 输出 40/48 kHz，直接按原采样率返回，
     * 不要压成 16 kHz——那一步是给 RVC 入口用的。
     *
     * @param outSr 输出采样率（&lt;=0 表示 48000）
     * @return 输出时长（秒）
     */
    public static double denoiseWav(Context ctx, String inWav, String outWav, int outSr,
                                    Progress cb) throws Exception {
        ensure(ctx);
        final long t0 = System.currentTimeMillis();
        if (outSr <= 0) outSr = SR;

        int[] sr = new int[1];
        float[] x = WavIo.readWav(inWav, sr);
        if (x.length == 0) throw new Exception("降噪输入为空");
        if (cb != null) cb.onProgress(2);
        RvcLog.i("RvcDFN", String.format("输入 %.2fs @ %d Hz -> 输出 @ %d Hz",
                x.length / (double) Math.max(1, sr[0]), sr[0], outSr));

        float[] x48;
        if (sr[0] == SR) {
            x48 = x;
        } else if (sr[0] == RvcDsp.SR_IN) {
            x48 = upsample3(x);
        } else {
            // 少见采样率：先线性重采样到 48k（模型固定 48k）
            x48 = WavIo.resample(x, sr[0], SR);
            x = null;
        }
        if (cb != null) cb.onProgress(5);

        float[] y48 = enhance48(x48, cb);
        if (cb != null) cb.onProgress(94);

        float[] y;
        if (outSr == SR) {
            y = y48;
        } else if (outSr == RvcDsp.SR_IN) {
            y = downsample3(y48);
        } else {
            y = resampleFrom48(y48, outSr);
        }
        WavIo.writeWav(outWav, y, outSr, 1f);

        double sec = y.length / (double) outSr;
        double el = (System.currentTimeMillis() - t0) / 1000.0;
        RvcLog.i("RvcDFN", String.format("完成 %.2fs, 用时 %.2fs (%.2fx realtime)",
                sec, el, sec / Math.max(el, 1e-6)));
        if (cb != null) cb.onProgress(100);
        return sec;
    }

    // ------------------------------------------------------------------ 增强主流程
    /**
     * 对 48 kHz 波形做 DFN3 增强，返回等长 48 kHz 波形。
     * 内存占用约为「2 倍波形 + 特征（T×224 float）」，5 分钟录音也在百 MB 以内。
     */
    public static float[] enhance48(float[] x, Progress cb) throws Exception {
        if (session == null) throw new Exception("DFN 模型未加载");
        return enhance48(x, cb, ORT_RUNNER);
    }

    /** 同 {@link #enhance48(float[], Progress)}，但允许注入神经图执行器（离线复算/单测用）。 */
    static float[] enhance48(float[] x, Progress cb, MaskRunner runner) throws Exception {
        final int orig = x.length;
        // 前补 1 hop、后补 1 个 FFT：与 libdf 的算法延迟对齐，尾部够做完最后一次 OLA。
        final int T = (orig + HOP) / HOP + 1;
        RvcLog.i("RvcDFN", "帧数 T=" + T + "（" + (T - 1) * HOP + " 个 48k 采样）");

        final float[] fe = new float[T * NB_ERB];
        final float[] fsRe = new float[T * NB_DF];
        final float[] fsIm = new float[T * NB_DF];

        // ---- 1) STFT + 归一化特征（状态必须顺序递推，所以整段按时间顺序算一遍）----
        float[] mean = new float[NB_ERB];
        float[] unit = new float[NB_DF];
        for (int e = 0; e < NB_ERB; e++) mean[e] = -60f + (-30f) * e / (NB_ERB - 1f);
        for (int k = 0; k < NB_DF; k++) unit[k] = 0.001f + (0.0001f - 0.001f) * k / (NB_DF - 1f);

        float[] re = new float[NFFT];
        float[] im = new float[NFFT];
        float[] sr = new float[NFFT];
        float[] si = new float[NFFT];
        float[] power = new float[BINS];
        float[] erb = new float[NB_ERB];

        int lastPct = -1;
        for (int t = 0; t < T; t++) {
            analyze(x, t, re, im, sr, si);
            for (int k = 0; k < BINS; k++) power[k] = re[k] * re[k] + im[k] * im[k];
            java.util.Arrays.fill(erb, 0f);
            for (int k = 0; k < BINS; k++) erb[bandOf[k]] += power[k] * bandWeight[k];
            for (int e = 0; e < NB_ERB; e++) {
                float db = 10f * (float) Math.log10(erb[e] + 1e-10f);
                mean[e] = db * (1f - ALPHA) + mean[e] * ALPHA;
                fe[t * NB_ERB + e] = (db - mean[e]) / 40f;
            }
            for (int k = 0; k < NB_DF; k++) {
                float mag = (float) Math.sqrt(power[k]);
                unit[k] = mag * (1f - ALPHA) + unit[k] * ALPHA;
                float s = 1f / (float) Math.sqrt(unit[k]);
                fsRe[t * NB_DF + k] = re[k] * s;
                fsIm[t * NB_DF + k] = im[k] * s;
            }
            if (cb != null) {
                int pct = 5 + (int) (20L * t / Math.max(1, T));
                if (pct != lastPct) {
                    lastPct = pct;
                    cb.onProgress(pct);
                }
            }
        }

        // ---- 2) 分块跑网络 + 逐帧滤波 / OLA ----
        final float[] y = new float[orig];
        final float[] sRe = new float[(CHUNK_FRAMES + 4) * BINS];
        final float[] sIm = new float[(CHUNK_FRAMES + 4) * BINS];
        final float[] outRe = new float[BINS];
        final float[] outIm = new float[BINS];
        final float[] zr = new float[NFFT];
        final float[] zi = new float[NFFT];
        final float[] ir = new float[NFFT];
        final float[] ii = new float[NFFT];

        for (int c0 = 0; c0 < T; c0 += CHUNK_FRAMES) {
            final int c1 = Math.min(T, c0 + CHUNK_FRAMES);
            final int a = Math.max(0, c0 - CTX_FRAMES);
            final int b = Math.min(T, c1 + CTX_FRAMES);
            final int nf = b - a;

            float[] erbIn = new float[nf * NB_ERB];
            float[] specIn = new float[2 * nf * NB_DF];
            System.arraycopy(fe, a * NB_ERB, erbIn, 0, nf * NB_ERB);
            System.arraycopy(fsRe, a * NB_DF, specIn, 0, nf * NB_DF);
            System.arraycopy(fsIm, a * NB_DF, specIn, nf * NB_DF, nf * NB_DF);

            float[] mask = new float[nf * NB_ERB];
            float[] coefs = new float[DF_ORDER * nf * NB_DF * 2];
            runner.run(erbIn, specIn, nf, mask, coefs);

            // 本块输出需要的含噪谱：[c0-2, c1+2)，深层滤波的 lookahead 是 2 帧
            final int sa = Math.max(0, c0 - 2);
            final int sb = Math.min(T, c1 + 2);
            for (int f = sa; f < sb; f++) {
                analyze(x, f, re, im, sr, si);
                int off = (f - sa) * BINS;
                System.arraycopy(re, 0, sRe, off, BINS);
                System.arraycopy(im, 0, sIm, off, BINS);
            }

            for (int t = c0; t < c1; t++) {
                final int row = t - a;

                // ERB mask -> 每个频点的实增益
                for (int k = 0; k < BINS; k++) {
                    float g = mask[row * NB_ERB + bandOf[k]];
                    int si2 = (t - sa) * BINS + k;
                    outRe[k] = sRe[si2] * g;
                    outIm[k] = sIm[si2] * g;
                }

                // 深层滤波：out[t] = Σ_{i=0..4} spec[t-2+i] * coefs[i, t]
                for (int k = 0; k < NB_DF; k++) {
                    float accR = 0f, accI = 0f;
                    for (int i = 0; i < DF_ORDER; i++) {
                        int f = t - 2 + i;
                        if (f < sa || f >= sb) continue;
                        int ci = ((i * nf + row) * NB_DF + k) * 2;
                        float cr = coefs[ci];
                        float cim = coefs[ci + 1];
                        int sp = (f - sa) * BINS + k;
                        float xr = sRe[sp], xi = sIm[sp];
                        accR += xr * cr - xi * cim;
                        accI += xr * cim + xi * cr;
                    }
                    outRe[k] = accR;
                    outIm[k] = accI;
                }

                // 半谱 -> 共轭对称全谱 -> IFFT
                java.util.Arrays.fill(zr, 0f);
                java.util.Arrays.fill(zi, 0f);
                zr[0] = outRe[0];
                for (int k = 1; k < BINS - 1; k++) {
                    zr[k] = outRe[k];
                    zi[k] = outIm[k];
                    zr[NFFT - k] = outRe[k];
                    zi[NFFT - k] = -outIm[k];
                }
                zr[BINS - 1] = outRe[BINS - 1]; // Nyquist 必须是实数
                FFT.inverse(zr, zi, ir, ii);

                // Vorbis 窗 + OLA；减去 1 个 hop 的算法延迟
                final int base = t * HOP - HOP;
                for (int i = 0; i < NFFT; i++) {
                    int p = base + i;
                    if (p >= 0 && p < orig) y[p] += zr[i] * win[i];
                }
            }

            if (cb != null) {
                int pct = 25 + (int) (68L * c1 / Math.max(1, T));
                cb.onProgress(Math.min(93, pct));
            }
            RvcLog.i("RvcDFN", String.format("  %d/%d 帧 (%.1fs)", c1, T, c1 * HOP / (double) SR));
        }
        return y;
    }

    // ------------------------------------------------------------------ STFT
    /** 分析第 t 帧（补 1 hop 前导零），结果在 re/im 里（长度 960）。 */
    private static void analyze(float[] x, int t, float[] re, float[] im,
                                float[] sr, float[] si) {
        final int base = t * HOP - HOP;
        for (int i = 0; i < NFFT; i++) {
            int p = base + i;
            re[i] = (p >= 0 && p < x.length) ? x[p] * win[i] : 0f;
            im[i] = 0f;
        }
        FFT.forward(re, im, sr, si);
    }

    // ------------------------------------------------------------------ 3× 重采样
    private static void buildResamplers() {
        // 16k -> 48k：每输出 3 个采样一个相位，理想 sinc + Kaiser 窗
        for (int p = 0; p < 3; p++) {
            double sum = 0;
            for (int j = 0; j <= 2 * UP_R; j++) {
                int d = j - UP_R;
                double u = p / 3.0 - d;
                double v = sinc(u) * kaiser(u / UP_R);
                UP[p][j] = (float) v;
                sum += v;
            }
            for (int j = 0; j < UP[p].length; j++) UP[p][j] /= (float) sum;
        }
        // 48k -> 16k：截止 8 kHz 的低通（fc = 1/6 cycles/sample），再每 3 个取 1 个
        double fc = 1.0 / 6.0;
        double sum = 0;
        for (int j = 0; j <= 2 * DN_R; j++) {
            int d = j - DN_R;
            double v = 2 * fc * sinc(2 * fc * d) * kaiser(d / (double) DN_R);
            DOWN[j] = (float) v;
            sum += v;
        }
        for (int j = 0; j < DOWN.length; j++) DOWN[j] /= (float) sum;
    }

    private static double sinc(double x) {
        if (Math.abs(x) < 1e-9) return 1.0;
        double a = Math.PI * x;
        return Math.sin(a) / a;
    }

    private static double kaiser(double u) {
        if (u <= -1.0 || u >= 1.0) return 0.0;
        double r = Math.sqrt(1.0 - u * u);
        return besselI0(KAISER_BETA * r) / besselI0(KAISER_BETA);
    }

    private static double besselI0(double x) {
        double sum = 1.0, term = 1.0, y = x * x / 4.0;
        for (int k = 1; k < 64; k++) {
            term *= y / (k * k);
            sum += term;
            if (term < sum * 1e-12) break;
        }
        return sum;
    }

    /** 16k -> 48k，3 相位多相插值。 */
    public static float[] upsample3(float[] x) {
        int n = x.length;
        float[] y = new float[n * 3];
        for (int m = 0; m < n; m++) {
            for (int p = 0; p < 3; p++) {
                float acc = 0f;
                float[] c = UP[p];
                for (int j = 0; j < c.length; j++) {
                    int idx = m + j - UP_R;
                    if (idx >= 0 && idx < n) acc += c[j] * x[idx];
                }
                y[m * 3 + p] = acc;
            }
        }
        return y;
    }

    /** 48k -> 16k，63 抽头低通 + 3 倍抽取。 */
    public static float[] downsample3(float[] x) {
        int n = (x.length + 2) / 3;
        float[] y = new float[n];
        for (int i = 0; i < n; i++) {
            int center = i * 3;
            float acc = 0f;
            for (int j = 0; j < DOWN.length; j++) {
                int idx = center + j - DN_R;
                if (idx >= 0 && idx < x.length) acc += DOWN[j] * x[idx];
            }
            y[i] = acc;
        }
        return y;
    }

    /**
     * 48k -> 任意输出采样率（推理后降噪用，比如 RVC 合成器的 40 kHz）。
     * 降采样时先做 63 抽头抗混叠低通，再线性插值；升采样直接线性插值。
     */
    public static float[] resampleFrom48(float[] x, int outSr) {
        if (outSr == SR) return x;
        if (outSr <= 0) return x;
        if (outSr > SR) return WavIo.resample(x, SR, outSr);
        float fc = 0.45f * outSr / SR;              // 截止频率（cycles/sample @48k）
        float[] h = designLowpass(fc, DN_R);
        float[] lp = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            float acc = 0f;
            int lo = Math.max(-DN_R, -i);
            int hi = Math.min(DN_R, x.length - 1 - i);
            for (int d = lo; d <= hi; d++) acc += h[d + DN_R] * x[i + d];
            lp[i] = acc;
        }
        return WavIo.resample(lp, SR, outSr);
    }

    private static float[] designLowpass(double fc, int r) {
        float[] h = new float[2 * r + 1];
        double sum = 0;
        for (int j = 0; j <= 2 * r; j++) {
            int d = j - r;
            double v = 2 * fc * sinc(2 * fc * d) * kaiser(d / (double) r);
            h[j] = (float) v;
            sum += v;
        }
        for (int j = 0; j < h.length; j++) h[j] /= (float) sum;
        return h;
    }

    // ------------------------------------------------------------------ 任意长度 FFT
    /**
     * 混合基（2/3/5）Cooley–Tukey FFT。960 = 2^6·3·5，不是 2 的幂，
     * 所以不能用最常见的 radix-2 实现；Bluestein 又太慢，这里直接做混合基递归。
     */
    static final class Fft {
        final int n;
        final float[] cos;
        final float[] sin;

        Fft(int n) {
            this.n = n;
            cos = new float[n];
            sin = new float[n];
            for (int k = 0; k < n; k++) {
                double a = 2 * Math.PI * k / n;
                cos[k] = (float) Math.cos(a);
                sin[k] = (float) Math.sin(a);
            }
        }

        void forward(float[] re, float[] im, float[] sr, float[] si) {
            rec(re, im, 0, 1, n, sr, si);
        }

        void inverse(float[] re, float[] im, float[] sr, float[] si) {
            for (int i = 0; i < n; i++) im[i] = -im[i];
            rec(re, im, 0, 1, n, sr, si);
            float inv = 1f / n;
            for (int i = 0; i < n; i++) {
                re[i] *= inv;
                im[i] = -im[i] * inv;
            }
        }

        private void rec(float[] re, float[] im, int off, int stride, int len,
                         float[] sr, float[] si) {
            if (len <= 1) return;
            int p = smallestFactor(len);
            int m = len / p;
            for (int j = 0; j < p; j++) {
                rec(re, im, off + j * stride, stride * p, m, sr, si);
            }
            int levelN = len;
            int scale = n / levelN;
            for (int q = 0; q < m; q++) {
                for (int r = 0; r < p; r++) {
                    float sumR = 0f, sumI = 0f;
                    for (int j = 0; j < p; j++) {
                        int idx = off + (j + q * p) * stride;
                        float fr = re[idx], fi = im[idx];
                        int total = (int) (((long) j * q + (long) j * r * m) % levelN);
                        int ti = (total * scale) % n;
                        float c = cos[ti];
                        float s = -sin[ti];
                        sumR += fr * c - fi * s;
                        sumI += fr * s + fi * c;
                    }
                    sr[q + m * r] = sumR;
                    si[q + m * r] = sumI;
                }
            }
            for (int k = 0; k < len; k++) {
                int idx = off + k * stride;
                re[idx] = sr[k];
                im[idx] = si[k];
            }
        }

        private static int smallestFactor(int n) {
            if ((n & 1) == 0) return 2;
            for (int i = 3; (long) i * i <= n; i += 2) {
                if (n % i == 0) return i;
            }
            return n;
        }
    }
}
