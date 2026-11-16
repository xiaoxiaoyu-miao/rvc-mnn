package com.rvc.mnn;

import java.io.File;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Random;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

/**
 * SGMSE+（Score-based Generative Model for Speech Enhancement）的 Java 运行时。
 *
 * <p>模型：sp-uhh / speechbrain 的 SGMSE+（VoiceBank-DEMAND，16 kHz），
 * 由 {@code tools/sgmse/build_sgmse.py} 用 PyTorch 权重直接生成 ONNX
 * （不依赖 Python onnx）。ONNX 是固定形状的分数网络：
 * 输入 {@code features[1,4,256,T]}、{@code time[1]}，输出 {@code score[1,2,256,T]}，
 * 其中 4 个通道依次是 state.real / state.imag / noisy.real / noisy.imag。
 *
 * <p>本类实现官方 16 kHz 复数 STFT、OUVE predictor/corrector 采样器、
 * 以及分块重叠拼接。采样步数默认 8（原项目的 Fast 档），可选 20/30。
 */
public final class Sgmse {

    public static final int SR = 16000;
    private static final int NFFT = 510;
    private static final int HOP = 128;
    private static final int BINS = NFFT / 2 + 1;   // 256
    /** 训练时的帧数，也是 ONNX 的固定 T。 */
    private static final int CHUNK = 256;
    /** 相邻块重叠 64 帧（0.512 s），前进 192 帧。 */
    private static final int HOP_FRAMES = 192;
    private static final float SPEC_FACTOR = 0.15f;
    private static final float THETA = 1.5f;
    private static final float SIGMA_MIN = 0.05f;
    private static final float SIGMA_MAX = 0.5f;
    private static final float EPSILON = 0.03f;
    private static final float CORRECTOR_SNR = 0.5f;

    public interface Progress {
        void onProgress(int percent);
    }

    private static final Object LOCK = new Object();
    private static OrtEnvironment env;
    private static OrtSession session;
    private static String loadedPath;

    private static final DeepFilter.Fft FFT = new DeepFilter.Fft(NFFT);
    private static final float[] WINDOW = new float[NFFT];

    static {
        // periodic Hann，与 torch.hann_window(N, periodic=True) / rustfft periodic_hann 一致
        for (int i = 0; i < NFFT; i++) {
            WINDOW[i] = (float) (0.5 - 0.5 * Math.cos(2 * Math.PI * i / NFFT));
        }
    }

    private Sgmse() {
    }

    public static synchronized boolean isReady() {
        return session != null;
    }

    public static void ensure(String path) throws Exception {
        synchronized (LOCK) {
            if (session != null && path.equals(loadedPath)) return;
            release();
            long t0 = System.currentTimeMillis();
            RvcLog.i("RvcSGMSE", "加载 SGMSE+ ONNX：" + path);
            env = OrtEnvironment.getEnvironment();
            OrtSession.SessionOptions so = new OrtSession.SessionOptions();
            so.setIntraOpNumThreads(4);
            so.setInterOpNumThreads(1);
            session = env.createSession(path, so);
            loadedPath = path;
            RvcLog.i("RvcSGMSE", "就绪，用时 " + (System.currentTimeMillis() - t0) + "ms");
        }
    }

    public static void release() {
        synchronized (LOCK) {
            if (session != null) {
                try {
                    session.close();
                } catch (Exception ignored) {
                }
            }
            session = null;
            loadedPath = null;
        }
    }

    /**
     * 对 16 kHz 单声道波形做 SGMSE+ 增强。
     *
     * @param steps 采样步数（8=Fast，20=Balanced，30=Quality）
     * @return 等长 16 kHz 波形
     */
    public static float[] process16(float[] x, int steps, Progress cb) throws Exception {
        if (session == null) throw new Exception("SGMSE+ 模型未加载");
        if (x.length == 0) return new float[0];
        if (steps < 2) steps = 2;

        float peak = 0f;
        for (float v : x) peak = Math.max(peak, Math.abs(v));
        if (peak <= 1e-8f) return new float[x.length];

        float[] norm = new float[x.length];
        for (int i = 0; i < x.length; i++) norm[i] = x[i] / peak;

        final long t0 = System.currentTimeMillis();
        final int natural = norm.length / HOP + 1;
        final float[] specRe = new float[BINS * natural];
        final float[] specIm = new float[BINS * natural];
        stft(norm, specRe, specIm, natural);
        RvcLog.i("RvcSGMSE", String.format("输入 %.2fs, %d 帧, %d 步, 分块 %d 帧",
                x.length / (double) SR, natural, steps, CHUNK));
        if (cb != null) cb.onProgress(5);

        final float[] out = new float[x.length];
        final int segLen = (CHUNK - 1) * HOP;      // 32640
        final int hopSamples = HOP_FRAMES * HOP;
        final Random rnd = new Random(0x53474D53452B0030L);
        final float[] nRe = new float[BINS * CHUNK];
        final float[] nIm = new float[BINS * CHUNK];
        int prevStart = -1;

        for (int s = 0; s < natural; s += HOP_FRAMES) {
            if (s * HOP >= norm.length) break;
            java.util.Arrays.fill(nRe, 0f);
            java.util.Arrays.fill(nIm, 0f);
            for (int f = 0; f < CHUNK; f++) {
                int gf = s + f;
                if (gf >= natural) break;
                for (int b = 0; b < BINS; b++) {
                    nRe[b * CHUNK + f] = specRe[b * natural + gf];
                    nIm[b * CHUNK + f] = specIm[b * natural + gf];
                }
            }
            float[] eRe = new float[BINS * CHUNK];
            float[] eIm = new float[BINS * CHUNK];
            sample(nRe, nIm, steps, rnd, eRe, eIm, cb);
            float[] seg = istft(eRe, eIm, CHUNK, segLen);

            int start = s * HOP;
            if (prevStart < 0) {
                int n = Math.min(seg.length, out.length - start);
                System.arraycopy(seg, 0, out, start, n);
            } else {
                int prevEnd = prevStart + segLen;
                int ovLen = Math.max(1, prevEnd - start);
                for (int i = 0; i < seg.length; i++) {
                    int p = start + i;
                    if (p >= out.length) break;
                    if (p < prevEnd) {
                        float a = (p - start) / (float) ovLen;
                        out[p] = out[p] * (1f - a) + seg[i] * a;
                    } else {
                        out[p] = seg[i];
                    }
                }
            }
            prevStart = start;
            if (cb != null) {
                int pct = 5 + (int) (88L * Math.min(natural, s + HOP_FRAMES) / Math.max(1, natural));
                cb.onProgress(Math.min(93, pct));
            }
            RvcLog.i("RvcSGMSE", String.format("  块 %d..%d (%.1fs)",
                    s, Math.min(natural, s + CHUNK), (System.currentTimeMillis() - t0) / 1000.0));
        }
        for (int i = 0; i < out.length; i++) out[i] *= peak;
        RvcLog.i("RvcSGMSE", String.format("完成 %.2fs, 用时 %.1fs",
                x.length / (double) SR, (System.currentTimeMillis() - t0) / 1000.0));
        if (cb != null) cb.onProgress(95);
        return out;
    }

    // ------------------------------------------------------------------ sampler
    private static void sample(float[] noisyRe, float[] noisyIm, int steps, Random rnd,
                               float[] outRe, float[] outIm, Progress cb) throws Exception {
        final int n = noisyRe.length;
        final float priorStd = marginalStd(1f);
        float[] sRe = new float[n];
        float[] sIm = new float[n];
        float[] meanRe = new float[n];
        float[] meanIm = new float[n];
        for (int i = 0; i < n; i++) {
            sRe[i] = noisyRe[i] + (float) rnd.nextGaussian() * priorStd;
            sIm[i] = noisyIm[i] + (float) rnd.nextGaussian() * priorStd;
            meanRe[i] = sRe[i];
            meanIm[i] = sIm[i];
        }

        float[] scRe = new float[n];
        float[] scIm = new float[n];
        for (int stepI = 0; stepI < steps; stepI++) {
            final float time = 1f - (1f - EPSILON) * stepI / (steps - 1f);
            final float next = (stepI + 1 < steps)
                    ? 1f - (1f - EPSILON) * (stepI + 1) / (steps - 1f) : 0f;
            final float dt = time - next;

            // corrector (ALD)
            score(sRe, sIm, noisyRe, noisyIm, time, scRe, scIm);
            final float cstep = 2f * (float) Math.pow(CORRECTOR_SNR * marginalStd(time), 2);
            final float cnoise = (float) Math.sqrt(2f * cstep);
            for (int i = 0; i < n; i++) {
                sRe[i] += scRe[i] * cstep + (float) rnd.nextGaussian() * cnoise;
                sIm[i] += scIm[i] * cstep + (float) rnd.nextGaussian() * cnoise;
            }

            // predictor (reverse diffusion)
            score(sRe, sIm, noisyRe, noisyIm, time, scRe, scIm);
            final float diffusion = (float) (SIGMA_MIN
                    * Math.pow(SIGMA_MAX / SIGMA_MIN, time)
                    * Math.sqrt(2 * Math.log(SIGMA_MAX / SIGMA_MIN)));
            final float variance = diffusion * diffusion * dt;
            final float noise = (float) Math.sqrt(variance);
            for (int i = 0; i < n; i++) {
                float mRe = sRe[i] - THETA * (noisyRe[i] - sRe[i]) * dt + scRe[i] * variance;
                float mIm = sIm[i] - THETA * (noisyIm[i] - sIm[i]) * dt + scIm[i] * variance;
                meanRe[i] = mRe;
                meanIm[i] = mIm;
                sRe[i] = mRe + (float) rnd.nextGaussian() * noise;
                sIm[i] = mIm + (float) rnd.nextGaussian() * noise;
            }
            if (cb != null) {
                // 只占进度条的很小一段；块进度在外面更新
                cb.onProgress(5 + (int) (80L * (stepI + 1) / steps));
            }
        }
        System.arraycopy(meanRe, 0, outRe, 0, n);
        System.arraycopy(meanIm, 0, outIm, 0, n);
    }

    private static float marginalStd(float t) {
        final double logsig = Math.log(SIGMA_MAX / SIGMA_MIN);
        double numerator = Math.pow(SIGMA_MIN, 2)
                * Math.exp(-2 * THETA * t)
                * (Math.exp(2 * (THETA + logsig) * t) - 1)
                * logsig;
        return (float) Math.sqrt(numerator / (THETA + logsig));
    }

    /** 分数网络一次前向。 */
    private static void score(float[] sRe, float[] sIm, float[] nRe, float[] nIm,
                              float time, float[] outRe, float[] outIm) throws Exception {
        final int plane = BINS * CHUNK;
        float[] feat = new float[4 * plane];
        for (int i = 0; i < plane; i++) {
            feat[i] = sRe[i];
            feat[plane + i] = sIm[i];
            feat[2 * plane + i] = nRe[i];
            feat[3 * plane + i] = nIm[i];
        }
        OnnxTensor tf = OnnxTensor.createTensor(env, java.nio.FloatBuffer.wrap(feat),
                new long[]{1, 4, BINS, CHUNK});
        OnnxTensor tt = OnnxTensor.createTensor(env,
                java.nio.FloatBuffer.wrap(new float[]{time}), new long[]{1});
        Map<String, OnnxTensor> in = new LinkedHashMap<>();
        in.put("features", tf);
        in.put("time", tt);
        try (OrtSession.Result r = session.run(in)) {
            java.nio.FloatBuffer fb = ((OnnxTensor) r.get(0)).getFloatBuffer();
            for (int i = 0; i < plane; i++) outRe[i] = fb.get(i);
            for (int i = 0; i < plane; i++) outIm[i] = fb.get(plane + i);
        } finally {
            tf.close();
            tt.close();
        }
    }

    // ------------------------------------------------------------------ STFT
    private static void stft(float[] x, float[] re, float[] im, int frames) {
        float[] fr = new float[NFFT];
        float[] fi = new float[NFFT];
        float[] sr = new float[NFFT];
        float[] si = new float[NFFT];
        for (int f = 0; f < frames; f++) {
            int start = f * HOP;
            for (int i = 0; i < NFFT; i++) {
                int p = reflect(start + i - NFFT / 2, x.length);
                fr[i] = x[p] * WINDOW[i];
                fi[i] = 0f;
            }
            FFT.forward(fr, fi, sr, si);
            for (int b = 0; b < BINS; b++) {
                float m = (float) Math.sqrt(Math.hypot(fr[b], fi[b])) * SPEC_FACTOR;
                double ph = Math.atan2(fi[b], fr[b]);
                re[b * frames + f] = (float) (m * Math.cos(ph));
                im[b * frames + f] = (float) (m * Math.sin(ph));
            }
        }
    }

    private static float[] istft(float[] re, float[] im, int frames, int outLen) {
        int natural = outLen / HOP + 1;
        if (natural > frames) natural = frames;
        int padded = (natural - 1) * HOP + NFFT;
        float[] sig = new float[padded];
        float[] env = new float[padded];
        float[] br = new float[NFFT];
        float[] bi = new float[NFFT];
        float[] sr = new float[NFFT];
        float[] si = new float[NFFT];
        for (int f = 0; f < natural; f++) {
            for (int b = 0; b < BINS; b++) {
                float xr = re[b * frames + f] / SPEC_FACTOR;
                float xi = im[b * frames + f] / SPEC_FACTOR;
                float mag = xr * xr + xi * xi;
                double ph = Math.atan2(xi, xr);
                br[b] = (float) (mag * Math.cos(ph));
                bi[b] = (float) (mag * Math.sin(ph));
            }
            for (int b = BINS; b < NFFT; b++) {
                br[b] = br[NFFT - b];
                bi[b] = -bi[NFFT - b];
            }
            FFT.inverse(br, bi, sr, si);
            int start = f * HOP;
            for (int i = 0; i < NFFT; i++) {
                sig[start + i] += br[i] * WINDOW[i];
                env[start + i] += WINDOW[i] * WINDOW[i];
            }
        }
        for (int i = 0; i < sig.length; i++) {
            if (env[i] > 1e-8f) sig[i] /= env[i];
        }
        int pad = NFFT / 2;
        float[] out = new float[outLen];
        int copy = Math.min(outLen, Math.max(0, sig.length - pad));
        System.arraycopy(sig, pad, out, 0, copy);
        return out;
    }

    /** reflect（不含边缘重复）padding，与 torch.stft(center=True, pad_mode='reflect') 一致。 */
    private static int reflect(int i, int n) {
        if (n <= 1) return 0;
        int m = i;
        while (m < 0 || m >= n) {
            if (m < 0) m = -m;
            if (m >= n) m = 2 * n - 2 - m;
        }
        return m;
    }

    /** 合并用低通（零相位），供 {@link Diffusion} 做高频保留。 */
    static float[] zeroPhaseLowpass(float[] x, double fc, int r) {
        float[] h = new float[2 * r + 1];
        double sum = 0;
        for (int j = 0; j <= 2 * r; j++) {
            int d = j - r;
            double v = 2 * fc * sinc(2 * fc * d) * kaiser(d / (double) r);
            h[j] = (float) v;
            sum += v;
        }
        for (int j = 0; j < h.length; j++) h[j] /= (float) sum;
        float[] y = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            float acc = 0f;
            int lo = Math.max(-r, -i);
            int hi = Math.min(r, x.length - 1 - i);
            for (int d = lo; d <= hi; d++) acc += h[d + r] * x[i + d];
            y[i] = acc;
        }
        return y;
    }

    private static double sinc(double x) {
        if (Math.abs(x) < 1e-9) return 1.0;
        double a = Math.PI * x;
        return Math.sin(a) / a;
    }

    private static double kaiser(double u) {
        if (u <= -1.0 || u >= 1.0) return 0.0;
        double r = Math.sqrt(1.0 - u * u);
        return besselI0(8.0 * r) / besselI0(8.0);
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
}
