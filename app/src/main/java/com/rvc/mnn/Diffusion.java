package com.rvc.mnn;

import android.content.Context;

import java.io.File;

/**
 * Diffusion 精修阶段：接在「RVC 推理 → DFN3 降噪」之后。
 *
 * <p>当前实现是 <b>SGMSE+</b>（sp-uhh / speechbrain 的 16 kHz VoiceBank-DEMAND 模型）：
 * 一个 65M 参数的复数 STFT 域扩散分数网络，配合 8/20/30 步 OUVE predictor-corrector 采样。
 * 模型本身不随 APK 分发（约 262 MB），放到模型目录即可：
 * <pre>
 *   /sdcard/models/mm/sgmse.onnx
 * </pre>
 * 或把 {@link Config#diffusionModel} 改成绝对路径。找不到模型时 {@link #isAvailable} 为 false，
 * 调用方会跳过这一段，不影响前面的推理/降噪结果。
 *
 * <p><b>采样率</b>：SGMSE+ 固定 16 kHz。RVC 输出是 40/48 kHz，所以这里先降到 16k 做扩散，
 * 再把增强后的低频与原始 RVC 输出的高频拼接（{@link Config#diffusionKeepHigh}），
 * 最终仍按 RVC 的采样率写回。
 */
public final class Diffusion {

    public interface Progress {
        void onProgress(int percent);
    }

    private Diffusion() {
    }

    /** 解析模型路径；相对路径按 modelDir 解析，再回退到 /sdcard/models/mm。不存在返回 null。 */
    public static File modelFile(Config cfg) {
        if (cfg == null || cfg.diffusionModel == null || cfg.diffusionModel.trim().isEmpty()) {
            return null;
        }
        String p = cfg.diffusionModel.trim();
        File f = new File(p);
        if (!f.isAbsolute()) {
            if (cfg.modelDir != null && !cfg.modelDir.isEmpty()) {
                File a = new File(cfg.modelDir, p);
                if (a.isFile()) return a;
            }
            File b = new File(Config.DEF_MODEL_DIR, p);
            if (b.isFile()) return b;
            return null;
        }
        return f.isFile() ? f : null;
    }

    public static boolean isAvailable(Config cfg) {
        return modelFile(cfg) != null;
    }

    public static String describe(Config cfg) {
        File f = modelFile(cfg);
        return f != null ? f.getAbsolutePath() : ("未找到 " + cfg.diffusionModel);
    }

    /**
     * 整段跑一遍 SGMSE+，写回与输入同采样率的 WAV。
     *
     * @return 输出时长（秒）
     */
    public static double refine(Context ctx, Config cfg, String inWav, String outWav,
                                Progress cb) throws Exception {
        File model = modelFile(cfg);
        if (model == null) {
            throw new Exception("SGMSE+ 模型未配置：" + cfg.diffusionModel
                    + "（放到 " + Config.DEF_MODEL_DIR + " 或 " + cfg.modelDir + "）");
        }
        final long t0 = System.currentTimeMillis();
        int[] sr = new int[1];
        float[] x = WavIo.readWav(inWav, sr);
        if (x.length == 0) throw new Exception("Diffusion 输入为空");
        final int outSr = sr[0];
        if (cb != null) cb.onProgress(2);

        Sgmse.ensure(model.getAbsolutePath());
        float[] x16 = to16k(x, outSr);
        if (cb != null) cb.onProgress(4);

        final int[] last = {-10};
        float[] enh16 = Sgmse.process16(x16, cfg.diffusionSteps, p -> {
            if (cb == null) return;
            if (p < last[0] + 3 && p < 95) return;
            last[0] = p;
            cb.onProgress(5 + p * 88 / 100);
        });
        if (cb != null) cb.onProgress(94);

        float[] enh = from16k(enh16, outSr);
        float[] y;
        if (cfg.diffusionKeepHigh && outSr > 16000) {
            y = mergeHighBand(x, enh, outSr);
        } else {
            y = new float[x.length];
            System.arraycopy(enh, 0, y, 0, Math.min(enh.length, y.length));
        }
        WavIo.writeWav(outWav, y, outSr, 1f);

        double sec = y.length / (double) outSr;
        double el = (System.currentTimeMillis() - t0) / 1000.0;
        RvcLog.i("RvcSGMSE", String.format("完成 %.2fs @ %d Hz, 用时 %.1fs（%d 步）",
                sec, outSr, el, cfg.diffusionSteps));
        if (cb != null) cb.onProgress(100);
        return sec;
    }

    // ------------------------------------------------------------------ sample rate
    private static float[] to16k(float[] x, int sr) {
        if (sr == 16000) return x;
        if (sr == 48000) return DeepFilter.resampleFrom48(x, 16000);
        float[] x48 = WavIo.resample(x, sr, 48000);
        return DeepFilter.resampleFrom48(x48, 16000);
    }

    private static float[] from16k(float[] enh, int outSr) {
        if (outSr == 16000) return enh;
        float[] up48 = DeepFilter.upsample3(enh);
        if (outSr == 48000) return up48;
        return DeepFilter.resampleFrom48(up48, outSr);
    }

    /**
     * 用增强后的 0~8 kHz 替换原信号低频，保留原信号高频：
     * {@code out = orig + lowpass(enh - orig)}。
     */
    private static float[] mergeHighBand(float[] orig, float[] enh, int sr) {
        int n = Math.min(orig.length, enh.length);
        float[] diff = new float[n];
        for (int i = 0; i < n; i++) diff[i] = enh[i] - orig[i];
        float fc = (float) Math.min(7600.0 / sr, 0.45);
        float[] lp = Sgmse.zeroPhaseLowpass(diff, fc, 31);
        float[] out = new float[orig.length];
        for (int i = 0; i < n; i++) out[i] = orig[i] + lp[i];
        for (int i = n; i < orig.length; i++) {
            out[i] = i < enh.length ? enh[i] : 0f;
        }
        return out;
    }
}
