package com.rvc.mnn;

/**
 * 变声链路共用的 DSP 小工具（Activity 的离线转换和 Service 的实时循环都要用）。
 */
public final class RvcDsp {
    public static final int SR_IN = 16000;

    /**
     * 输出采样率。<b>不再写死 40k</b>：由合成器每个窗口吐出的样本数推导
     * （见 {@link #deriveSrOut}），所以 40k 模型 → 40000、48k 模型 → 48000，
     * 两种模型都能直接用。{@link #srOutKnown} 为 false 时这只是个不可信的默认值。
     */
    public static volatile int SR_OUT = 40000;

    /** {@link #SR_OUT} 是否已经按实际加载的模型推导过。 */
    public static volatile boolean srOutKnown = false;

    /**
     * 由合成器输出窗口长度推导输出采样率：{@code SR_IN × audioLen / seg}。
     * 例：49 帧 × 400 = 19600 → 40000；49 帧 × 480 = 23520 → 48000。
     */
    public static int deriveSrOut(int audioLen, int seg) {
        if (audioLen <= 0 || seg <= 0) return SR_OUT;
        long r = Math.round((double) audioLen * SR_IN / seg);
        return (r >= 8000 && r <= 192000) ? (int) r : SR_OUT;
    }

    /** 按模型实际输出长度更新 {@link #SR_OUT}；返回是否发生了改变。 */
    public static boolean applySrOut(int audioLen, int seg) {
        int r = deriveSrOut(audioLen, seg);
        srOutKnown = true;
        if (r == SR_OUT) return false;
        SR_OUT = r;
        return true;
    }

    private RvcDsp() {}

    /** f0 → 音高编码表的粗略 mel 分档（1..255）。 */
    public static int melCoarse(float f) {
        if (f <= 0) return 1;
        double mel = 1127.0 * Math.log(f / 700.0 + 1.0);
        double fmn = 1127.0 * Math.log(50.0 / 700.0 + 1.0);
        double fmx = 1127.0 * Math.log(1100.0 / 700.0 + 1.0);
        return (int) Math.max(1, Math.min(255, Math.round((mel - fmn) * 254.0 / (fmx - fmn) + 1.0)));
    }

    /** f0 倍率变调：k=2^(semi/12)，对应 RVC 的 f0_up_key。 */
    public static void shiftF0(float[] f0, float ratio) {
        if (ratio == 1f) return;
        for (int i = 0; i < f0.length; i++) {
            if (f0[i] > 0) f0[i] *= ratio;
        }
    }

    /**
     * 逐帧自适应噪声门（移植自 MeowRVC 的 adaptiveNoiseGate）。
     *
     * <p>麦克风拾到扬声器的变声输出后，会被 hubert/fcpe 当成"人声"继续合成，
     * 形成「扬声器→麦→再合成→扬声器」的自激循环，听感就是一直有声音在响。
     * 这里在合成输出上按帧做能量门控：低于 thresholdDb 的帧被压到 0.1 倍增益，
     * 带 12dB 软拐点 + 慢速攻击/释放（避免泵动感），把循环尾巴掐掉。
     *
     * @param audio       合成器输出的音频（原地改写）
     * @param thresholdDb 门限（dB，0=关闭）
     */
    public static void applyFeedbackGate(float[] audio, float thresholdDb) {
        if (audio == null || audio.length == 0 || thresholdDb <= 0f) return;
        final int frameSize = Math.max(1, SR_OUT / 50);      // 20ms 帧
        final float minGain = 0.1f, att = 0.05f, rel = 0.02f, kneeDb = 12f;
        float gain = 1.0f;
        for (int off = 0; off < audio.length; off += frameSize) {
            int end = Math.min(off + frameSize, audio.length);
            double sum = 0;
            for (int i = off; i < end; i++) sum += audio[i] * audio[i];
            double rms = Math.sqrt(sum / (end - off));
            double frameDb = 20.0 * Math.log10(Math.max(rms, 1e-10)) + 100.0;
            float target;
            if (frameDb >= thresholdDb) {
                target = 1.0f;
            } else if (frameDb >= thresholdDb - kneeDb) {
                float r = (float) ((frameDb - (thresholdDb - kneeDb)) / kneeDb);
                r = Math.max(0f, Math.min(1f, r));
                target = minGain + (1f - minGain) * r * r;
            } else {
                target = minGain;
            }
            gain += (target - gain) * (target > gain ? att : rel);
            for (int i = off; i < end; i++) audio[i] *= gain;
        }
    }

    /** 线性插值重采样 from→to。 */
    public static float[] resample(float[] x, int from, int to) {
        if (from == to) return x;
        int n = (int) Math.round(x.length * (double) to / from);
        float[] y = new float[n];
        for (int i = 0; i < n; i++) {
            double pos = i * (double) from / to;
            int i0 = (int) pos;
            int i1 = Math.min(i0 + 1, x.length - 1);
            float fr = (float) (pos - i0);
            y[i] = x[i0] * (1 - fr) + x[i1] * fr;
        }
        return y;
    }

    public static short[] toShorts(float[] y, float gain, int n) {
        short[] s = new short[n];
        for (int i = 0; i < n && i < y.length; i++) {
            float v = y[i] * gain;
            if (v > 1f) v = 1f;
            else if (v < -1f) v = -1f;
            s[i] = (short) (v * 32767f);
        }
        return s;
    }
}
