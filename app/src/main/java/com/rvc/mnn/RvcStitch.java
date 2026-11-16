package com.rvc.mnn;

/**
 * 块间接缝处理：**重叠窗口 + 交叉淡化**。
 *
 * <p>为什么需要它——「每两块之间轻轻的顿一下」的根因：
 * 合成器是**逐块独立**跑前向的，每块的输入窗口长 {@link RvcEngine#SEG}
 * （49×160 = 7840 样本 ≈ 0.49s）。卷积网络的感受野在窗口边缘拿不到上下文，
 * 于是每块**首尾各有一小段输出是“歪”的**；把这种块首尾直接拼起来，
 * 波形在每个 0.49s 边界上就会出现一次**跳变**（实测边界上相邻样本差
 * 是块内平均差的 10 倍以上），听感就是“说两个字轻轻顿一下”，
 * 而且顿的那一下还带着一点很小、很不正常的声音。
 *
 * <p>做法（不需要改模型输入尺寸，代价约 +11% 计算量）：
 * <ul>
 *   <li>窗口仍在 {@code SEG} 长度上跑，但**每次只前进 {@code HOP_SEG}**，
 *       相邻窗口重叠 {@link #OV_FR} 帧（50ms）；</li>
 *   <li>重叠那段在两块里各被合成过一次，用**线性交叉淡化**混合，
 *       把各自边缘的瑕疵互相盖掉，接缝处波形连续；</li>
 *   <li>每块只向输出吐 {@code segOut - ovOut} 个样本（净新增长度不变），
 *       所以输出总时长 / 时间轴与原来完全一致。</li>
 * </ul>
 *
 * <p>静音块（实时链路的输入门限）走 {@link #pushSilence}：
 * 把上一块的尾巴**淡出**再补零，避免“声音被硬切”的咔哒声。
 */
public final class RvcStitch {

    /** 相邻窗口的重叠帧数（1 帧 = 160 样本 @16k = 10ms）。5 帧 = 50ms。 */
    public static final int OV_FR = 5;
    /** 相邻窗口的输入前进量（样本 @16k）。 */
    public static final int HOP_SEG = RvcEngine.SEG - OV_FR * RvcEngine.HOP;

    private final int segOut;
    private final int ovOut;
    private final int hopOut;
    private final float[] ramp;   // 交叉淡化系数 0..1
    private final float[] tail;   // 上一块末尾 ovOut 个样本（与下一块开头是同一段音频）
    private boolean primed;

    /** @param segOut 单块合成输出样本数（{@link RvcEngine#outLen()}） */
    public RvcStitch(int segOut) {
        this.segOut = Math.max(1, segOut);
        this.ovOut = Math.max(1, Math.round((float) this.segOut * OV_FR / RvcEngine.SEG_FR));
        this.hopOut = Math.max(1, this.segOut - ovOut);
        this.ramp = new float[ovOut];
        for (int i = 0; i < ovOut; i++) ramp[i] = (i + 1f) / (ovOut + 1f);
        this.tail = new float[ovOut];
    }

    /** 每块向输出贡献的净样本数。 */
    public int hopOut() {
        return hopOut;
    }

    public int ovOut() {
        return ovOut;
    }

    public int segOut() {
        return segOut;
    }

    /** 重置（换输入 / 重新开始实时链路时调）。 */
    public void reset() {
        primed = false;
        java.util.Arrays.fill(tail, 0f);
    }

    /**
     * 并入一块合成结果。
     *
     * @param block  合成器输出，长度 {@link RvcEngine#outLen()}
     * @param dst    输出缓冲
     * @param dstOff 写入位置
     * @return 写入的样本数（= {@link #hopOut()}，块长不足时按实际限制）
     */
    public int push(float[] block, float[] dst, int dstOff) {
        int n = hopOut;
        if (block.length < segOut) n = Math.min(n, block.length);
        if (!primed) {
            System.arraycopy(block, 0, dst, dstOff, n);
            primed = true;
        } else {
            int ov = Math.min(ovOut, n);
            for (int i = 0; i < ov; i++) {
                float w = ramp[i];
                dst[dstOff + i] = tail[i] * (1f - w) + block[i] * w;
            }
            if (n > ov) System.arraycopy(block, ov, dst, dstOff + ov, n - ov);
        }
        // 存本块末尾，作为下一块开头的交叉淡化对象
        if (block.length >= segOut) System.arraycopy(block, hopOut, tail, 0, ovOut);
        return n;
    }

    /**
     * 静音块：把上一块尾巴淡出到 0 再补零，避免硬切咔哒。
     *
     * @return 写入样本数（= {@link #hopOut()}）
     */
    public int pushSilence(float[] dst, int dstOff) {
        int n = hopOut;
        int ov = Math.min(ovOut, n);
        if (primed) {
            for (int i = 0; i < ov; i++) dst[dstOff + i] = tail[i] * (1f - ramp[i]);
            primed = false;                       // 之后重新开始（静音尾巴不是可淡化的音频）
        } else {
            java.util.Arrays.fill(dst, dstOff, dstOff + ov, 0f);
        }
        if (n > ov) java.util.Arrays.fill(dst, dstOff + ov, dstOff + n, 0f);
        java.util.Arrays.fill(tail, 0f);
        return n;
    }

    /** 离线整段：给定输入样本数，算出需要跑多少块。 */
    public static int offlineBlockCount(int samples) {
        if (samples <= 0) return 0;
        return (samples + HOP_SEG - 1) / HOP_SEG;
    }
}
