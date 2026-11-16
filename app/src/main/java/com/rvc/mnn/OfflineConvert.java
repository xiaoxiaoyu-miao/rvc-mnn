package com.rvc.mnn;

import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/**
 * 离线（非实时）整段变声。
 *
 * <p>与实时链路的区别：
 * <ul>
 *   <li>不需要 AudioTrack —— 产出直接落成 WAV 文件，手机一点声音都不出；</li>
 *   <li>不是逐块跟着实时节奏走，而是把整段音频切块后**尽可能快地**吃完，
 *       所以能做到比 1.0x 实时更快；</li>
 *   <li>仍然沿用验证过的两级流水线
 *       （producer: hubert+fcpe+pitchEmb ／ consumer: 合成器），
 *       只是把播放换成了写盘。</li>
 * </ul>
 *
 * <p>切块方式在这版改了：窗口长保持 {@link RvcEngine#SEG}，但**每次前进
 * {@link RvcStitch#HOP_SEG}**（相邻窗口重叠 {@link RvcStitch#OV_FR} 帧），
 * 重叠段用交叉淡化缝合。原因见 {@link RvcStitch}：逐块独立合成时块边缘拿不到
 * 上下文，直接拼接会在每个 0.49s 边界上留一个波形跳变（听感＝“轻轻顿一下”）。
 * 代价是块数 +11%，本机仍然比实时快得多。
 */
public final class OfflineConvert {

    public interface Log {
        void line(String s);
    }

    private OfflineConvert() {
    }

    /** 一块的中间产物（hubert 特征 + 变调后 f0 + 音高编码）。 */
    private static class Blk {
        float[] phone, f0, pe;
    }

    /**
     * @param inWav  输入 WAV（任意采样率/声道，内部重采样到 16 kHz 单声道）
     * @param outWav 输出 WAV（{@link RvcDsp#SR_OUT} 采样率，乘 cfg.gain）
     * @return 输出音频秒数
     */
    public static double convert(RvcEngine engine, Config cfg, String inWav, String outWav,
                                 Log log) throws Exception {
        final long t0 = System.currentTimeMillis();
        final float ratio = cfg.pitchRatio();
        final float gain = cfg.gain;
        if (log != null) {
            log.line(String.format("音色=%s 变调 %+.1f 半音 (倍率 %.3f) 增益 %.2fx",
                    cfg.voiceTag(), cfg.pitchSemi, ratio, gain));
        }

        int[] srHolder = new int[1];
        float[] a = WavIo.readWav(inWav, srHolder);
        a = WavIo.resample(a, srHolder[0], RvcDsp.SR_IN);
        if (a.length == 0) {
            throw new Exception("输入音频是空的");
        }

        final int aLen = engine.outLen();
        if (aLen <= 0) throw new Exception("合成器输出为空");
        final RvcStitch stitch = new RvcStitch(aLen);
        final int hop = RvcStitch.HOP_SEG;
        final int hopOut = stitch.hopOut();
        final int nblk = RvcStitch.offlineBlockCount(a.length);
        if (log != null) {
            log.line(String.format("音频 %.2fs, %d 块（重叠 %d 帧 / %.0fms 交叉淡化）",
                    a.length / (double) RvcDsp.SR_IN, nblk, RvcStitch.OV_FR,
                    stitch.ovOut() * 1000.0 / RvcDsp.SR_OUT));
        }

        final float[] src = a;
        final BlockingQueue<Blk> q = new ArrayBlockingQueue<>(4);
        final Blk POISON = new Blk();
        final AtomicBoolean err = new AtomicBoolean(false);
        final AtomicLong doneCount = new AtomicLong();
        final float[] out = new float[(int) ((long) nblk * hopOut) + stitch.ovOut()];
        final AtomicLong written = new AtomicLong();
        final AtomicBoolean loggedT = new AtomicBoolean(false);

        Thread producer = new Thread(() -> {
            float[] featBuf = new float[64 * RvcEngine.FEAT];
            try {
                for (int s = 0; s < nblk; s++) {
                    int off = s * hop;
                    float[] seg = new float[RvcEngine.SEG];
                    int take = Math.min(RvcEngine.SEG, src.length - off);
                    if (take > 0) System.arraycopy(src, off, seg, 0, take);
                    int T = engine.hubFeatures(seg, featBuf);
                    if (T <= 0) {
                        err.set(true);
                        break;
                    }
                    if (log != null && loggedT.compareAndSet(false, true)) {
                        log.line("  hubert 帧数 T=" + T + " / 合成器 " + RvcEngine.SEG_FR
                                + " 帧" + (T < RvcEngine.SEG_FR ? "（会做帧补齐）" : ""));
                    }
                    Blk blk = new Blk();
                    blk.phone = new float[RvcEngine.SEG_FR * RvcEngine.FEAT];
                    for (int t = 0; t < RvcEngine.SEG_FR; t++) {
                        int fr = Math.min(T - 1, (int) ((long) t * T / RvcEngine.SEG_FR));
                        System.arraycopy(featBuf, fr * RvcEngine.FEAT, blk.phone,
                                t * RvcEngine.FEAT, RvcEngine.FEAT);
                    }
                    float[] f0 = new float[RvcEngine.SEG_FR];
                    if (!engine.f0Fill(seg, f0, ratio)) {
                        err.set(true);
                        break;
                    }
                    blk.f0 = f0;
                    blk.pe = new float[RvcEngine.SEG_FR * RvcEngine.PET];
                    engine.pitchEmb(f0, blk.pe);
                    q.put(blk);
                }
            } catch (Exception e) {
                if (log != null) log.line("producer err " + e);
                err.set(true);
            } finally {
                try {
                    q.put(POISON);
                } catch (Exception ignored) {
                }
            }
        }, "offline-producer");

        Thread consumer = new Thread(() -> {
            try {
                float[] audioBuf = new float[aLen];
                while (true) {
                    Blk blk = q.poll(30, TimeUnit.SECONDS);
                    if (blk == null) {
                        if (log != null) log.line("consumer timeout");
                        break;
                    }
                    if (blk == POISON) break;
                    if (!engine.synthesize(blk.phone, blk.f0, blk.pe, audioBuf)) {
                        err.set(true);
                        break;
                    }
                    long d = doneCount.get();
                    int n = stitch.push(audioBuf, out, (int) (d * hopOut));
                    written.set(d * hopOut + n);
                    d = doneCount.incrementAndGet();
                    if (log != null && d % 10 == 0) {
                        log.line("  " + d + "/" + nblk + " blocks, "
                                + (System.currentTimeMillis() - t0) / 1000.0 + "s");
                    }
                }
            } catch (Exception e) {
                if (log != null) log.line("consumer err " + e);
                err.set(true);
            }
        }, "offline-consumer");

        producer.start();
        consumer.start();
        producer.join();
        consumer.join();

        if (doneCount.get() == 0) {
            throw new Exception("没有产出（err=" + err.get() + "）");
        }
        int total = (int) Math.min(written.get(), out.length);
        float[] y = new float[total];
        System.arraycopy(out, 0, y, 0, total);

        int want = (int) ((long) a.length * RvcDsp.SR_OUT / RvcDsp.SR_IN);
        if (y.length > want) {
            float[] t = new float[want];
            System.arraycopy(y, 0, t, 0, want);
            y = t;
        }
        WavIo.writeWav(outWav, y, RvcDsp.SR_OUT, gain);

        double sec = y.length / (double) RvcDsp.SR_OUT;
        double el = (System.currentTimeMillis() - t0) / 1000.0;
        if (log != null) {
            log.line(String.format("完成: %.2fs 音频, 用时 %.2fs -> %.2fx realtime -> %s",
                    sec, el, sec / Math.max(el, 1e-6), outWav));
        }
        return sec;
    }
}
