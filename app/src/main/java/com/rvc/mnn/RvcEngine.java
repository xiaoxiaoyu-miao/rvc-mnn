package com.rvc.mnn;

import android.content.Context;

import java.io.File;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

/**
 * 模型单例：hubert + 合成器（MNN native）+ fcpe（ORT）+ 音高编码表。
 *
 * <p>Activity 的离线转换与 Service 的实时循环共用同一个实例 —— native 侧的 session 是
 * 全局的，重复创建只会互相踩；统一在这里按「配置指纹」懒加载/重建。
 */
public class RvcEngine {
    static {
        System.loadLibrary("c++_shared");
        System.loadLibrary("MNN_Vulkan");
        System.loadLibrary("MNN");
        System.loadLibrary("rvcmnn");
    }

    // native（rvc_jni.cpp）
    public static native boolean init(String hubPath, String ctPath, boolean vulkan, int hubPrec, int ctPrec, int threads);
    public static native int hubert(float[] seg, float[] featOut);
    public static native boolean synth(float[] phone, float[] nsff0, float[] pe, float[] audioOut);
    public static native int audioLen();
    public static native String backendInfo();
    public static native void release();

    public static final int SEG_FR = 49, HOP = 160, SEG = SEG_FR * HOP, PET = 192, FEAT = 768;

    private static volatile RvcEngine INSTANCE;

    public static RvcEngine get(Context c) {
        RvcEngine e = INSTANCE;
        if (e == null) {
            synchronized (RvcEngine.class) {
                if (INSTANCE == null) INSTANCE = new RvcEngine(c.getApplicationContext());
                e = INSTANCE;
            }
        }
        return e;
    }

    private final Context ctx;
    /** 保护 native/ORT 调用：实时线程持有读锁，release/ensure 持有写锁，避免释放时正在推理。 */
    private final java.util.concurrent.locks.ReentrantReadWriteLock rw =
            new java.util.concurrent.locks.ReentrantReadWriteLock();

    /**
     * 每个 native session 自己的互斥锁。
     *
     * <p>为什么还需要它们：{@link #rw} 的读锁是**共享**的，只保证「release/ensure vs 推理」
     * 不打架，并不阻止两个线程同时调**同一个** MNN session。MNN 的 interpreter 复用内部
     * scratch buffer，并发 runSession 会互相踩内存——实测表现就是
     * {@code hubertFeatures} 返回 T=0、合成器出噪声。
     *
     * <p>粒度按 session 分（hub / synth / fcpe 三把互不相干），所以实时管线的
     * producer(hub+fcpe) 与 consumer(synth) 双线程并行完全不受影响，
     * 只是「同一 session 的并发调用」被串起来。
     */
    private final Object hubLock = new Object();
    private final Object synthLock = new Object();
    private final Object f0Lock = new Object();
    private OrtEnvironment oenv;
    private OrtSession fcpe;
    private float[] embTable;
    private volatile String nativeKey = "";
    private volatile String javaKey = "";

    private RvcEngine(Context c) { this.ctx = c; }

    private static String path(Config cfg, String name) { return cfg.modelDir + "/" + name; }

    /** 确保模型就绪；配置变了会自动重建。 */
    public boolean ensure(Config cfg) {
        rw.writeLock().lock();
        try {
            return ensureLocked(cfg);
        } finally {
            rw.writeLock().unlock();
        }
    }

    private boolean ensureLocked(Config cfg) {
        String nk = path(cfg, cfg.hubert) + "|" + path(cfg, cfg.synth) + "|"
                + cfg.vulkan() + "|" + cfg.hubPrec() + "|" + cfg.ctPrec() + "|" + cfg.numThreads;
        if (!nk.equals(nativeKey)) {
            long t0 = System.currentTimeMillis();
            RvcLog.i("RvcEngine", (cfg.vulkan() != 0 ? "[Vulkan] " : "[CPU] ")
                    + "加载 hubert=" + cfg.hubert + " synth=" + cfg.synth);
            boolean ok = init(path(cfg, cfg.hubert), path(cfg, cfg.synth),
                    cfg.vulkan() != 0, cfg.hubPrec(), cfg.ctPrec(), cfg.numThreads);
            RvcLog.i("RvcEngine", "  init=" + ok + " (" + (System.currentTimeMillis() - t0) + "ms) " + backendInfo());
            if (!ok) {
                nativeKey = "";
                RvcLog.i("RvcEngine", "模型加载失败：检查 ② 里的四个权重文件是否都存在");
                return false;
            }
            nativeKey = nk;
            RvcLog.i("RvcEngine", "  合成器输出 " + audioLen() + " samples");
            boolean rateChanged = RvcDsp.applySrOut(audioLen(), SEG);
            RvcLog.i("RvcEngine", "  输出采样率 " + RvcDsp.SR_OUT + " Hz"
                    + (rateChanged ? "（按模型自动更新）" : ""));
        }

        String jk = path(cfg, cfg.fcpe) + "|" + path(cfg, cfg.emb);
        if (!jk.equals(javaKey) || fcpe == null || embTable == null) {
            long t0 = System.currentTimeMillis();
            RvcLog.i("RvcEngine", "加载 fcpe=" + cfg.fcpe + " emb=" + cfg.emb);
            try {
                if (fcpe != null) { try { fcpe.close(); } catch (Exception ignored) {} fcpe = null; }
                oenv = OrtEnvironment.getEnvironment();
                OrtSession.SessionOptions so = new OrtSession.SessionOptions();
                so.setIntraOpNumThreads(4);
                fcpe = oenv.createSession(path(cfg, cfg.fcpe), so);
                embTable = new float[256 * PET];
                try (RandomAccessFile f = new RandomAccessFile(path(cfg, cfg.emb), "r")) {
                    ByteBuffer bb = ByteBuffer.allocate(256 * PET * 4).order(ByteOrder.LITTLE_ENDIAN);
                    f.readFully(bb.array());
                    bb.asFloatBuffer().get(embTable);
                }
                javaKey = jk;
                RvcLog.i("RvcEngine", "  就绪 (" + (System.currentTimeMillis() - t0) + "ms)");
            } catch (Exception e) {
                RvcLog.i("RvcEngine", "fcpe/emb 加载失败: " + e);
                fcpe = null;
                embTable = null;
                return false;
            }
        }
        return true;
    }

    /** 配置变化 → 丢掉 native session，下次 ensure 重建。 */
    public void invalidate() {
        rw.writeLock().lock();
        try {
            nativeKey = "";
            javaKey = "";
            try { release(); } catch (Throwable ignored) {}
        } finally {
            rw.writeLock().unlock();
        }
    }

    public void releaseAll() {
        rw.writeLock().lock();
        try {
            try { if (fcpe != null) fcpe.close(); } catch (Exception ignored) {}
            fcpe = null;
            embTable = null;
            nativeKey = "";
            javaKey = "";
            try { release(); } catch (Throwable ignored) {}
        } finally {
            rw.writeLock().unlock();
        }
    }

    public boolean isReady() {
        return fcpe != null && embTable != null && audioLen() > 0;
    }

    /** fcpe 提取 f0 并乘变调倍率（ratio=1 就是不处理）。 */
    public boolean f0Fill(float[] seg, float[] f0, float ratio) {
        rw.readLock().lock();
        try {
            synchronized (f0Lock) {
                return f0FillLocked(seg, f0, ratio);
            }
        } finally {
            rw.readLock().unlock();
        }
    }

    private boolean f0FillLocked(float[] seg, float[] f0, float ratio) {
        if (fcpe == null) return false;
        java.util.Arrays.fill(f0, 0f);
        try {
            OnnxTensor in = OnnxTensor.createTensor(oenv, java.nio.FloatBuffer.wrap(seg), new long[]{1, SEG, 1});
            OrtSession.Result r = fcpe.run(java.util.Collections.singletonMap("audio", in));
            float[][][] v = (float[][][]) r.get(0).getValue();
            int nf = Math.min(v[0].length, SEG_FR);
            for (int i = 0; i < nf; i++) f0[i] = Float.isFinite(v[0][i][0]) ? v[0][i][0] : 0f;
            in.close();
            r.close();
        } catch (Exception e) {
            RvcLog.i("RvcEngine", "fcpe err: " + e);
            return false;
        }
        RvcDsp.shiftF0(f0, ratio);
        return true;
    }

    /** f0 → 音高编码（emb 查表），写入 pe（SEG_FR*PET）。 */
    public void pitchEmb(float[] f0, float[] pe) {
        rw.readLock().lock();
        try {
            for (int t = 0; t < SEG_FR; t++) {
                int bin = RvcDsp.melCoarse(f0[t]) - 1;
                if (bin < 0) bin = 0;
                if (bin > 255) bin = 255;
                System.arraycopy(embTable, bin * PET, pe, t * PET, PET);
            }
        } finally {
            rw.readLock().unlock();
        }
    }

    /** hubert 前向：读锁 + hub session 互斥。 */
    public int hubFeatures(float[] seg, float[] featOut) {
        rw.readLock().lock();
        try {
            synchronized (hubLock) {
                return hubert(seg, featOut);
            }
        } finally {
            rw.readLock().unlock();
        }
    }

    /** 合成器前向：读锁 + synth session 互斥。 */
    public boolean synthesize(float[] phone, float[] f0, float[] pe, float[] out) {
        rw.readLock().lock();
        try {
            synchronized (synthLock) {
                return synth(phone, f0, pe, out);
            }
        } finally {
            rw.readLock().unlock();
        }
    }

    /** 合成器输出长度（读锁保护）。 */
    public int outLen() {
        rw.readLock().lock();
        try {
            return audioLen();
        } finally {
            rw.readLock().unlock();
        }
    }

    public String modelDir() { return ctx.getFilesDir().getAbsolutePath(); }

    public static boolean fileOk(Config cfg, String name) {
        return new File(path(cfg, name)).isFile();
    }
}
