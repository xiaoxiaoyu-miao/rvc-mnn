package com.rvc.mnn;

/**
 * 引擎预热。
 *
 * <p>为什么要预热：MNN 的 session 建好 ≠ 内核就绪。
 * <ul>
 *   <li>Vulkan 后端第一次跑某个卷积形状时才真正编译 pipeline（可能有几十~几百 ms 抖动）；</li>
 *   <li>ONNX Runtime（fcpe）首次推理要分配 arena、起线程池；</li>
 *   <li>CPU 后端（arm82/fp16）首次会做一次 kernel 选择与权重重排。</li>
 * </ul>
 * 这些都摊在「第一次真实推理」上。录音模式是「录完才开始推」，如果不在进页面时预热，
 * 用户点「停止并变声」后要额外等一段冷启动。
 *
 * <p>做法：用一段合成正弦（有真实基频，走的是和真人声一样的路径）把
 * hubert → fcpe → pitchEmb → 合成器 整链跑 {@code passes} 遍，丢弃输出，
 * 只量时间。第二遍的数字才是「稳定态」性能，正好用来确认预热是否奏效。
 */
public final class EngineWarmup {

    /**
     * 互斥只用 AtomicBoolean，**不持有任何 monitor 去调引擎**。
     *
     * <p>原因：预热会调 {@link RvcEngine} 的前向（内部取读锁）。
     * 如果这里持锁再取引擎读锁，而另一条路径（invalidate/releaseAll）先取引擎写锁
     * 再回来取本类的锁，就会形成锁序反转 → 死锁。
     */
    private static final java.util.concurrent.atomic.AtomicBoolean BUSY =
            new java.util.concurrent.atomic.AtomicBoolean(false);
    private static volatile String warmKey = "";

    public interface Log {
        void line(String s);
    }

    /** 预热结果。 */
    public static final class Result {
        public boolean ok;
        public int passes;
        public long totalMs;
        /** 每遍里 hubert + fcpe + 合成的耗时 (ms)，[遍][0]=hubert [1]=f0 [2]=synth */
        public long[][] stageMs;
        /** 第一遍 vs 最后一遍的总耗时，直接体现冷启动被吃掉了多少。 */
        public long firstPassMs;
        public long lastPassMs;
        public String error = "";

        public String summary() {
            if (!ok) {
                return "预热失败" + (error.isEmpty() ? "" : "：" + error);
            }
            return String.format(java.util.Locale.US,
                    "预热完成 %d 遍，共 %d ms（首遍 %d ms → 末遍 %d ms，省 %d ms）",
                    passes, totalMs, firstPassMs, lastPassMs,
                    Math.max(0, firstPassMs - lastPassMs));
        }
    }

    private EngineWarmup() {
    }

    /** 预热指纹：模型 / 后端 / 精度 / 线程数任一变化都要重来。 */
    public static String key(Config cfg) {
        return cfg.modelDir + '|' + cfg.hubert + '|' + cfg.synth + '|' + cfg.fcpe + '|' + cfg.emb
                + '|' + cfg.vulkan() + '|' + cfg.hubPrec() + '|' + cfg.ctPrec()
                + '|' + cfg.numThreads;
    }

    /** 配置/后端变了（invalidate / releaseAll）时调用，下次会重新预热。 */
    public static void invalidate() {
        warmKey = "";
    }

    public static boolean isWarm(Config cfg) {
        String k = key(cfg);
        return !warmKey.isEmpty() && warmKey.equals(k);
    }

    /** 已预热且配置没变就跳过，否则跑一次 {@link #warm}。 */
    public static Result warmIfNeeded(RvcEngine engine, Config cfg, Log log, int passes) {
        Result skip = skipIfRealtime(log);
        if (skip != null) return skip;
        String k = key(cfg);
        if (!warmKey.isEmpty() && warmKey.equals(k)) {
            Result r = new Result();
            r.ok = true;
            r.error = "已预热，跳过";
            if (log != null) log.line("引擎已预热（配置未变），跳过");
            return r;
        }
        return runGuarded(engine, cfg, log, passes, k);
    }

    public static Result warm(RvcEngine engine, Config cfg, Log log) {
        return warm(engine, cfg, log, 2);
    }

    public static Result warm(RvcEngine engine, Config cfg, Log log, int passes) {
        Result skip = skipIfRealtime(log);
        if (skip != null) return skip;
        return runGuarded(engine, cfg, log, passes, key(cfg));
    }

    /**
     * 无视「实时变声在跑就跳过」的预热——给 {@link RealtimeService} 自己用。
     *
     * <p>实时链路刚起来的头几块是**最冷**的时候：Vulkan 第一次遇到某个卷积形状才编译
     * pipeline、ORT 还没分配 arena、CPU 频率也还没拉起来。这些都摊在前几块上，
     * 表现就是"刚开实时变声的前几秒一顿一顿的，过几秒才顺"。
     * 线程真正开始跑之前先在这儿把这个代价付掉。
     */
    public static Result warmForce(RvcEngine engine, Config cfg, Log log, int passes) {
        return runGuarded(engine, cfg, log, passes, key(cfg));
    }

    /**
     * 实时变声在跑时直接跳过预热。
     *
     * <p>两个理由：
     * <ol>
     *   <li>实时管线本身已经把整链跑热了，再预热纯属浪费；</li>
     *   <li>虽然 {@link RvcEngine} 已按 session 加了互斥、并发不会再算错，
     *       但预热会和实时线程抢 CPU，容易把实时块的节奏拖到丢掉当前块。</li>
     * </ol>
     * 注意这里返回 ok=true（是「不需要」而不是「失败」），且不写 warmKey，
     * 等实时停下来后下次还会正常预热。
     */
    private static Result skipIfRealtime(Log log) {
        if (!RealtimeService.running) {
            return null;
        }
        Result r = new Result();
        r.ok = true;
        r.error = "实时变声运行中";
        if (log != null) {
            log.line("实时变声运行中，引擎本来就在跑，跳过预热");
        }
        return r;
    }

    /** 抢占预热权后执行；不持锁调引擎，避免锁序反转。 */
    private static Result runGuarded(RvcEngine engine, Config cfg, Log log, int passes, String k) {
        if (!BUSY.compareAndSet(false, true)) {
            Result r = new Result();
            r.ok = false;
            r.error = "已在预热中，跳过本次";
            if (log != null) log.line("预热进行中，跳过重复请求");
            return r;
        }
        try {
            Result r = warmLocked(engine, cfg, log, passes);
            if (r.ok) {
                warmKey = k;
            }
            return r;
        } finally {
            BUSY.set(false);
        }
    }

    private static Result warmLocked(RvcEngine engine, Config cfg, Log log, int passes) {
        Result r = new Result();
        r.passes = Math.max(1, passes);
        r.stageMs = new long[r.passes][3];
        if (engine == null) {
            r.error = "engine=null";
            return r;
        }
        long tAll = System.currentTimeMillis();
        try {
            // 220Hz 正弦 + 轻噪声：有明确基频（fcpe 走正常路径），幅度小（不进静音分支）
            float[] seg = new float[RvcEngine.SEG];
            for (int i = 0; i < seg.length; i++) {
                seg[i] = (float) (Math.sin(2.0 * Math.PI * 220.0 * i / RvcDsp.SR_IN) * 0.3);
            }
            float[] featBuf = new float[64 * RvcEngine.FEAT];
            float[] phone = new float[RvcEngine.SEG_FR * RvcEngine.FEAT];
            float[] f0 = new float[RvcEngine.SEG_FR];
            float[] pe = new float[RvcEngine.SEG_FR * RvcEngine.PET];
            float ratio = cfg.pitchRatio();

            for (int p = 0; p < r.passes; p++) {
                long t0 = System.currentTimeMillis();
                int T = engine.hubFeatures(seg, featBuf);
                long t1 = System.currentTimeMillis();
                if (T <= 0) {
                    r.error = "hubert 返回 T=" + T;
                    return r;
                }
                for (int t = 0; t < RvcEngine.SEG_FR; t++) {
                    int fr = Math.min(T - 1, (int) ((long) t * T / RvcEngine.SEG_FR));
                    System.arraycopy(featBuf, fr * RvcEngine.FEAT, phone,
                            t * RvcEngine.FEAT, RvcEngine.FEAT);
                }
                if (!engine.f0Fill(seg, f0, ratio)) {
                    r.error = "fcpe 失败";
                    return r;
                }
                long t2 = System.currentTimeMillis();
                engine.pitchEmb(f0, pe);
                float[] out = new float[Math.max(1, engine.outLen())];
                if (!engine.synthesize(phone, f0, pe, out)) {
                    r.error = "合成器失败";
                    return r;
                }
                long t3 = System.currentTimeMillis();
                r.stageMs[p][0] = t1 - t0;
                r.stageMs[p][1] = t2 - t1;
                r.stageMs[p][2] = t3 - t2;
                if (p == 0) {
                    r.firstPassMs = t3 - t0;
                    if (log != null) {
                        log.line(String.format(java.util.Locale.US,
                                "预热第 1 遍（冷）：hubert %d ms · fcpe %d ms · synth %d ms",
                                t1 - t0, t2 - t1, t3 - t2));
                    }
                } else if (p == r.passes - 1) {
                    r.lastPassMs = t3 - t0;
                    if (log != null) {
                        log.line(String.format(java.util.Locale.US,
                                "预热第 %d 遍（稳）：hubert %d ms · fcpe %d ms · synth %d ms",
                                p + 1, t1 - t0, t2 - t1, t3 - t2));
                    }
                }
            }
            if (r.passes == 1) {
                r.lastPassMs = r.firstPassMs;
            }
            r.totalMs = System.currentTimeMillis() - tAll;
            r.ok = true;
            if (log != null) log.line(r.summary());
        } catch (Throwable t) {
            r.ok = false;
            r.error = String.valueOf(t);
            if (log != null) log.line("预热异常：" + t);
        }
        return r;
    }
}
