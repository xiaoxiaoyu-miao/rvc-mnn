package com.rvc.mnn;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Intent;
import android.content.pm.ServiceInfo;
import android.media.AudioAttributes;
import android.media.AudioFormat;
import android.media.AudioManager;
import android.media.AudioRecord;
import android.media.AudioTrack;
import android.media.MediaRecorder;
import android.media.audiofx.AcousticEchoCanceler;
import android.media.audiofx.AutomaticGainControl;
import android.media.audiofx.NoiseSuppressor;
import android.os.Build;
import android.os.IBinder;
import android.os.PowerManager;

import java.io.FileOutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.atomic.AtomicLong;

/**
 * 实时变声前台服务：把「采集 → hubert+fcpe → 合成 → 播放」整条流水线放到 Service 里跑，
 * 这样退到后台 / 息屏也能持续工作。
 *
 * <p>保活三件套：
 * <ul>
 *   <li>startForeground + microphone|mediaPlayback 类型（Android 12+ 后台受限的核心）</li>
 *   <li>PARTIAL_WAKE_LOCK 防止 CPU 休眠断流</li>
 *   <li>电池优化白名单（见 App 内「后台保活」按钮 / tools/keepalive.sh 的 root 白名单）</li>
 * </ul>
 */
public class RealtimeService extends Service {
    public static final String ACTION_START = "com.rvc.mnn.action.START";
    public static final String ACTION_STOP = "com.rvc.mnn.action.STOP";
    /** 仅开关录屏（服务已在跑时用，不动实时链路）。 */
    public static final String ACTION_START_REC = "com.rvc.mnn.action.START_REC";
    public static final String ACTION_STOP_REC = "com.rvc.mnn.action.STOP_REC";

    /** MediaProjection 授权结果（Activity 弹窗拿到后随 Intent 传进来）。 */
    public static final String EXTRA_PROJ_CODE = "projCode";
    public static final String EXTRA_PROJ_DATA = "projData";
    public static final String EXTRA_REC_PATH = "recPath";

    private static final String TAG = "RvcRtService";
    private static final String CHANNEL = "rvc_realtime";
    private static final int NID = 0x5256;

    /** UI 用来同步按钮状态。 */
    public static volatile boolean running = false;

    private PowerManager.WakeLock wake;
    private Thread worker;
    private volatile boolean stopRequested;

    // ---- 录屏（MediaProjection）----
    private volatile int projCode = Integer.MIN_VALUE;
    private volatile Intent projData;
    private volatile String recPath;
    private volatile boolean fgWithProjection;

    private static final int SR_IN = RvcDsp.SR_IN;
    // SR_OUT 现在由合成器输出长度推导（40k/48k 模型不同），必须每次读 RvcDsp.SR_OUT，
    // 不能在类初始化时快照成常量。
    private static final int SEG_FR = RvcEngine.SEG_FR;
    private static final int SEG = RvcEngine.SEG;
    private static final int PET = RvcEngine.PET;
    private static final int FEAT = RvcEngine.FEAT;

    private AcousticEchoCanceler aec;
    private NoiseSuppressor nsFx;
    private AutomaticGainControl agcFx;

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        String act = intent == null ? null : intent.getAction();

        if (ACTION_STOP.equals(act)) {
            RvcLog.i(TAG, "收到停止指令");
            ScreenRecorder.stop();
            stopAll();
            return START_NOT_STICKY;
        }
        if (ACTION_STOP_REC.equals(act)) {
            RvcLog.i(TAG, "收到停止录屏指令");
            ScreenRecorder.stop();
            return running ? START_STICKY : START_NOT_STICKY;
        }

        // START / START_REC 都可能附带 MediaProjection 授权结果
        if (intent != null) {
            int code = intent.getIntExtra(EXTRA_PROJ_CODE, Integer.MIN_VALUE);
            if (code != Integer.MIN_VALUE) {
                projCode = code;
                projData = intent.getParcelableExtra(EXTRA_PROJ_DATA);
                recPath = intent.getStringExtra(EXTRA_REC_PATH);
            }
        }

        if (running) {
            // 服务已在跑：这里只处理「补开录屏」
            if (projCode != Integer.MIN_VALUE) tryStartRecording();
            return START_STICKY;
        }

        startAsForeground(projData != null);
        acquireWakeLock();
        stopRequested = false;
        running = true;
        worker = new Thread(this::loop, "rvc-rt-svc");
        worker.start();
        return START_STICKY;
    }

    /** 有 MediaProjection 授权就开录屏；音频由 loop() 每块喂给 ScreenRecorder。 */
    private void tryStartRecording() {
        int code = projCode;
        Intent data = projData;
        String path = recPath != null ? recPath : defaultRecPath();
        if (code == Integer.MIN_VALUE || data == null) return;
        if (ScreenRecorder.isRecording()) {
            RvcLog.i(TAG, "已在录屏中");
            return;
        }
        // 先用掉这次授权，避免并发重复启动
        projCode = Integer.MIN_VALUE;
        projData = null;

        // 建 MediaCodec / VirtualDisplay 有点耗时，别阻塞主线程
        new Thread(() -> {
            // Android 14+：必须先以 mediaProjection 类型处于前台，才能 getMediaProjection
            startAsForeground(true);
            boolean ok = ScreenRecorder.start(this, code, data, path);
            if (ok) {
                RvcLog.i(TAG, "已开始 App 内录屏 -> " + path);
            } else {
                RvcLog.i(TAG, "录屏启动失败（授权过期或系统拒绝）");
            }
        }, "rvc-rec-start").start();
    }

    private static String defaultRecPath() {
        return "/sdcard/Download/rvc_rec_" + System.currentTimeMillis() + ".mp4";
    }

    @Override
    public void onDestroy() {
        stopAll();
        super.onDestroy();
    }

    // ------------------------------------------------------------------ 通知 / 保活
    /** @param withProjection 是否需要 mediaProjection 前台类型（App 内录屏时） */
    private void startAsForeground(boolean withProjection) {
        fgWithProjection = withProjection;
        NotificationManager nm = getSystemService(NotificationManager.class);
        if (Build.VERSION.SDK_INT >= 26) {
            NotificationChannel ch = new NotificationChannel(CHANNEL, "实时变声", NotificationManager.IMPORTANCE_LOW);
            ch.setShowBadge(false);
            nm.createNotificationChannel(ch);
        }
        Intent stopIntent = new Intent(this, RealtimeService.class).setAction(ACTION_STOP);
        PendingIntent stopPi = PendingIntent.getService(this, 7, stopIntent,
                PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE);
        Intent openIntent = new Intent(this, MainActivity.class)
                .addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP | Intent.FLAG_ACTIVITY_NEW_TASK);
        PendingIntent openPi = PendingIntent.getActivity(this, 8, openIntent,
                PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE);

        Notification.Builder b = Build.VERSION.SDK_INT >= 26
                ? new Notification.Builder(this, CHANNEL)
                : new Notification.Builder(this);
        b.setSmallIcon(android.R.drawable.ic_btn_speak_now)
                .setContentTitle("RVC 实时变声运行中")
                .setContentText("后台持续工作，点此返回 App")
                .setOngoing(true)
                .setContentIntent(openPi)
                .addAction(new Notification.Action.Builder(
                        android.R.drawable.ic_media_pause, "停止", stopPi).build());

        if (Build.VERSION.SDK_INT >= 29) {
            int type = ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE
                    | ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PLAYBACK;
            if (withProjection) type |= ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION;
            try {
                startForeground(NID, b.build(), type);
            } catch (Exception e) {
                // 某些 ROM 不允许运行中把类型改成 mediaProjection：退一步只保留原类型
                RvcLog.i(TAG, "带 mediaProjection 类型的前台启动失败，回退: " + e);
                fgWithProjection = false;
                startForeground(NID, b.build(),
                        ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE
                                | ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PLAYBACK);
                if (withProjection) {
                    RvcLog.i(TAG, "该 ROM 不支持运行中补开 mediaProjection，请在「不外放」模式下先开始录屏再变声");
                }
            }
        } else {
            startForeground(NID, b.build());
        }
    }

    private void acquireWakeLock() {
        try {
            PowerManager pm = getSystemService(PowerManager.class);
            wake = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "rvc:realtime");
            wake.setReferenceCounted(false);
            wake.acquire();
        } catch (Exception e) {
            RvcLog.i(TAG, "唤醒锁申请失败: " + e);
        }
    }

    private void releaseWakeLock() {
        try {
            if (wake != null && wake.isHeld()) wake.release();
        } catch (Exception ignored) {}
        wake = null;
    }

    // ------------------------------------------------------------------ 停止
    private void stopAll() {
        if (!running && worker == null) return;
        stopRequested = true;
        Thread w = worker;
        worker = null;
        if (w != null && w != Thread.currentThread()) {
            try { w.join(3000); } catch (InterruptedException ignored) {}
        }
        running = false;
        releaseWakeLock();
        try { stopForeground(true); } catch (Exception ignored) {}
        try { stopSelf(); } catch (Exception ignored) {}
    }

    // ------------------------------------------------------------------ 主循环
    private void loop() {
        AudioRecord ar = null;
        AudioTrack at = null;
        AudioManager am = null;
        try {
            Config cfg = new Config();
            cfg.load(this);
            // 后台也可能被直接拉起：这里同样要保证模型可用（首次会解压 assets）
            if (!ModelBootstrap.prepare(this, cfg)) {
                RvcLog.i(TAG, "模型未就绪（内置解压失败？）");
                srcFailStop();
                return;
            }
            final float ratio = cfg.pitchRatio();
            final float gain = cfg.gain;
            final float gate = cfg.gate;
            final float fbDb = cfg.feedbackGateDb;

            RvcEngine engine = RvcEngine.get(this);
            if (!engine.ensure(cfg)) {
                RvcLog.i(TAG, "模型未就绪，无法启动实时变声");
                srcFailStop();
                return;
            }
            final int aLen = engine.outLen();
            if (aLen <= 0) { RvcLog.i(TAG, "合成器输出为空"); srcFailStop(); return; }
            // 块间接缝：窗口仍是 SEG，但每次只前进 HOP_SEG，重叠段交叉淡化。
            // 不做这个的话，每个 0.49s 边界都会有一次波形跳变（听感=“轻轻顿一下”）。
            final RvcStitch stitch = new RvcStitch(aLen);
            final int hopOut = stitch.hopOut();

            // 先把这个引擎跑热再开采集：开头几秒的"一顿一顿"就是冷启动摊在前几块上
            // （Vulkan pipeline 首次编译 / ORT arena / CPU 提频）。这里先把代价付掉。
            try {
                EngineWarmup.Result wr = EngineWarmup.warmForce(engine, cfg, s -> RvcLog.i(TAG, s), 2);
                RvcLog.i(TAG, "实时链路预热: " + wr.summary());
            } catch (Throwable t) {
                RvcLog.i(TAG, "实时链路预热失败（忽略继续）: " + t);
            }

            // ---- 输出路由：外放 / 听筒 / 不外放；以及是否走 media 用途（可被系统录屏捕获）----
            final int outMode = cfg.outputMode;
            final boolean muted = cfg.isMuted();          // 不外放：整个 AudioTrack 都不建
            final boolean mediaUsage = cfg.useMediaUsage();
            // 走 media 用途后，系统 AEC 不再把这份输出当参考信号，实时通话场景必须保留话音路径
            final boolean useAec = !mediaUsage;

            am = (AudioManager) getSystemService(AUDIO_SERVICE);
            try {
                if (am != null) {
                    if (mediaUsage) {
                        am.setMode(AudioManager.MODE_NORMAL);
                        am.setSpeakerphoneOn(false);
                    } else {
                        am.setMode(AudioManager.MODE_IN_COMMUNICATION);
                        am.setSpeakerphoneOn(outMode == Config.OUT_SPEAKER);
                    }
                }
            } catch (Exception e) { RvcLog.i(TAG, "音频路由设置失败: " + e); }

            // 不外放就不需要 AEC 的「采集端抵消」，用普通识别源即可；否则沿用话音源让 AEC 生效
            final int recSource = useAec
                    ? MediaRecorder.AudioSource.VOICE_COMMUNICATION
                    : MediaRecorder.AudioSource.VOICE_RECOGNITION;
            int minIn = AudioRecord.getMinBufferSize(SR_IN, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT);
            int inBuf = Math.max(minIn, SEG * 2 * 4);
            ar = new AudioRecord(recSource, SR_IN,
                    AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, inBuf);
            if (ar.getState() != AudioRecord.STATE_INITIALIZED) { RvcLog.i(TAG, "AudioRecord 初始化失败"); srcFailStop(); return; }

            if (!muted) {
                int minOut = AudioTrack.getMinBufferSize(RvcDsp.SR_OUT, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_16BIT);
                int outBuf = Math.max(minOut, hopOut * 2 * 4);
                // media 用途必须显式允许被录屏（AudioPlaybackCapture）捕获；
                // 该开关是 AudioAttributes 上的，AudioTrack.Builder 没有这个方法
                AudioAttributes attrs = new AudioAttributes.Builder()
                        .setUsage(mediaUsage ? AudioAttributes.USAGE_MEDIA
                                             : AudioAttributes.USAGE_VOICE_COMMUNICATION)
                        .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                        .setAllowedCapturePolicy(mediaUsage
                                ? AudioAttributes.ALLOW_CAPTURE_BY_ALL
                                : AudioAttributes.ALLOW_CAPTURE_BY_NONE)
                        .build();
                AudioFormat fmt = new AudioFormat.Builder()
                        .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                        .setSampleRate(RvcDsp.SR_OUT)
                        .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                        .build();
                AudioTrack.Builder tb = new AudioTrack.Builder()
                        .setAudioAttributes(attrs)
                        .setAudioFormat(fmt)
                        .setBufferSizeInBytes(outBuf)
                        .setTransferMode(AudioTrack.MODE_STREAM)
                        .setSessionId(AudioManager.AUDIO_SESSION_ID_GENERATE);
                at = tb.build();
                if (at.getState() != AudioTrack.STATE_INITIALIZED) {
                    RvcLog.i(TAG, "AudioTrack 初始化失败，本次改为静音输出");
                    at = null;
                }
            }

            if (useAec) {
                int sid = ar.getAudioSessionId();
                try {
                    if (AcousticEchoCanceler.isAvailable()) {
                        aec = AcousticEchoCanceler.create(sid);
                        if (aec != null) aec.setEnabled(true);
                    }
                    if (NoiseSuppressor.isAvailable()) {
                        nsFx = NoiseSuppressor.create(sid);
                        if (nsFx != null) nsFx.setEnabled(true);
                    }
                    if (AutomaticGainControl.isAvailable()) {
                        agcFx = AutomaticGainControl.create(sid);
                        if (agcFx != null) agcFx.setEnabled(false);
                    }
                } catch (Exception e) { RvcLog.i(TAG, "音频效果初始化: " + e); }
            }

            ar.startRecording();
            // 注意：AudioTrack.play() 挪到「拿到第一块合成结果之后」（见下面播放循环）。
            // 一上来就按实时节奏灌静音，会让开头几秒听起来一顿一顿。

            // 有 MediaProjection 授权就同时开录屏（音频由下面播放循环逐块喂过去）
            tryStartRecording();

            final WavSink sink = new WavSink();
            sink.enabled = cfg.saveRealtime;
            final String rtPath = "/sdcard/Download/rt_" + System.currentTimeMillis() + ".wav";

            RvcLog.i(TAG, "AEC=" + (aec != null && aec.getEnabled()) + " NS=" + (nsFx != null && nsFx.getEnabled())
                    + "  路由=" + cfg.outputTag()
                    + (mediaUsage ? " [media 用途，系统录屏可捕获]" : "")
                    + (muted ? "  → 不外放，声音只进「App 内录屏」" : "  外放会回环，建议戴耳机"));
            RvcLog.i(TAG, String.format("实时变声启动 (音色 %s %+.1f 半音, %d->%d Hz, 窗口 %d/步长 %d（重叠 %d 帧）, 反馈抑制 %s)%s",
                    cfg.voiceTag(), cfg.pitchSemi, SR_IN, RvcDsp.SR_OUT, SEG, RvcStitch.HOP_SEG, RvcStitch.OV_FR,
                    fbDb <= 0f ? "关" : String.format("%.0fdB", fbDb),
                    sink.enabled ? ", 录制中" : ""));

            final BlockingQueue<float[]> qIn = new ArrayBlockingQueue<>(3);
            final BlockingQueue<float[]> qOut = new ArrayBlockingQueue<>(3);
            final AudioRecord fAr = ar;
            final AtomicLong nblk = new AtomicLong();
            final long tStart = System.currentTimeMillis();
            final AtomicLong dropped = new AtomicLong();

            Thread recT = new Thread(() -> {
                raiseThreadPriority(true);          // 采集丢帧不可逆，给最高音频优先级
                // 滑动窗口：每次**固定**前进 hop 个样本，窗口里始终是最近 SEG 个样本。
                // 固定步长是交叉淡化能对齐的前提（步长漂了，重叠段就不是同一段音频了）。
                final int hop = RvcStitch.HOP_SEG;
                short[] sh = new short[hop];
                float[] win = new float[SEG];
                int fill = 0;
                while (!stopRequested) {
                    int got = 0;
                    while (got < hop && !stopRequested) {
                        int r = fAr.read(sh, got, hop - got);
                        if (r <= 0) break;
                        got += r;
                    }
                    if (got <= 0) continue;
                    if (got < hop) {                 // 短读：补零也要凑满一个 hop，保持步长
                        java.util.Arrays.fill(sh, got, hop, (short) 0);
                        got = hop;
                    }
                    System.arraycopy(win, got, win, 0, SEG - got);
                    for (int i = 0; i < got; i++) win[SEG - got + i] = sh[i] / 32768f;
                    fill += got;
                    if (fill < SEG) continue;        // 首块：先攒够一个窗口
                    float[] cp = new float[SEG];
                    System.arraycopy(win, 0, cp, 0, SEG);
                    if (!qIn.offer(cp)) {            // 跟不上就只留最新块：延迟有界，不会越播越旧
                        qIn.clear();
                        qIn.offer(cp);
                        dropped.incrementAndGet();
                    }
                }
            }, "rt-rec");

            // ---------------- 两级流水线 ----------------
            // 阶段一 hub+fcpe 跑 CPU，阶段二 合成器 跑 Vulkan(或 CPU)，两条线程重叠后
            // 单块吞吐从 (hub+fc+syn) 降到 max(hub+fc, syn)，本机约 380ms -> 270ms。
            final BlockingQueue<Feat> qFeat = new ArrayBlockingQueue<>(3);
            final BlockingQueue<Feat> featPool = new ArrayBlockingQueue<>(6);
            final AtomicLong hubNs = new AtomicLong(), fcNs = new AtomicLong(), synNs = new AtomicLong();

            Thread procT = new Thread(() -> {          // 阶段一：hub + fcpe
                raiseThreadPriority(false);
                float[] featBuf = new float[64 * FEAT];
                while (!stopRequested) {
                    float[] seg;
                    try { seg = qIn.poll(200, java.util.concurrent.TimeUnit.MILLISECONDS); }
                    catch (InterruptedException ie) { break; }
                    if (seg == null) continue;
                    Feat f = featPool.poll();
                    if (f == null) f = new Feat(SEG_FR, PET, FEAT);
                    double energy = 0;
                    for (int i = 0; i < SEG; i++) energy += seg[i] * seg[i];
                    float rms = (float) Math.sqrt(energy / SEG);
                    if (rms < gate) {                  // 输入静音门限：标记静音，交给阶段二出零
                        f.silent = true;
                    } else {
                        f.silent = false;
                        long _t0 = System.nanoTime();
                        int T = engine.hubFeatures(seg, featBuf);
                        hubNs.addAndGet(System.nanoTime() - _t0);
                        if (T <= 0) { featPool.offer(f); continue; }
                        for (int t = 0; t < SEG_FR; t++) {
                            int fr = Math.min(T - 1, (int) ((long) t * T / SEG_FR));
                            System.arraycopy(featBuf, fr * FEAT, f.phone, t * FEAT, FEAT);
                        }
                        long _t1 = System.nanoTime();
                        engine.f0Fill(seg, f.f0, ratio);   // f0 + 变调
                        fcNs.addAndGet(System.nanoTime() - _t1);
                        engine.pitchEmb(f.f0, f.pe);
                    }
                    if (!qFeat.offer(f)) {             // 只保留最新一块，延迟有界
                        Feat old = qFeat.poll();
                        if (old != null) featPool.offer(old);
                        qFeat.offer(f);
                    }
                }
            }, "rt-feat");

            Thread synT = new Thread(() -> {           // 阶段二：合成器
                raiseThreadPriority(false);
                float[] outA = new float[aLen];         // 合成器原始输出（一整窗口）
                final float[][] pool = new float[4][];  // 吐给播放的块 = 窗口 - 重叠段
                for (int i = 0; i < pool.length; i++) pool[i] = new float[hopOut];
                int poolIdx = 0;
                while (!stopRequested) {
                    Feat f;
                    try { f = qFeat.poll(200, java.util.concurrent.TimeUnit.MILLISECONDS); }
                    catch (InterruptedException ie) { break; }
                    if (f == null) continue;
                    float[] res = pool[poolIdx];
                    poolIdx = (poolIdx + 1) % pool.length;
                    if (!f.silent) {
                        long _t2 = System.nanoTime();
                        boolean ok = engine.synthesize(f.phone, f.f0, f.pe, outA);
                        synNs.addAndGet(System.nanoTime() - _t2);
                        featPool.offer(f);
                        if (!ok) continue;
                        // 回环第三步：门控掉残余回声尾巴。不外放 / media 用途没有扬声器回环，别误伤轻声
                        if (useAec) RvcDsp.applyFeedbackGate(outA, fbDb);
                        // 与上一块重叠的那一段做交叉淡化，然后把净新增部分交出去
                        int n = stitch.push(outA, res, 0);
                        if (n < hopOut) java.util.Arrays.fill(res, n, hopOut, 0f);
                    } else {
                        featPool.offer(f);
                        stitch.pushSilence(res, 0);     // 尾巴淡出，别硬切
                    }
                    if (!qOut.offer(res)) { qOut.clear(); qOut.offer(res); }
                    long d = nblk.incrementAndGet();
                    if (d % 5 == 0) {
                        double el = (System.currentTimeMillis() - tStart) / 1000.0;
                        RvcLog.i(TAG, String.format("实时 %d 块 / %.1fs = %.2fx RT, 丢弃 %d | hub %.0fms fc %.0fms syn %.0fms",
                                d, el, d * (RvcStitch.HOP_SEG / 16000.0) / el, dropped.get(),
                                hubNs.getAndSet(0) / 1e6 / 5, fcNs.getAndSet(0) / 1e6 / 5,
                                synNs.getAndSet(0) / 1e6 / 5));
                    }
                }
            }, "rt-syn");

            recT.start();
            procT.start();      // 阶段一：hub + fcpe
            synT.start();       // 阶段二：合成器（与阶段一重叠）

            short[] outSh = new short[hopOut];
            // 欠载时只补「一小截」静音：补一整块（0.44s）会在语音中间留一个明显空洞
            final int starveLen = Math.max(1, stitch.ovOut());
            final float[] silence = new float[starveLen];   // 复用的静音小块
            // 一块音频对应的真实时长。「不外放」模式没有 AudioTrack 的阻塞写来做节流，
            // 必须自己按时间推进，否则 while 会空转、把成倍的静音块灌进录屏音轨。
            final long blockNs = (long) hopOut * 1_000_000_000L / RvcDsp.SR_OUT;
            long nextDeadline = System.nanoTime();
            java.util.Arrays.fill(silence, 0f);
            // 这个线程只做「按节拍往外送」，优先级必须跟上，否则会被两个计算线程挤掉：
            // 一旦它迟到，AudioTrack 缓冲被抽干就是一次欠载（听感同样是一个小顿）。
            raiseThreadPriority(true);
            long starved = 0;
            // 先攒到第一块再开播（最多等 5s）：开头不灌静音，起步就没有空洞
            float[] pending = null;
            long waitUntil = System.currentTimeMillis() + 5000;
            while (!stopRequested && pending == null && System.currentTimeMillis() < waitUntil) {
                try {
                    pending = qOut.poll(100, java.util.concurrent.TimeUnit.MILLISECONDS);
                } catch (InterruptedException ie) {
                    break;
                }
            }
            if (at != null && !stopRequested) at.play();
            RvcLog.i(TAG, "首块" + (pending != null ? "已就绪" : "超时未到")
                    + "，开始对外送音频");
            while (!stopRequested) {
                float[] a;
                if (pending != null) {
                    a = pending;
                    pending = null;
                } else {
                    try {
                        // 短暂等一小会儿：合成偶尔晚个几十毫秒时，等它比立刻塞一整块静音好
                        a = qOut.poll(60, java.util.concurrent.TimeUnit.MILLISECONDS);
                    } catch (InterruptedException ie) {
                        break;
                    }
                }
                int n = hopOut;
                if (a == null) {
                    n = starveLen;
                    java.util.Arrays.fill(outSh, 0, n, (short) 0);
                    // 静音也要喂给录屏，保证音轨时间轴不跳
                    ScreenRecorder.feed(silence, n, 0f);
                    if ((++starved & 31) == 1) {
                        RvcLog.i(TAG, "播放端等待合成（累计 " + starved + " 次），队列 "
                                + qOut.size() + " / qFeat " + qFeat.size());
                    }
                } else {
                    for (int i = 0; i < hopOut; i++) {
                        float v = a[i] * gain;
                        if (v > 1f) v = 1f; else if (v < -1f) v = -1f;
                        outSh[i] = (short) (v * 32767f);
                    }
                    sink.write(a, gain);
                    ScreenRecorder.feed(a, hopOut, gain);
                }
                if (at != null) {
                    at.write(outSh, 0, n);             // 阻塞写，天然按实时速度节流
                } else {
                    // 不外放：手动按实际送出的时长节流（欠载时补的是小截静音）
                    nextDeadline += (long) n * 1_000_000_000L / RvcDsp.SR_OUT;
                    long wait = nextDeadline - System.nanoTime();
                    if (wait > 0) {
                        try {
                            Thread.sleep(wait / 1_000_000L, (int) (wait % 1_000_000L));
                        } catch (InterruptedException ie) { break; }
                    } else if (wait < -3 * blockNs) {
                        nextDeadline = System.nanoTime();     // 落后太多，重新对齐
                    }
                }
            }
            try { recT.join(500); procT.join(500); synT.join(500); } catch (Exception ignored) {}
            if (sink.enabled) {
                try {
                    sink.save(rtPath, RvcDsp.SR_OUT);
                    RvcLog.i(TAG, "实时录音已保存: " + rtPath);
                } catch (Exception e) {
                    RvcLog.i(TAG, "实时录音保存失败: " + e);
                }
            }
            RvcLog.i(TAG, "实时变声停止, 共 " + nblk.get() + " 块");
        } catch (Exception e) {
            RvcLog.i(TAG, "实时错误: " + e);
        } finally {
            try { if (aec != null) { aec.setEnabled(false); aec.release(); aec = null; } } catch (Exception ignored) {}
            try { if (nsFx != null) { nsFx.setEnabled(false); nsFx.release(); nsFx = null; } } catch (Exception ignored) {}
            try { if (agcFx != null) { agcFx.setEnabled(false); agcFx.release(); agcFx = null; } } catch (Exception ignored) {}
            try { if (ar != null) { ar.stop(); ar.release(); } } catch (Exception ignored) {}
            try { if (at != null) { at.stop(); at.release(); } } catch (Exception ignored) {}
            try { ScreenRecorder.stop(); } catch (Exception ignored) {}
            // 退出通话模式，否则会影响系统下次播放/其它 App
            try {
                if (am != null) {
                    am.setSpeakerphoneOn(false);
                    am.setMode(AudioManager.MODE_NORMAL);
                }
            } catch (Exception ignored) {}
            running = false;
            releaseWakeLock();
            try { stopForeground(true); } catch (Exception ignored) {}
            try { stopSelf(); } catch (Exception ignored) {}
        }
    }

    private void srcFailStop() {
        running = false;
        releaseWakeLock();
        try { stopForeground(true); } catch (Exception ignored) {}
        try { stopSelf(); } catch (Exception ignored) {}
    }

    /** 流水线阶段一 → 阶段二 的特征包（池化复用，避免每块分配大数组）。 */
    private static final class Feat {
        float[] phone;
        float[] f0;
        float[] pe;
        boolean silent;

        Feat(int segFr, int pet, int feat) {
            phone = new float[segFr * feat];
            f0 = new float[segFr];
            pe = new float[segFr * pet];
        }
    }

    /**
     * 给音频链路线程提优先级，避免被大小核调度/后台任务挤压造成欠载。
     * 计算线程用 AUDIO 级，采集/播放用 URGENT_AUDIO 级更稳。
     */
    private static void raiseThreadPriority(boolean urgent) {
        try {
            android.os.Process.setThreadPriority(urgent
                    ? android.os.Process.THREAD_PRIORITY_URGENT_AUDIO
                    : android.os.Process.THREAD_PRIORITY_AUDIO);
        } catch (Exception ignored) {}
    }

    /** 实时输出录音缓冲（40kHz 单声道 PCM16 → wav）。 */
    private static class WavSink {
        private final java.io.ByteArrayOutputStream buf = new java.io.ByteArrayOutputStream();
        private boolean enabled;
        private int maxSamples = RvcDsp.SR_OUT * 180;   // 最多 3 分钟

        void write(float[] a, float gain) {
            if (!enabled || buf.size() / 2 >= maxSamples) return;
            for (float v : a) {
                float g = v * gain;
                int s = (int) Math.max(-32768, Math.min(32767, Math.round(g * 32767f)));
                buf.write(s & 0xff);
                buf.write((s >> 8) & 0xff);
            }
        }

        void save(String path, int sr) throws Exception {
            byte[] pcm = buf.toByteArray();
            ByteBuffer bb = ByteBuffer.allocate(44 + pcm.length).order(ByteOrder.LITTLE_ENDIAN);
            bb.put("RIFF".getBytes()); bb.putInt(36 + pcm.length); bb.put("WAVE".getBytes());
            bb.put("fmt ".getBytes()); bb.putInt(16); bb.putShort((short) 1); bb.putShort((short) 1);
            bb.putInt(sr); bb.putInt(sr * 2); bb.putShort((short) 2); bb.putShort((short) 16);
            bb.put("data".getBytes()); bb.putInt(pcm.length);
            bb.put(pcm);
            try (FileOutputStream fo = new FileOutputStream(path)) { fo.write(bb.array()); }
        }
    }
}
