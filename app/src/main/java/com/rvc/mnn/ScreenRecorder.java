package com.rvc.mnn;

import android.content.Context;
import android.content.Intent;
import android.hardware.display.DisplayManager;
import android.hardware.display.VirtualDisplay;
import android.media.MediaCodec;
import android.media.MediaCodecInfo;
import android.media.MediaFormat;
import android.media.MediaMuxer;
import android.media.projection.MediaProjection;
import android.media.projection.MediaProjectionManager;
import android.os.Build;
import android.os.Handler;
import android.os.Looper;
import android.util.DisplayMetrics;
import android.view.Surface;

import java.io.File;
import java.nio.ByteBuffer;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * App 内录屏：把屏幕画面 + <b>实时变声 PCM</b> 合成一个 mp4。
 *
 * <p>为什么需要它：Android 的「不外放但又要有声音」只有这一条可靠路径。
 * 系统/第三方录屏靠 AudioPlaybackCapture，它复制的是「正在播放的音频」，
 * 而播放就意味着扬声器出声（本机 AudioPolicy 也没有 REMOTE_SUBMIX 虚拟输出可绕开）；
 * 音量调 0 连捕获到的副本也是静音。所以由本 App 自己录屏、音轨直接取自实时管线，
 * 声音根本不经过扬声器，就能做到「手机不出声、mp4 里有声音」。
 *
 * <p>音频：{@link RvcDsp#SR_OUT}（40k 模型 40000 / 48k 模型 48000）实时重采样到 48kHz，AAC-LC 单声道 128kbps。
 * 视频：屏幕镜像 H.264（Surface 输入），按屏幕比例缩放，长边上限 1920。
 *
 * <p>调用顺序（必须在<b>前台服务</b>里，Android 14+ 强制 mediaProjection 类型）：
 * <pre>
 *   MediaProjectionManager.createScreenCaptureIntent() → onActivityResult 拿到 resultCode/data
 *   ScreenRecorder.start(ctx, resultCode, data, outPath)   // 服务里
 *   ...
 *   ScreenRecorder.feed(pcm, len, gain)                    // 实时线程每块调用
 *   ScreenRecorder.stop()
 * </pre>
 */
public final class ScreenRecorder {
    private static final String TAG = "RvcScreenRec";

    /** AAC 编码采样率（40000 不在编码器支持列表里，必须重采样）。 */
    public static final int SR_AAC = 48000;

    private static volatile ScreenRecorder INSTANCE;

    /** 是否正在录屏（UI 用）。 */
    public static boolean isRecording() {
        ScreenRecorder r = INSTANCE;
        return r != null && r.running;
    }

    /** 最近一次录制的输出路径（为空表示还没录过）。 */
    private static volatile String lastPath;

    public static String lastPath() {
        return lastPath;
    }

    /**
     * 开始录屏。必须在持有 MediaProjection 的前台服务里调用。
     *
     * @param resultCode MediaProjectionManager.createScreenCaptureIntent() 的结果
     * @param data       同一结果里返回的 Intent
     * @param outPath    输出 mp4 路径
     * @return 成功返回 true；失败已写日志
     */
    public static synchronized boolean start(Context ctx, int resultCode, Intent data, String outPath) {
        if (isRecording()) {
            RvcLog.i(TAG, "已在录屏中，忽略重复请求");
            return true;
        }
        ScreenRecorder r = new ScreenRecorder();
        try {
            r.open(ctx, resultCode, data, outPath);
            INSTANCE = r;
            lastPath = outPath;
            return true;
        } catch (Exception e) {
            RvcLog.i(TAG, "录屏启动失败: " + e);
            try { r.closeQuietly(); } catch (Exception ignored) {}
            return false;
        }
    }

    /** 追加实时变声 PCM（float -1..1，采样率 RvcDsp.SR_OUT）。任意线程可调；未录制时直接丢弃。 */
    public static void feed(float[] pcm, int len, float gain) {
        ScreenRecorder r = INSTANCE;
        if (r == null || !r.running || len <= 0) return;
        r.enqueue(pcm, len, gain);
    }

    public static synchronized void stop() {
        ScreenRecorder r = INSTANCE;
        if (r == null) return;
        INSTANCE = null;
        r.finish();
    }

    // ================================================================== 实例
    private MediaProjection projection;
    private VirtualDisplay display;
    private MediaCodec vEnc, aEnc;
    private Surface inputSurface;
    private MediaMuxer muxer;

    private int vTrack = -1, aTrack = -1;
    private boolean muxerStarted;
    private final Object muxLock = new Object();

    private Thread pump;
    private volatile boolean running;
    private final AtomicBoolean stopRequested = new AtomicBoolean(false);
    private final BlockingQueue<byte[]> aq = new ArrayBlockingQueue<>(24);

    private int outW, outH;
    /**
     * 视频轨道首帧的原始 pts（用于把视频时间轴归一到 0 起）。
     *
     * <p>Surface 输入编码器输出的 pts 来自帧提交时的系统时间戳，很多 ROM 上是
     * {@code uptimeMillis()*1000}（开机至今的微秒）。直接写进 MediaMuxer 会把 mp4
     * 的时长撑成"开机至今"（几天量级），同时这帧在音轨之后，看起来就是一张静态图。
     * 用它做基准把视频样本归一化到从 0 开始，与音频轨道（本来就 0 起）对齐。
     */
    private long videoPtsBaseUs = -1L;
    private long startNs;
    private final MediaCodec.BufferInfo vInfo = new MediaCodec.BufferInfo();
    private final MediaCodec.BufferInfo aInfo = new MediaCodec.BufferInfo();

    private ScreenRecorder() {}

    private void open(Context ctx, int resultCode, Intent data, String outPath) throws Exception {
        MediaProjectionManager mpm = (MediaProjectionManager)
                ctx.getSystemService(Context.MEDIA_PROJECTION_SERVICE);
        if (mpm == null) throw new Exception("没有 MEDIA_PROJECTION 服务");

        // Android 14+ 必须在 createVirtualDisplay 前 registerCallback
        projection = mpm.getMediaProjection(resultCode, data);
        if (projection == null) throw new Exception("MediaProjection 为空（授权被拒？）");
        projection.registerCallback(new MediaProjection.Callback() {
            @Override public void onStop() {
                RvcLog.i(TAG, "MediaProjection 被系统停止");
                finish();
            }
        }, new Handler(Looper.getMainLooper()));

        DisplayMetrics dm = ctx.getResources().getDisplayMetrics();
        int sw = dm.widthPixels, sh = dm.heightPixels;
        // 长边限制 1920，宽高取偶数（H.264 编码器要求）
        double scale = Math.min(1.0, 1920.0 / Math.max(sw, sh));
        outW = (int) (sw * scale) & ~1;
        outH = (int) (sh * scale) & ~1;
        if (outW <= 0 || outH <= 0) throw new Exception("屏幕尺寸异常: " + sw + "x" + sh);

        // ---- 视频编码器（Surface 输入）----
        MediaFormat vf = MediaFormat.createVideoFormat(MediaFormat.MIMETYPE_VIDEO_AVC, outW, outH);
        vf.setInteger(MediaFormat.KEY_COLOR_FORMAT,
                MediaCodecInfo.CodecCapabilities.COLOR_FormatSurface);
        vf.setInteger(MediaFormat.KEY_BIT_RATE, 6_000_000);
        vf.setInteger(MediaFormat.KEY_FRAME_RATE, 30);
        vf.setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 2);
        vEnc = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_VIDEO_AVC);
        vEnc.configure(vf, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE);
        inputSurface = vEnc.createInputSurface();
        vEnc.start();

        // ---- 音频编码器（AAC-LC）----
        MediaFormat af = MediaFormat.createAudioFormat(MediaFormat.MIMETYPE_AUDIO_AAC, SR_AAC, 1);
        af.setInteger(MediaFormat.KEY_AAC_PROFILE, MediaCodecInfo.CodecProfileLevel.AACObjectLC);
        af.setInteger(MediaFormat.KEY_BIT_RATE, 128_000);
        af.setInteger(MediaFormat.KEY_MAX_INPUT_SIZE, 131072);   // 一块 PCM ~47KB，不能截断
        aEnc = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_AUDIO_AAC);
        aEnc.configure(af, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE);
        aEnc.start();

        // ---- Muxer ----
        File f = new File(outPath);
        File dir = f.getParentFile();
        if (dir != null && !dir.isDirectory() && !dir.mkdirs()) {
            throw new Exception("无法创建输出目录: " + dir);
        }
        muxer = new MediaMuxer(outPath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4);

        // ---- 虚拟显示（镜像屏幕到编码器 Surface）----
        display = projection.createVirtualDisplay("rvc-rec", outW, outH, dm.densityDpi,
                DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
                inputSurface, null, null);

        running = true;
        startNs = System.nanoTime();
        pump = new Thread(this::pumpLoop, "rvc-rec-pump");
        pump.start();

        RvcLog.i(TAG, String.format("录屏开始: %dx%d @30fps, 音频 %dHz AAC, -> %s",
                outW, outH, SR_AAC, outPath));
    }

    // ------------------------------------------------------------------ 音频入队
    private void enqueue(float[] pcm, int len, float gain) {
        try {
            // 重采样 RvcDsp.SR_OUT -> 48000（等于 48k 模型时是恒等变换）
            final int n = len;
            final int m = (int) Math.round(n * (double) SR_AAC / RvcDsp.SR_OUT);
            byte[] bytes = new byte[m * 2];
            for (int i = 0; i < m; i++) {
                double pos = i * (double) RvcDsp.SR_OUT / SR_AAC;
                int i0 = (int) pos;
                if (i0 >= n) i0 = n - 1;
                if (i0 < 0) i0 = 0;
                int i1 = Math.min(i0 + 1, n - 1);
                float fr = (float) (pos - i0);
                float v = (pcm[i0] * (1 - fr) + pcm[i1] * fr) * gain;
                if (v > 1f) v = 1f; else if (v < -1f) v = -1f;
                short s = (short) (v * 32767f);
                bytes[i * 2] = (byte) (s & 0xff);
                bytes[i * 2 + 1] = (byte) ((s >> 8) & 0xff);
            }
            if (!aq.offer(bytes)) {          // 跟不上就丢最旧的，保证延迟有界
                aq.poll();
                aq.offer(bytes);
            }
        } catch (Exception e) {
            RvcLog.i(TAG, "音频入队失败: " + e);
        }
    }

    // ------------------------------------------------------------------ 编码泵
    private void pumpLoop() {
        long ptsUs = 0;
        try {
            while (!stopRequested.get()) {
                boolean didWork = false;
                didWork |= drainVideo(false);
                didWork |= feedAudio();
                didWork |= drainAudio(false);
                if (!didWork) {
                    sleep(5);
                }
            }
            // 收尾：先把音频队列排空，再送 EOS，最后把两路残留帧全部写出
            for (int i = 0; i < 100 && !aq.isEmpty(); i++) {
                feedAudio();
                drainAudio(false);
                sleep(5);
            }
            try { if (vEnc != null) vEnc.signalEndOfInputStream(); } catch (Exception ignored) {}
            try { signalAudioEos(); } catch (Exception ignored) {}
            for (int i = 0; i < 300; i++) {          // 最多等 ~3s
                boolean more = false;
                more |= drainVideo(true);
                more |= drainAudio(true);
                if (!more) break;
            }
        } catch (Exception e) {
            RvcLog.i(TAG, "编码线程异常: " + e);
        }
    }

    /** 视频：Surface 输入，只 drain 输出。 */
    private boolean drainVideo(boolean eos) throws Exception {
        if (vEnc == null) return false;
        boolean busy = false;
        while (true) {
            int idx = vEnc.dequeueOutputBuffer(vInfo, eos ? 30_000 : 0);
            if (idx == MediaCodec.INFO_TRY_AGAIN_LATER) {
                if (eos) break;
                return busy;
            }
            if (idx == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                synchronized (muxLock) {
                    if (vTrack < 0) {
                        vTrack = muxer.addTrack(vEnc.getOutputFormat());
                        tryStartMuxer();
                    }
                }
                busy = true;
                continue;
            }
            if (idx < 0) continue;
            ByteBuffer buf = vEnc.getOutputBuffer(idx);
            if (buf != null && vInfo.size > 0 && (vInfo.flags & MediaCodec.BUFFER_FLAG_CODEC_CONFIG) == 0) {
                buf.position(vInfo.offset);
                buf.limit(vInfo.offset + vInfo.size);
                // 归一化视频 pts（见 videoPtsBaseUs 注释）
                if (videoPtsBaseUs < 0) {
                    videoPtsBaseUs = vInfo.presentationTimeUs;
                    if (videoPtsBaseUs > 5_000_000L) {
                        RvcLog.i(TAG, "视频首帧 pts=" + videoPtsBaseUs + "us，按基准归一化");
                    }
                }
                vInfo.presentationTimeUs = Math.max(0L, vInfo.presentationTimeUs - videoPtsBaseUs);
                synchronized (muxLock) {
                    if (muxerStarted && vTrack >= 0) {
                        muxer.writeSampleData(vTrack, buf, vInfo);
                    }
                }
            }
            vEnc.releaseOutputBuffer(idx, false);
            busy = true;
            if ((vInfo.flags & MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) break;
        }
        return busy;
    }

    /** 音频累计样本数，用来算精确 pts（墙钟会让 AAC 时间轴漂移）。 */
    private long aSamples;

    private long nextAPtsUs() {
        long us = (long) ((double) aSamples / SR_AAC * 1_000_000L);
        return us;
    }

    /** 音频：从队列取 PCM 填进编码器。数据块按 input buffer 大小分多次入队，绝不截断。 */
    private boolean feedAudio() throws Exception {
        if (aEnc == null) return false;
        boolean busy = false;
        while (true) {
            if (pendBuf == null) {
                pendBuf = aq.poll();
                pendOff = 0;
                if (pendBuf == null) return busy;
            }
            int idx = aEnc.dequeueInputBuffer(10_000);
            if (idx < 0) return busy;          // 编码器满，下次再喂
            ByteBuffer in = aEnc.getInputBuffer(idx);
            if (in == null) {
                aEnc.queueInputBuffer(idx, 0, 0, nextAPtsUs(), 0);
                continue;
            }
            in.clear();
            int avail = in.remaining();
            int n = Math.min(avail, pendBuf.length - pendOff);
            in.put(pendBuf, pendOff, n);
            aEnc.queueInputBuffer(idx, 0, n, nextAPtsUs(), 0);
            aSamples += n / 2;
            pendOff += n;
            busy = true;
            if (pendOff >= pendBuf.length) {      // 这一块喂完了
                pendBuf = null;
                pendOff = 0;
            }
        }
    }

    /** feedAudio 的跨调用状态：一块 PCM 可能被拆到多个 input buffer。 */
    private byte[] pendBuf;
    private int pendOff;

    /** 给音频编码器送 EOS，让 drain 能收尾。 */
    private void signalAudioEos() throws Exception {
        if (aEnc == null) return;
        int idx = aEnc.dequeueInputBuffer(50_000);
        if (idx >= 0) {
            aEnc.queueInputBuffer(idx, 0, 0, nextAPtsUs(),
                    MediaCodec.BUFFER_FLAG_END_OF_STREAM);
        }
    }

    /** 音频：drain 输出。 */
    private boolean drainAudio(boolean eos) throws Exception {
        if (aEnc == null) return false;
        boolean busy = false;
        while (true) {
            int idx = aEnc.dequeueOutputBuffer(aInfo, eos ? 30_000 : 0);
            if (idx == MediaCodec.INFO_TRY_AGAIN_LATER) {
                if (eos) break;
                return busy;
            }
            if (idx == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                synchronized (muxLock) {
                    if (aTrack < 0) {
                        aTrack = muxer.addTrack(aEnc.getOutputFormat());
                        tryStartMuxer();
                    }
                }
                busy = true;
                continue;
            }
            if (idx < 0) continue;
            ByteBuffer buf = aEnc.getOutputBuffer(idx);
            if (buf != null && aInfo.size > 0 && (aInfo.flags & MediaCodec.BUFFER_FLAG_CODEC_CONFIG) == 0) {
                buf.position(aInfo.offset);
                buf.limit(aInfo.offset + aInfo.size);
                synchronized (muxLock) {
                    if (muxerStarted && aTrack >= 0) {
                        muxer.writeSampleData(aTrack, buf, aInfo);
                    }
                }
            }
            aEnc.releaseOutputBuffer(idx, false);
            busy = true;
            if ((aInfo.flags & MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) break;
        }
        return busy;
    }

    /** 两路 track 都加进来之后才能 start（muxer 的硬性要求）。 */
    private void tryStartMuxer() {
        if (muxerStarted || vTrack < 0 || aTrack < 0) return;
        try {
            muxer.start();
            muxerStarted = true;
            RvcLog.i(TAG, "muxer 已启动 (video=" + vTrack + ", audio=" + aTrack + ")");
        } catch (Exception e) {
            RvcLog.i(TAG, "muxer.start 失败: " + e);
        }
    }

    private static void sleep(long ms) {
        try { Thread.sleep(ms); } catch (InterruptedException ignored) {}
    }

    // ------------------------------------------------------------------ 收尾
    private final AtomicBoolean closed = new AtomicBoolean(false);

    private void finish() {
        running = false;
        if (!closed.compareAndSet(false, true)) return;
        stopRequested.set(true);
        try { if (pump != null) pump.join(3000); } catch (InterruptedException ignored) {}

        try { if (display != null) display.release(); } catch (Exception ignored) {}
        display = null;

        try { if (vEnc != null) { vEnc.stop(); vEnc.release(); } } catch (Exception ignored) {}
        vEnc = null;
        try { if (aEnc != null) { aEnc.stop(); aEnc.release(); } } catch (Exception ignored) {}
        aEnc = null;
        try { if (inputSurface != null) inputSurface.release(); } catch (Exception ignored) {}
        inputSurface = null;

        String path = lastPath;
        try {
            synchronized (muxLock) {
                if (muxer != null) {
                    if (muxerStarted) muxer.stop();
                    muxer.release();
                }
            }
        } catch (Exception e) {
            RvcLog.i(TAG, "muxer 收尾: " + e);
        }
        muxer = null;
        muxerStarted = false;

        try { if (projection != null) projection.stop(); } catch (Exception ignored) {}
        projection = null;

        if (path != null) {
            java.io.File f = new java.io.File(path);
            RvcLog.i(TAG, "录屏结束: " + path + " (" + (f.length() / 1024 / 1024) + " MB)");
        } else {
            RvcLog.i(TAG, "录屏结束");
        }
    }

    private void closeQuietly() {
        try { finish(); } catch (Exception ignored) {}
    }
}
