package com.rvc.mnn;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.net.Uri;
import android.provider.OpenableColumns;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.text.method.ScrollingMovementMethod;
import android.view.Gravity;
import android.view.View;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import com.google.android.material.button.MaterialButton;
import com.google.android.material.dialog.MaterialAlertDialogBuilder;

import com.a2942.silkq.Injector;
import com.a2942.silkq.QqDraftInjector;
import com.a2942.silkq.Silk;

import java.io.File;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;
import java.util.Locale;

/**
 * 录音模式：**录完即停 · 不实时 · 不外放**。
 *
 * <p>工作方式与「④ 实时变声」是两条完全独立的路径：
 * <ol>
 *   <li>点「开始录音」→ 只开 AudioRecord 往 WAV 里灌数据。
 *       这段期间**不做任何推理**，也**不创建 AudioTrack**，手机一声不吭。</li>
 *   <li>点「停止并变声」→ 录音结束，WAV 封口，
 *       然后才把整段音频丢给 {@link OfflineConvert} 做一次离线变声（RVC 推理）。</li>
 *   <li>产出 {@code *_rvc_*.wav}；可以直接转成 SILK 语音，
 *       或用 root/Shizuku 写进 QQ 语音草稿。</li>
 * </ol>
 *
 * <p><b>也可以不录音</b>：点「选择音频文件」挑一段现成的语音（wav / mp3 / m4a /
 * aac / amr / flac…，系统能解的都行），走同一条离线变声链路；勾上
 * 「转换完直接发 QQ」之后，选完文件就会自动跑完
 * 变声 → SILK → 写进 QQ 语音草稿，最后只剩你在 QQ 里按一下发送。
 *
 * <p>和实时链路相比的好处：没有「处理跟不上采集」的焦虑，也不需要
 * 反馈抑制门控（根本没有扬声器路径）；整段一次吃，吞吐通常比 1.0x 实时快。
 */
public class RecordActivity extends AppCompatActivity {

    private static final int REQ_MIC = 2001;
    private static final int REQ_PICK_AUDIO = 2002;
    /** QQ 的包名（写完草稿后跳过去用）。 */
    private static final String QQ_PKG = "com.tencent.mobileqq";
    private static final int SR = RvcDsp.SR_IN;           // 录音就用 16 kHz，RVC 的入口采样率
    private static final double MAX_SECONDS = 300d;       // 兜底：最长录 5 分钟

    private static final int ST_IDLE = 0;
    private static final int ST_RECORDING = 1;
    private static final int ST_CONVERTING = 2;
    private static final int ST_READY = 3;

    private final Handler ui = new Handler(Looper.getMainLooper());

    private Config cfg;
    private RvcEngine engine;
    private int state = ST_IDLE;

    private Button btnToggle;
    private Button btnPick;
    private Button btnConvert;
    private Button btnAutoSend;
    private Button btnWarm;
    private Button btnSilk;
    private Button btnDraft;
    private Button btnPreview;
    private TextView tvStatus;
    private TextView tvTimer;
    private TextView tvResult;
    private TextView tvLog;

    private Thread recThread;
    private volatile boolean recording;
    private WavIo.Writer wavWriter;
    private File recWav;
    /** 当前要变声的输入（录音产物，或「选择音频文件」导入的那一份）。 */
    private File srcWav;
    private String srcName;
    private File rvcWav;
    private File silkFile;
    private long recStartMs;
    /** 转换完是否自动转 SILK + 写进 QQ 语音草稿。 */
    private boolean autoSend = true;

    private int bitrate = 16000;

    // ================================================================== 生命周期
    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        cfg = new Config();
        cfg.load(this);
        buildUi();
        RvcLog.setSink(s -> ui.post(() -> appendLog(s)));
        prepareEngineAsync();
        probeBackendAsync();
        handleIntent(getIntent());
    }

    @Override
    protected void onNewIntent(Intent intent) {
        super.onNewIntent(intent);
        setIntent(intent);
        handleIntent(intent);
    }

    /**
     * 命令行入口：{@code am start -n com.rvc.mnn/.RecordActivity --es in /sdcard/a.mp3 --ez auto true}
     * 有 root 时可以直接把外部音频喂进来跑完整条「导入 → 变声 → SILK → 写 QQ 草稿」。
     */
    private void handleIntent(Intent it) {
        if (it == null || !it.hasExtra("in")) return;
        String in = it.getStringExtra("in");
        if (in == null || in.isEmpty()) return;
        final File f = new File(in);
        final boolean auto = it.getBooleanExtra("auto", false);
        ui.postDelayed(() -> importFromPath(f, auto), 400);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        stopRecordingQuietly();
        RvcLog.setSink(null);
        // 与 MainActivity 一致：实时服务在跑时不要抢着释放 native session
        if (!RealtimeService.running && engine != null) {
            engine.releaseAll();
        }
    }

    // ================================================================== UI
    private int dp(int v) {
        return (int) (v * getResources().getDisplayMetrics().density);
    }

    /** M3 bodyMedium。 */
    private TextView label(String s) {
        TextView t = new TextView(this);
        t.setText(s);
        t.setTextAppearance(com.google.android.material.R.style.TextAppearance_Material3_BodyMedium);
        t.setAlpha(0.78f);
        t.setPadding(0, dp(4), 0, dp(4));
        return t;
    }

    /** M3 titleMedium + 主色竖条。 */
    private void header(LinearLayout root, String s) {
        LinearLayout row = new LinearLayout(this);
        row.setOrientation(LinearLayout.HORIZONTAL);
        android.view.View bar = new android.view.View(this);
        android.util.TypedValue tv = new android.util.TypedValue();
        getTheme().resolveAttribute(com.google.android.material.R.attr.colorPrimary, tv, true);
        bar.setBackgroundColor(tv.data);
        LinearLayout.LayoutParams barLp = new LinearLayout.LayoutParams(dp(3), dp(18));
        barLp.setMargins(0, dp(16), dp(8), dp(4));
        row.addView(bar, barLp);

        TextView t = new TextView(this);
        t.setText(s);
        t.setTextAppearance(com.google.android.material.R.style.TextAppearance_Material3_TitleMedium);
        t.setPadding(0, dp(12), 0, dp(6));
        row.addView(t);
        root.addView(row);
    }

    /** M3 filled button。 */
    private Button mkButton(String s, View.OnClickListener l) {
        MaterialButton b = new MaterialButton(this);
        b.setText(s);
        b.setAllCaps(false);
        b.setOnClickListener(l);
        return b;
    }

    private void buildUi() {
        ScrollView sv = new ScrollView(this);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(dp(12), dp(12), dp(12), dp(12));
        sv.addView(root);
        setContentView(sv);

        header(root, "录音模式 · 录完即停 / 不实时 / 不外放");
        root.addView(label("按「开始录音」→ 说完整段 → 按「停止并变声」。"
                + "\n录音期间不做推理、不建 AudioTrack，手机全程不出声。"
                + "\n不想录？下面可以直接挑一段现成的音频文件来变声。"));
        root.addView(label("采样率 " + SR + " Hz · 单声道 · 16-bit · 上限 "
                + (int) MAX_SECONDS + " 秒"));

        tvStatus = label("");
        tvStatus.setTextAppearance(
                com.google.android.material.R.style.TextAppearance_Material3_TitleSmall);
        tvStatus.setAlpha(1f);
        root.addView(tvStatus);

        tvTimer = label("");
        tvTimer.setTextAppearance(
                com.google.android.material.R.style.TextAppearance_Material3_HeadlineMedium);
        tvTimer.setAlpha(1f);
        tvTimer.setGravity(Gravity.CENTER);
        tvTimer.setPadding(0, dp(12), 0, dp(12));
        root.addView(tvTimer);

        btnToggle = mkButton("开始录音", v -> onToggle());
        root.addView(btnToggle, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT));

        header(root, "或者：用现成的音频文件");
        root.addView(label("挑一段已经存在的语音（wav / mp3 / m4a / aac / amr / flac…），"
                + "走同一条离线变声链路，不用真的去录音。"));

        btnPick = mkButton("选择音频文件", v -> pickAudio());
        root.addView(btnPick, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT));

        LinearLayout rowFile = new LinearLayout(this);
        rowFile.setOrientation(LinearLayout.HORIZONTAL);
        btnConvert = mkButton("转换", v -> convertFile());
        btnAutoSend = mkButton(autoSendLabel(), v -> {
            autoSend = !autoSend;
            ((Button) v).setText(autoSendLabel());
        });
        rowFile.addView(btnConvert, new LinearLayout.LayoutParams(0,
                LinearLayout.LayoutParams.WRAP_CONTENT, 1f));
        rowFile.addView(btnAutoSend, new LinearLayout.LayoutParams(0,
                LinearLayout.LayoutParams.WRAP_CONTENT, 1f));
        root.addView(rowFile);

        btnWarm = mkButton("重新预热引擎", v -> warmNow());
        root.addView(btnWarm, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT));

        header(root, "产物处理");
        tvResult = label("还没有录音结果");
        root.addView(tvResult);

        LinearLayout row1 = new LinearLayout(this);
        row1.setOrientation(LinearLayout.HORIZONTAL);
        Button btnBr = mkButton("码率 " + (bitrate / 1000) + "k", v -> {
            int[] all = Silk.BITRATES;
            int idx = 0;
            for (int i = 0; i < all.length; i++) {
                if (all[i] == bitrate) {
                    idx = i;
                    break;
                }
            }
            bitrate = all[(idx + 1) % all.length];
            ((Button) v).setText("码率 " + (bitrate / 1000) + "k");
        });
        btnSilk = mkButton("转 SILK", v -> onMakeSilk());
        row1.addView(btnBr, new LinearLayout.LayoutParams(0,
                LinearLayout.LayoutParams.WRAP_CONTENT, 1f));
        row1.addView(btnSilk, new LinearLayout.LayoutParams(0,
                LinearLayout.LayoutParams.WRAP_CONTENT, 1f));
        root.addView(row1);

        LinearLayout row2 = new LinearLayout(this);
        row2.setOrientation(LinearLayout.HORIZONTAL);
        btnDraft = mkButton("写 QQ 草稿", v -> onInjectDraft());
        btnPreview = mkButton("试听（手动）", v -> onPreview());
        row2.addView(btnDraft, new LinearLayout.LayoutParams(0,
                LinearLayout.LayoutParams.WRAP_CONTENT, 1f));
        row2.addView(btnPreview, new LinearLayout.LayoutParams(0,
                LinearLayout.LayoutParams.WRAP_CONTENT, 1f));
        root.addView(row2);

        header(root, "日志");
        tvLog = label("");
        tvLog.setTextSize(11f);
        tvLog.setMovementMethod(new ScrollingMovementMethod());
        root.addView(tvLog);

        updateButtons();
    }

    private String autoSendLabel() {
        return autoSend ? "转换后直接发 QQ：开" : "转换后直接发 QQ：关";
    }

    private void updateButtons() {
        boolean busy = state == ST_RECORDING || state == ST_CONVERTING;
        btnToggle.setEnabled(!busy || state == ST_RECORDING);
        btnToggle.setText(state == ST_RECORDING ? "停止并变声" : "开始录音");
        btnPick.setEnabled(!busy);
        btnConvert.setEnabled(!busy && srcWav != null && srcWav.isFile());
        btnWarm.setEnabled(!busy);
        boolean ready = state == ST_READY && rvcWav != null && rvcWav.isFile();
        btnSilk.setEnabled(ready && !busy);
        btnDraft.setEnabled(ready && silkFile != null && !busy);
        btnPreview.setEnabled(ready && !busy);
    }

    private void setState(int s) {
        state = s;
        ui.post(this::updateButtons);
    }

    private void appendLog(String s) {
        tvLog.append(s + "\n");
        int sc = tvLog.getLayout() == null ? 0 : tvLog.getLayout().getLineTop(tvLog.getLineCount())
                - tvLog.getHeight();
        if (sc > 0) tvLog.scrollTo(0, sc);
    }

    private void log(String s) {
        RvcLog.i("RvcRec", s);
    }

    private void toast(String s) {
        Toast.makeText(this, s, Toast.LENGTH_SHORT).show();
    }

    // ================================================================== 引擎 / 后端
    private void prepareEngineAsync() {
        new Thread(() -> {
            try {
                if (!ModelBootstrap.prepare(this, cfg)) {
                    ui.post(() -> setStatus("模型未就绪（内置解压失败？去主界面看看）"));
                    return;
                }
                engine = RvcEngine.get(this);
                boolean ok = engine.ensure(cfg);
                EngineWarmup.Result warm = null;
                if (ok) {
                    // 进页面就把整链跑热：Vulkan 管线编译 / ORT arena / CPU kernel 选择
                    // 都摊在这里，而不是摊在用户点「停止并变声」之后。
                    ui.post(() -> setStatus("引擎就绪，正在预热（首次会稍慢）…"));
                    warm = EngineWarmup.warmIfNeeded(engine, cfg, this::log, 2);
                }
                final EngineWarmup.Result wr = warm;
                ui.post(() -> {
                    setStatus("引擎 " + (ok ? "就绪" : "初始化失败")
                            + " · 音色 " + cfg.voiceTag()
                            + " · " + RvcEngine.backendInfo()
                            + (wr != null && wr.ok ? "\n" + wr.summary() : ""));
                    log("engine ok=" + ok + " backend=" + RvcEngine.backendInfo());
                });
            } catch (Throwable t) {
                ui.post(() -> setStatus("引擎异常：" + t));
            }
        }, "rec-engine").start();
    }

    private void probeBackendAsync() {
        new Thread(() -> {
            boolean ready = QqDraftInjector.ready();
            String name = QqDraftInjector.backendName();
            String desc = QqDraftInjector.describe();
            ui.post(() -> {
                log("QQ 草稿后端: " + desc);
                if (!ready) {
                    log("没有 root / Shizuku —— 「写 QQ 草稿」会不可用；"
                            + "在 KernelSU/Magisk 里给本 App 授权后会变成 root");
                }
            });
        }, "rec-backend").start();
    }

    /** 手动重新预热（换了引擎预设 / 刚装完模型后想立刻热起来时用）。 */
    private void warmNow() {
        if (state == ST_RECORDING || state == ST_CONVERTING) {
            toast("正在忙，稍后再预热");
            return;
        }
        final Button b = btnWarm;
        b.setEnabled(false);
        setStatus("正在预热引擎…");
        new Thread(() -> {
            try {
                if (engine == null) {
                    if (!ModelBootstrap.prepare(this, cfg)) {
                        throw new Exception("模型未就绪");
                    }
                    engine = RvcEngine.get(this);
                }
                if (!engine.ensure(cfg)) {
                    throw new Exception("引擎初始化失败");
                }
                EngineWarmup.Result r = EngineWarmup.warm(engine, cfg, this::log, 2);
                ui.post(() -> {
                    b.setEnabled(true);
                    setStatus(r.summary());
                    toast(r.ok ? "预热完成" : "预热失败");
                });
            } catch (Throwable t) {
                log("预热失败：" + t);
                ui.post(() -> {
                    b.setEnabled(true);
                    setStatus("预热失败：" + t);
                });
            }
        }, "rec-warm").start();
    }

    private void setStatus(String s) {
        tvStatus.setText(s);
        log(s);
    }

    // ================================================================== 录音
    private void onToggle() {
        if (state == ST_RECORDING) {
            stopAndConvert();
        } else if (state == ST_IDLE || state == ST_READY) {
            startRecording();
        }
    }

    private void startRecording() {
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO)
                != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.RECORD_AUDIO}, REQ_MIC);
            return;
        }
        try {
            recWav = new File(outDir(), "rvc_rec_"
                    + new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(new Date())
                    + ".wav");
            wavWriter = new WavIo.Writer(recWav, SR);
        } catch (Exception e) {
            toast("建录音文件失败：" + e);
            log("建录音文件失败：" + e);
            return;
        }
        srcWav = recWav;
        srcName = recWav.getName();
        rvcWav = null;
        silkFile = null;
        recording = true;
        recStartMs = System.currentTimeMillis();
        setState(ST_RECORDING);
        setStatus("录音中…（不做推理、不出声）");
        tvResult.setText("录音中：" + recWav.getName());

        recThread = new Thread(() -> {
            AudioRecord ar = null;
            try {
                int min = AudioRecord.getMinBufferSize(SR,
                        AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT);
                int bufSize = Math.max(min, SR / 5 * 2 * 4);
                ar = new AudioRecord(MediaRecorder.AudioSource.VOICE_RECOGNITION, SR,
                        AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, bufSize);
                if (ar.getState() != AudioRecord.STATE_INITIALIZED) {
                    ui.post(() -> failRecord("AudioRecord 初始化失败"));
                    return;
                }
                ar.startRecording();
                log("录音开始 -> " + recWav.getAbsolutePath());
                short[] buf = new short[SR / 10];   // 100ms 一块
                long lastTick = 0;
                while (recording) {
                    int n = ar.read(buf, 0, buf.length);
                    if (n > 0) {
                        wavWriter.write(buf, n);
                        long now = System.currentTimeMillis();
                        if (now - lastTick > 100) {
                            lastTick = now;
                            final double sec = wavWriter.seconds();
                            ui.post(() -> tvTimer.setText(String.format(Locale.US,
                                    "● REC %.1f s", sec)));
                            if (sec >= MAX_SECONDS) {
                                recording = false;
                                ui.post(this::stopAndConvert);
                            }
                        }
                    } else if (n < 0) {
                        log("AudioRecord.read 错误 " + n);
                        break;
                    }
                }
                try {
                    ar.stop();
                } catch (Exception ignored) {
                }
                ar.release();
                ar = null;
                wavWriter.close();
                final double sec = wavWriter.seconds();
                log(String.format("录音结束 %.2fs -> %s", sec, recWav.getAbsolutePath()));
                ui.post(() -> {
                    tvTimer.setText(String.format(Locale.US, "已录 %.1f s", sec));
                    startConvert();
                });
            } catch (Throwable t) {
                log("录音异常：" + t);
                if (ar != null) {
                    try {
                        ar.release();
                    } catch (Exception ignored) {
                    }
                }
                ui.post(() -> failRecord("录音异常：" + t));
            }
        }, "rec-capture");
        recThread.start();
    }

    private void failRecord(String msg) {
        recording = false;
        setState(ST_IDLE);
        setStatus(msg);
        toast(msg);
    }

    private void stopRecordingQuietly() {
        recording = false;
        Thread t = recThread;
        recThread = null;
        if (t != null) {
            try {
                t.join(2000);
            } catch (InterruptedException ignored) {
            }
        }
        if (wavWriter != null) {
            wavWriter.close();
            wavWriter = null;
        }
    }

    private void stopAndConvert() {
        if (state != ST_RECORDING) return;
        setStatus("正在收尾录音…");
        // recording=false 让采集线程退出循环，它会在退出时自行封口并调 startConvert()
        recording = false;
    }

    // ================================================================== 离线变声
    private void startConvert() {
        startConvert(recWav);
    }

    /** 对指定输入做一次离线变声（录音产物和外部音频文件共用这条路径）。 */
    private void startConvert(final File inWav) {
        if (inWav == null || !inWav.isFile() || inWav.length() <= 44) {
            failRecord("输入音频太短或为空");
            return;
        }
        setState(ST_CONVERTING);
        setStatus("离线变声中（不实时、不外放）…");
        tvResult.setText("输入：" + inWav.getName());

        new Thread(() -> {
            try {
                if (engine == null) {
                    if (!ModelBootstrap.prepare(this, cfg)) {
                        throw new Exception("模型未就绪");
                    }
                    engine = RvcEngine.get(this);
                }
                if (!engine.ensure(cfg)) {
                    throw new Exception("引擎初始化失败");
                }
                if (!EngineWarmup.isWarm(cfg)) {
                    // 录制期间用户可能换过引擎预设，这里补一次（配置没变时是空操作）
                    EngineWarmup.warmIfNeeded(engine, cfg, this::log, 1);
                }
                String name = inWav.getName();
                int dot = name.lastIndexOf('.');
                String base = dot > 0 ? name.substring(0, dot) : name;
                File out = new File(outDir(), base + "_rvc_" + cfg.voiceTag() + ".wav");
                double sec = OfflineConvert.convert(engine, cfg, inWav.getAbsolutePath(),
                        out.getAbsolutePath(), this::log);
                final File f = out;
                ui.post(() -> {
                    rvcWav = f;
                    silkFile = null;
                    setState(ST_READY);
                    setStatus("变声完成");
                    tvResult.setText(String.format(Locale.US,
                            "输出：%s\n时长 %.2fs · %.0f KB\n%s",
                            f.getName(), sec, f.length() / 1024.0, f.getAbsolutePath()));
                    updateButtons();
                    toast("变声完成");
                    if (autoSend) sendResult(f);
                });
            } catch (Throwable t) {
                log("变声失败：" + t);
                ui.post(() -> {
                    setState(ST_IDLE);
                    setStatus("变声失败：" + t);
                    toast("变声失败：" + t);
                });
            }
        }, "rec-convert").start();
    }

    // ================================================================== 选择音频文件
    /**
     * 挑一段现成的音频来变声（wav / mp3 / m4a / aac / amr / flac…）。
     *
     * <p>SAF 只给 Uri，这里先让 {@link AudioImport} 落成 WAV 到 Download，
     * 之后和录音产物完全同一条链路，不用为「文件输入」再写一套推理代码。
     */
    private void pickAudio() {
        if (state == ST_RECORDING || state == ST_CONVERTING) {
            toast("正在忙，稍后再选");
            return;
        }
        Intent i = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        i.addCategory(Intent.CATEGORY_OPENABLE);
        i.setType("audio/*");
        try {
            startActivityForResult(i, REQ_PICK_AUDIO);
        } catch (Exception e) {
            toast("打不开文件选择器：" + e);
            log("打开文件选择器失败：" + e);
        }
    }

    /** 导入进度：每 10% 才刷一次状态（否则日志会被刷屏）。 */
    private AudioImport.Progress importProgress() {
        return new AudioImport.Progress() {
            private int last = -10;

            @Override
            public void onProgress(int p) {
                if (p < last + 10 && p < 100) return;
                last = p;
                final int v = p;
                ui.post(() -> setStatus("导入音频 " + v + "%"));
            }
        };
    }

    /** 读文件选择器给的名字（拿不到就用默认名）。 */
    private String displayName(Uri uri) {
        try (Cursor c = getContentResolver().query(uri, null, null, null, null)) {
            if (c != null && c.moveToFirst()) {
                int idx = c.getColumnIndex(OpenableColumns.DISPLAY_NAME);
                if (idx >= 0) {
                    String n = c.getString(idx);
                    if (n != null && !n.isEmpty()) return n;
                }
            }
        } catch (Exception ignored) {
        }
        String p = uri.getLastPathSegment();
        return p == null ? "audio" : p;
    }

    @Override
    protected void onActivityResult(int req, int res, Intent data) {
        super.onActivityResult(req, res, data);
        if (req != REQ_PICK_AUDIO || res != RESULT_OK || data == null || data.getData() == null) {
            return;
        }
        final Uri uri = data.getData();
        setStatus("正在导入音频…");
        new Thread(() -> {
            File dst = null;
            try {
                String name = displayName(uri);
                dst = new File(outDir(), "src_"
                        + new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(new Date())
                        + ".wav");
                final File dstF = dst;
                double sec = AudioImport.importToWav(this, uri, dst, importProgress());
                final String nm = name;
                final double s2 = sec;
                ui.post(() -> {
                    srcWav = dstF;
                    srcName = nm;
                    rvcWav = null;
                    silkFile = null;
                    if (state == ST_READY || state == ST_CONVERTING) setState(ST_IDLE);
                    setStatus("已导入：" + nm);
                    tvResult.setText(String.format(Locale.US,
                            "输入：%s\n时长 %.2fs · %.0f KB\n%s",
                            nm, s2, dstF.length() / 1024.0, dstF.getAbsolutePath()));
                    updateButtons();
                    toast(String.format(Locale.US, "已导入 %.1fs", s2));
                    if (autoSend) convertFile();
                });
            } catch (Throwable t) {
                log("导入音频失败：" + t);
                final String msg = String.valueOf(t);
                ui.post(() -> {
                    setStatus("导入音频失败：" + msg);
                    toast("导入音频失败：" + msg);
                });
            }
        }, "rec-import").start();
    }

    /** 按路径导入（命令行入口走这里；文件选择器走 onActivityResult）。 */
    private void importFromPath(File f, boolean auto) {
        if (f == null || !f.isFile()) {
            setStatus("找不到音频文件：" + f);
            return;
        }
        final boolean autoRun = auto;
        setStatus("正在导入 " + f.getName() + " …");
        new Thread(() -> {
            try {
                File dst = new File(outDir(), "src_"
                        + new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(new Date())
                        + ".wav");
                final File dstF = dst;
                double sec = AudioImport.importToWav(this, f.getAbsolutePath(), dst, importProgress());
                final double s2 = sec;
                ui.post(() -> {
                    srcWav = dstF;
                    srcName = f.getName();
                    rvcWav = null;
                    silkFile = null;
                    if (state == ST_READY || state == ST_CONVERTING) setState(ST_IDLE);
                    setStatus("已导入：" + f.getName());
                    tvResult.setText(String.format(Locale.US,
                            "输入：%s\n时长 %.2fs · %.0f KB\n%s",
                            f.getName(), s2, dstF.length() / 1024.0, dstF.getAbsolutePath()));
                    updateButtons();
                    if (autoRun) convertFile();
                });
            } catch (Throwable t) {
                log("导入音频失败：" + t);
                final String msg = String.valueOf(t);
                ui.post(() -> setStatus("导入音频失败：" + msg));
            }
        }, "rec-import-path").start();
    }

    /** 对已选中的音频文件跑一次离线变声。 */
    private void convertFile() {
        if (state == ST_RECORDING || state == ST_CONVERTING) {
            toast("正在忙，稍后再试");
            return;
        }
        if (srcWav == null || !srcWav.isFile()) {
            toast("先「选择音频文件」");
            return;
        }
        startConvert(srcWav);
    }

    // ================================================================== 转换完直接发 QQ
    /**
     * 自动发送链：WAV → SILK → 覆盖最近一条 QQ 语音草稿 → 提示去 QQ 按发送。
     *
     * <p>为什么只能做到"草稿"：QQ 的语音是写在它自己的 ptt 目录里的 slk 文件，
     * 我们能（凭 root/Shizuku）替它写好这条文件，但"点发送"那一下在 QQ 进程里，
     * 没有公开接口；把草稿准备好、跳过去，剩下就一下。
     */
    private void sendResult(final File wav) {
        if (wav == null || !wav.isFile()) return;
        setStatus("自动发送：正在编码 SILK @" + (bitrate / 1000) + "k …");
        new Thread(() -> {
            try {
                if (!QqDraftInjector.ready()) {
                    ui.post(() -> {
                        setStatus("变声完成，但写 QQ 草稿需要 root 或 Shizuku");
                        new MaterialAlertDialogBuilder(this)
                                .setTitle("发不出去：没有后端")
                                .setMessage(QqDraftInjector.describe()
                                        + "\n\n· 有 root：在 KernelSU / Magisk 里给本 App 授权\n"
                                        + "· 没 root：装 Shizuku 并授权\n\n"
                                        + "变声结果已经存好了，授权后点「写 QQ 草稿」即可。")
                                .setPositiveButton("重新探测", (d, w) -> {
                                    QqDraftInjector.reset();
                                    probeBackendAsync();
                                })
                                .setNegativeButton("好", null)
                                .show();
                    });
                    return;
                }
                File slk = new File(wav.getParentFile(),
                        wav.getName().replace(".wav", "_" + (bitrate / 1000) + "k.slk"));
                double sec = Silk.encodeFile(this, wav, slk, bitrate, null);
                int frames = Silk.countFrames(slk);
                List<Injector.Draft> drafts = QqDraftInjector.candidates(this);
                final File f = slk;
                ui.post(() -> {
                    silkFile = f;
                    setStatus("SILK 完成，准备写 QQ 草稿");
                    tvResult.setText(String.format(Locale.US,
                            "SILK：%s\n%.2fs · %d 帧 · %.1f KB",
                            f.getName(), sec, frames, f.length() / 1024.0));
                    updateButtons();
                    if (drafts.isEmpty()) {
                        setStatus("没找到 QQ 语音草稿");
                        new MaterialAlertDialogBuilder(this)
                                .setTitle("没有可覆盖的草稿")
                                .setMessage("目录：" + QqDraftInjector.qqRoot(this)
                                        + "\n\n先在 QQ 里用语音键留一条草稿（别发送），"
                                        + "回来再点「转换」或「写 QQ 草稿」。")
                                .setPositiveButton("好", null)
                                .show();
                        return;
                    }
                    injectAndOffer(f, drafts.get(0));
                });
            } catch (Throwable t) {
                log("自动发送失败：" + t);
                final String msg = String.valueOf(t);
                ui.post(() -> {
                    setStatus("自动发送失败：" + msg);
                    toast("自动发送失败：" + msg);
                });
            }
        }, "rec-autosend").start();
    }

    /** 写进草稿，成功后直接弹「去 QQ 发送」。 */
    private void injectAndOffer(File silk, Injector.Draft target) {
        setStatus("正在写入 " + target.name() + " …");
        QqDraftInjector.inject(this, silk, target, (ok, detail) -> {
            setStatus(ok ? "已写进 QQ 草稿" : "写入失败");
            log(detail.replace('\n', ' '));
            MaterialAlertDialogBuilder b = new MaterialAlertDialogBuilder(this)
                    .setTitle(ok ? "已写进 QQ 语音草稿" : "写入失败")
                    .setMessage(detail + (ok ? "\n\n接下来：点「去 QQ」→ 进聊天 → 按发送。" : ""));
            if (ok) {
                b.setPositiveButton("去 QQ", (d, w) -> openQq())
                        .setNegativeButton("留在这", null);
            } else {
                b.setPositiveButton("好", null);
            }
            b.show();
        });
    }

    /** 跳到 QQ（不指定聊天，用户自己点进对话）。 */
    private void openQq() {
        try {
            Intent i = getPackageManager().getLaunchIntentForPackage(QQ_PKG);
            if (i == null) {
                toast("没找到 QQ");
                return;
            }
            i.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
            startActivity(i);
        } catch (Exception e) {
            toast("打开 QQ 失败：" + e);
        }
    }

    // ================================================================== SILK
    private void onMakeSilk() {
        if (rvcWav == null || !rvcWav.isFile()) {
            toast("还没有变声结果");
            return;
        }
        final File wav = rvcWav;
        setStatus("正在编码 SILK @" + (bitrate / 1000) + "k…");
        new Thread(() -> {
            try {
                File out = new File(wav.getParentFile(),
                        wav.getName().replace(".wav", "_" + (bitrate / 1000) + "k.slk"));
                double sec = Silk.encodeFile(this, wav, out, bitrate, p -> {
                    if (p % 25 == 0) {
                        ui.post(() -> setStatus("编码 SILK " + p + "%"));
                    }
                });
                int frames = Silk.countFrames(out);
                final File f = out;
                ui.post(() -> {
                    silkFile = f;
                    setStatus("SILK 完成");
                    tvResult.setText(String.format(Locale.US,
                            "SILK：%s\n%.2fs · %d 帧 · %.1f KB\n%s",
                            f.getName(), sec, frames, f.length() / 1024.0,
                            f.getAbsolutePath()));
                    updateButtons();
                    toast("已生成 SILK");
                });
            } catch (Throwable t) {
                log("SILK 失败：" + t);
                ui.post(() -> setStatus("SILK 失败：" + t));
            }
        }, "rec-silk").start();
    }

    // ================================================================== QQ 草稿
    private void onInjectDraft() {
        File src = silkFile;
        if (src == null) {
            toast("先生成 SILK（点「转 SILK」）");
            return;
        }
        setStatus("正在查找 QQ 草稿…");
        new Thread(() -> {
            try {
                if (!QqDraftInjector.ready()) {
                    ui.post(() -> {
                        setStatus("写 QQ 草稿需要 root 或 Shizuku");
                        new MaterialAlertDialogBuilder(this)
                                .setTitle("没有可用后端")
                                .setMessage(QqDraftInjector.describe()
                                        + "\n\n· 有 root：在 KernelSU / Magisk 里给本 App 授权，再点一次\n"
                                        + "· 没 root：装上 Shizuku，宿主集成其 SDK 后即可走 shell 身份")
                                .setPositiveButton("重新探测", (d, w) -> {
                                    QqDraftInjector.reset();
                                    probeBackendAsync();
                                })
                                .setNegativeButton("好", null)
                                .show();
                    });
                    return;
                }
                List<Injector.Draft> drafts = QqDraftInjector.candidates(this);
                ui.post(() -> {
                    if (drafts.isEmpty()) {
                        setStatus("没找到 QQ 语音草稿");
                        new MaterialAlertDialogBuilder(this)
                                .setTitle("没找到草稿")
                                .setMessage("目录：" + QqDraftInjector.qqRoot(this)
                                        + "\n\n先用 QQ 语音键留一条草稿（别发送），再回来点一次。")
                                .setPositiveButton("好", null)
                                .show();
                        return;
                    }
                    String[] names = new String[drafts.size()];
                    for (int i = 0; i < drafts.size(); i++) {
                        Injector.Draft d = drafts.get(i);
                        names[i] = (i == 0 ? "★ " : "") + d.name() + "  " + (d.size / 1024) + " KB";
                    }
                    new MaterialAlertDialogBuilder(this)
                            .setTitle("覆盖哪条草稿（后端 " + QqDraftInjector.backendName() + "）")
                            .setItems(names, (dlg, which) -> doInject(src, drafts.get(which)))
                            .setNegativeButton("取消", null)
                            .show();
                });
            } catch (Throwable t) {
                log("查草稿失败：" + t);
                ui.post(() -> setStatus("查草稿失败：" + t));
            }
        }, "rec-draft-list").start();
    }

    private void doInject(File silk, Injector.Draft target) {
        injectAndOffer(silk, target);
    }

    // ================================================================== 试听（手动）
    private void onPreview() {
        if (rvcWav == null || !rvcWav.isFile()) {
            toast("还没有变声结果");
            return;
        }
        // 注意：这是用户手动点的试听，属于显式操作；
        // 录音与变声流程本身全程不创建任何 AudioTrack，不出声。
        try {
            final android.media.MediaPlayer mp = new android.media.MediaPlayer();
            mp.setDataSource(rvcWav.getAbsolutePath());
            mp.prepare();
            mp.start();
            mp.setOnCompletionListener(m -> {
                try {
                    m.release();
                } catch (Exception ignored) {
                }
            });
            toast("试听中（手动）");
        } catch (Exception e) {
            toast("试听失败：" + e);
            log("试听失败：" + e);
        }
    }

    // ================================================================== 杂项
    private File outDir() {
        File d = new File("/sdcard/Download");
        if (d.isDirectory() && d.canWrite()) {
            return d;
        }
        File e = getExternalFilesDir(null);
        if (e != null) {
            //noinspection ResultOfMethodCallIgnored
            e.mkdirs();
            return e;
        }
        return getFilesDir();
    }

    @Override
    public void onRequestPermissionsResult(int req, String[] perms, int[] res) {
        super.onRequestPermissionsResult(req, perms, res);
        if (req == REQ_MIC) {
            boolean ok = false;
            for (int i = 0; i < perms.length; i++) {
                if (Manifest.permission.RECORD_AUDIO.equals(perms[i])
                        && res[i] == PackageManager.PERMISSION_GRANTED) {
                    ok = true;
                }
            }
            if (ok) {
                startRecording();
            } else {
                toast("需要录音权限");
                setStatus("没有录音权限");
            }
        }
    }
}
