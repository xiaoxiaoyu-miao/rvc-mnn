package com.rvc.mnn;

import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.media.projection.MediaProjectionManager;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.Settings;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;

import androidx.appcompat.app.AppCompatActivity;

import com.google.android.material.button.MaterialButton;
import com.google.android.material.checkbox.MaterialCheckBox;
import com.google.android.material.slider.Slider;
import com.google.android.material.textfield.TextInputEditText;
import com.google.android.material.textfield.TextInputLayout;
import android.widget.CheckBox;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.SeekBar;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import java.io.File;
import java.io.FileOutputStream;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/**
 * RVC-MNN：hubert + 合成器 走 MNN，fcpe 音高提取走 ORT。
 *
 * <p>功能：
 * <ul>
 *   <li>音色预设（男→女 / 女→男 / 同性转换）：只改 f0 倍率，对应 RVC 的 f0_up_key；</li>
 *   <li>引擎预设（极速/均衡/高质量）：CPU/Vulkan × 计算精度；</li>
 *   <li>模型目录扫描 + 四个权重文件手动选择；</li>
 *   <li>输出增益 / 静音门限 / 反馈抑制 / 免提-听筒路由；</li>
 *   <li>WAV 文件选择、转换、试听；</li>
 *   <li>实时变声跑在 <b>前台服务</b> 里，退后台/息屏持续工作，可调低中后台保活；</li>
 *   <li>所有配置持久化（SharedPreferences），日志可清空/导出。</li>
 * </ul>
 */
public class MainActivity extends AppCompatActivity {
    private final Config cfg = new Config();
    private RvcEngine engine;

    // ---- UI ----
    private TextView tvStatus, tvLog, tvPitch, tvGain, tvGate, tvEngine, tvFb;
    private EditText etModels, etIn, etOut;
    private Spinner spVoice, spEngine, spHubert, spSynth, spFcpe, spEmb;
    private Slider sbPitch, sbGain, sbGate, sbFbDb;
    private CheckBox cbSaveRt, cbCapture;
    private Spinner spOut;
    private Button btnRec;
    private Button btnGo, btnRT, btnPlay;

    // ---- 运行时 ----
    private volatile boolean running = false;
    /** 内置模型解压完成前，转换按钮保持禁用。 */
    private volatile boolean modelsReady = false;

    private static final int REQ_PICK_WAV = 101;
    /** MediaProjection 屏幕录制授权请求码。 */
    private static final int REQ_PROJ = 102;
    /** 从「开始录屏」触发的权限请求：授权通过后自动继续录屏流程。 */
    private volatile boolean pendingRecordAfterPerm = false;
    /** 录屏前的输出路由，停止录屏后恢复（-1 表示没有可恢复的）。 */
    private int preRecOutputMode = -1;
    /** 实时变声是随录屏一起启动的：停止录屏时应该一并停掉，而不是留着继续外放。 */
    private volatile boolean rtStartedForRec = false;
    /** 录屏在服务里异步启停，按钮文案靠定时轮询跟上（前后台切换时自动停）。 */
    private final android.os.Handler uiHandler = new android.os.Handler(android.os.Looper.getMainLooper());
    private final Runnable recUiTick = new Runnable() {
        @Override public void run() {
            updateRecUi();
            uiHandler.postDelayed(this, 800);
        }
    };
    /** 程序化改下拉时置 true，避免回调重复落盘/刷日志。 */
    private boolean suppressOutCb = false;
    /** 滑块联动下拉时置 true，避免回调把刚拖的值又改回预设值。 */
    private boolean syncingVoice = false;

    // ================================================================== UI
    @Override
    protected void onCreate(Bundle b) {
        super.onCreate(b);
        cfg.load(this);
        engine = RvcEngine.get(this);

        RvcLog.setSink(s -> runOnUiThread(() -> {
            if (tvLog != null) tvLog.append(s + "\n");
        }));

        ScrollView sv = new ScrollView(this);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        int pad = (int) (12 * getResources().getDisplayMetrics().density);
        root.setPadding(pad, pad, pad, pad);
        sv.addView(root);
        setContentView(sv);

        // ---------- 状态 ----------
        tvStatus = label("");
        tvStatus.setTextSize(12);
        root.addView(tvStatus);

        // ---------- 音色预设 ----------
        header(root, "① 音色预设（变调）");
        spVoice = new Spinner(this);
        root.addView(spVoice);
        sbPitch = seekRow(root, -24, 24, 0.1f, 0, v -> {
            cfg.pitchSemi = v;
            cfg.voicePreset = nearestVoicePreset(v);
            tvPitch.setText(pitchLabel());
            syncVoiceSpinner(cfg.voicePreset);   // 滑块变了，下拉也跟着走到最近的预设
            cfg.save(MainActivity.this);
        });
        tvPitch = label("");
        root.addView(tvPitch);
        tvGain = label("");
        sbGain = seekRow(root, 0.25f, 3.0f, 0.01f, 0, v -> {
            cfg.gain = v;
            tvGain.setText(String.format("输出增益 %.2fx", v));
            cfg.save(MainActivity.this);
        });
        root.addView(tvGain);
        tvGate = label("");
        sbGate = seekRow(root, 0f, 0.02f, 0.0001f, 0, v -> {
            cfg.gate = v;
            tvGate.setText(String.format("静音门限 RMS %.4f（实时用，低于此值输出静音）", v));
            cfg.save(MainActivity.this);
        });
        root.addView(tvGate);

        // ---------- 引擎 / 模型 ----------
        header(root, "② 引擎与模型配置");
        spEngine = new Spinner(this);
        root.addView(spEngine);
        tvEngine = label("");
        root.addView(tvEngine);

        LinearLayout rowDir = new LinearLayout(this);
        rowDir.setOrientation(LinearLayout.HORIZONTAL);
        etModels = new TextInputEditText(this);
        etModels.setSingleLine(true);
        rowDir.addView(wrapInput(etModels, "模型目录"),
                new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f));
        Button btnScan = new MaterialButton(this);
        btnScan.setText("重新扫描");
        btnScan.setOnClickListener(v -> {
            cfg.modelDir = etModels.getText().toString().trim();
            cfg.save(this);
            refreshModels();
            invalidateNative();
            updateStatus();
            log("已扫描模型目录: " + cfg.modelDir);
        });
        rowDir.addView(btnScan);
        root.addView(label("模型目录"));
        root.addView(rowDir);

        spHubert = spinnerRow(root, "hubert 特征模型 (.mnn)");
        spSynth = spinnerRow(root, "合成器模型 (.mnn)");
        spFcpe = spinnerRow(root, "f0 提取模型 (.onnx)");
        spEmb = spinnerRow(root, "音高编码表 (.bin)");

        LinearLayout rowCfg = new LinearLayout(this);
        rowCfg.setOrientation(LinearLayout.HORIZONTAL);
        Button btnExport = new MaterialButton(this);
        btnExport.setText("导出配置");
        btnExport.setOnClickListener(v -> exportConfig());
        Button btnReset = new MaterialButton(this);
        btnReset.setText("重置默认");
        btnReset.setOnClickListener(v -> resetConfig());
        rowCfg.addView(btnExport, new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f));
        rowCfg.addView(btnReset, new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f));
        root.addView(rowCfg);

        LinearLayout rowModel = new LinearLayout(this);
        rowModel.setOrientation(LinearLayout.HORIZONTAL);
        MaterialButton btnReinstall = new MaterialButton(this);
        btnReinstall.setText("重解压内置模型");
        btnReinstall.setOnClickListener(v -> reinstallBundled());
        MaterialButton btnVerifyModel = new MaterialButton(this);
        btnVerifyModel.setText("校验模型 SHA-256");
        btnVerifyModel.setOnClickListener(v -> verifyModels());
        rowModel.addView(btnReinstall, new LinearLayout.LayoutParams(0,
                LinearLayout.LayoutParams.WRAP_CONTENT, 1f));
        rowModel.addView(btnVerifyModel, new LinearLayout.LayoutParams(0,
                LinearLayout.LayoutParams.WRAP_CONTENT, 1f));
        root.addView(rowModel);

        // ---------- 文件转换 ----------
        header(root, "③ WAV 文件转换");
        root.addView(label("输入 WAV"));
        etIn = new TextInputEditText(this);
        etIn.setSingleLine(true);
        root.addView(wrapInput(etIn, "输入 WAV 路径"));
        LinearLayout rowIn = new LinearLayout(this);
        rowIn.setOrientation(LinearLayout.HORIZONTAL);
        Button btnPick = new MaterialButton(this);
        btnPick.setText("选择输入音频");
        btnPick.setOnClickListener(v -> pickWav());
        Button btnAutoOut = new MaterialButton(this);
        btnAutoOut.setText("自动输出名");
        btnAutoOut.setOnClickListener(v -> {
            String in = etIn.getText().toString().trim();
            String name = "out_" + System.currentTimeMillis() + ".wav";
            if (!in.isEmpty()) {
                String base = new File(in).getName();
                int dot = base.lastIndexOf('.');
                if (dot > 0) base = base.substring(0, dot);
                name = base + "_rvc_" + cfg.voiceTag() + ".wav";
            }
            etOut.setText("/sdcard/Download/" + name);
        });
        rowIn.addView(btnPick, new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f));
        rowIn.addView(btnAutoOut, new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f));
        root.addView(rowIn);
        root.addView(label("输出 WAV"));
        etOut = new TextInputEditText(this);
        etOut.setSingleLine(true);
        root.addView(wrapInput(etOut, "输出 WAV 路径"));

        LinearLayout rowGo = new LinearLayout(this);
        rowGo.setOrientation(LinearLayout.HORIZONTAL);
        btnGo = new MaterialButton(this);
        btnGo.setText("开始转换");
        btnGo.setOnClickListener(v -> {
            cfg.inWav = etIn.getText().toString().trim();
            cfg.outWav = etOut.getText().toString().trim();
            cfg.save(this);
            if (!running) new Thread(this::runConvert).start();
        });
        btnPlay = new MaterialButton(this);
        btnPlay.setText("试听");
        btnPlay.setOnClickListener(v -> play(cfg.outWav));
        rowGo.addView(btnGo, new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f));
        rowGo.addView(btnPlay, new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f));
        root.addView(rowGo);

        // ---------- 实时 ----------
        header(root, "④ 实时变声（前台服务，可后台/息屏）");
        tvFb = label("");
        sbFbDb = seekRow(root, 0f, 60f, 1f, 0, v -> {
            cfg.feedbackGateDb = v;
            tvFb.setText(fbLabel());
            cfg.save(this);
        });
        root.addView(tvFb);
        // 输出路由：外放 / 听筒 / 不外放
        spOut = spinnerRow(root, "实时输出路由");
        ArrayAdapter<String> adOut = new ArrayAdapter<>(this,
                android.R.layout.simple_spinner_item, Config.OUTPUT_NAMES);
        adOut.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        spOut.setAdapter(adOut);
        spOut.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override public void onItemSelected(AdapterView<?> p, android.view.View v, int pos, long id) {
                if (suppressOutCb) return;
                cfg.outputMode = Math.max(0, Math.min(Config.OUTPUT_NAMES.length - 1, pos));
                cfg.save(MainActivity.this);
                log("实时输出 → " + Config.OUTPUT_NAMES[cfg.outputMode]
                        + "：" + Config.OUTPUT_HINTS[cfg.outputMode]);
            }
            @Override public void onNothingSelected(AdapterView<?> p) {}
        });

        cbCapture = new MaterialCheckBox(this);
        cbCapture.setText("让系统录屏能录到（切媒体用途，会关闭 AEC）");
        cbCapture.setOnCheckedChangeListener((v, on) -> {
            cfg.captureFriendly = on;
            cfg.save(this);
            log(on ? "已允许系统录屏捕获本 App 的声音（实时通话场景请关掉）"
                   : "已关闭系统录屏捕获");
        });
        root.addView(cbCapture);

        btnRec = new MaterialButton(this);
        btnRec.setText("开始录屏（App 内，mp4 带变声音轨）");
        btnRec.setOnClickListener(v -> toggleRecord());
        root.addView(btnRec);
        btnRT = new MaterialButton(this);
        btnRT.setText("实时变声 (开)");
        btnRT.setOnClickListener(v -> {
            if (RealtimeService.running) stopRealTime(); else startRealTime();
        });
        root.addView(btnRT);

        LinearLayout rowKeep = new LinearLayout(this);
        rowKeep.setOrientation(LinearLayout.HORIZONTAL);
        Button btnBattery = new MaterialButton(this);
        btnBattery.setText("关电池优化");
        btnBattery.setOnClickListener(v -> requestBatteryExempt());
        Button btnKeepInfo = new MaterialButton(this);
        btnKeepInfo.setText("保活说明");
        btnKeepInfo.setOnClickListener(v -> showKeepaliveTips());
        rowKeep.addView(btnBattery, new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f));
        rowKeep.addView(btnKeepInfo, new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f));
        root.addView(rowKeep);

        cbSaveRt = new MaterialCheckBox(this);
        cbSaveRt.setText("实时变声时把输出录成 wav");
        cbSaveRt.setOnCheckedChangeListener((v, on) -> {
            cfg.saveRealtime = on;
            cfg.save(this);
        });
        root.addView(cbSaveRt);

        // ---------- 录音模式（离线） ----------
        header(root, "★ 录音模式（录完即停 · 不实时 · 不外放）");
        root.addView(label("按「开始录音」→ 说完整段 → 按「停止并变声」。"
                + "录音期间不做推理、不建 AudioTrack，手机全程不出声；"
                + "整段说完才跑一次离线变声，产出可直接转 SILK 并写进 QQ 语音草稿。"));
        MaterialButton btnRecMode = new MaterialButton(this);
        btnRecMode.setText("进入录音模式");
        btnRecMode.setOnClickListener(v -> startActivity(new Intent(this, RecordActivity.class)));
        root.addView(btnRecMode);

        // ---------- 日志 ----------
        header(root, "⑤ 日志");
        LinearLayout rowLog = new LinearLayout(this);
        rowLog.setOrientation(LinearLayout.HORIZONTAL);
        Button btnClear = new MaterialButton(this);
        btnClear.setText("清空日志");
        btnClear.setOnClickListener(v -> clearLog());
        Button btnLogExport = new MaterialButton(this);
        btnLogExport.setText("导出日志");
        btnLogExport.setOnClickListener(v -> exportLog());
        rowLog.addView(btnClear, new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f));
        rowLog.addView(btnLogExport, new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f));
        root.addView(rowLog);

        tvLog = new TextView(this);
        tvLog.setTextSize(11);
        tvLog.setTextIsSelectable(true);
        ScrollView logSv = new ScrollView(this);
        logSv.addView(tvLog);
        root.addView(logSv, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f));

        // ---------- 恢复配置到控件 ----------
        etModels.setText(cfg.modelDir);
        etIn.setText(cfg.inWav);
        etOut.setText(cfg.outWav);
        cbSaveRt.setChecked(cfg.saveRealtime);
        setSliderValue(sbFbDb, cfg.feedbackGateDb);
        spOut.setSelection(Math.max(0, Math.min(Config.OUTPUT_NAMES.length - 1, cfg.outputMode)));
        cbCapture.setChecked(cfg.captureFriendly);
        tvFb.setText(fbLabel());
        updateRecUi();
        restorePresetUi();
        refreshModels();
        updateStatus();

        // 模型准备：首次启动会把 assets 里的模型解压到私有目录（几秒），期间先别让转换跑
        btnGo.setEnabled(false);
        new Thread(() -> {
            boolean ok = ModelBootstrap.prepare(this, cfg);
            runOnUiThread(() -> {
                modelsReady = true;
                etModels.setText(cfg.modelDir);
                refreshModels();
                updateStatus();
                syncRtUi();
                btnGo.setEnabled(!RealtimeService.running);
                log(ok ? ("模型就绪: " + cfg.modelDir)
                        : "模型准备失败：内置 assets 缺失或目录不可写");
                if (ok && !RealtimeService.running) {
                    // 主界面也顺手热一遍：③ WAV 转换 与 录音模式 共用同一个引擎单例
                    new Thread(() -> {
                        try {
                            if (engine.ensure(cfg)) {
                                EngineWarmup.Result r = EngineWarmup.warmIfNeeded(engine, cfg, s -> log(s), 2);
                                log(r.summary());
                            }
                        } catch (Throwable t) {
                            log("预热失败：" + t);
                        }
                    }, "engine-warm").start();
                }
            });
        }, "model-prepare").start();
        syncRtUi();

        // intent 入口（auto / rt / stoprec）；Activity 已存在时走 onNewIntent
        handleIntentExtras(getIntent());

        if (Build.VERSION.SDK_INT >= 30 && !Environment.isExternalStorageManager()) {
            try {
                startActivity(new Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION,
                        Uri.parse("package:" + getPackageName())));
            } catch (Exception e) {
                startActivity(new Intent(Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION));
            }
        }
        log("RVC-MNN 启动：音色=" + Config.VOICE_NAMES[cfg.voicePreset]
                + "  引擎=" + Config.ENGINE_NAMES[cfg.enginePreset]);
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (btnRT != null) syncRtUi();
        updateRecUi();
        uiHandler.removeCallbacks(recUiTick);
        uiHandler.post(recUiTick);
    }

    @Override
    protected void onPause() {
        super.onPause();
        uiHandler.removeCallbacks(recUiTick);
    }

    /** launchMode=singleTop：再次 am start 时复用实例，必须自己处理新 intent。 */
    @Override
    protected void onNewIntent(Intent intent) {
        super.onNewIntent(intent);
        setIntent(intent);
        handleIntentExtras(intent);
    }

    /** 命令行入口：--ez auto / rt / stoprec。 */
    private void handleIntentExtras(Intent it) {
        if (it == null) return;
        android.os.Handler h = new android.os.Handler(getMainLooper());
        if (it.hasExtra("in")) {
            String in = it.getStringExtra("in");
            if (in != null && !in.isEmpty()) etIn.setText(in);
        }
        if (it.hasExtra("out")) {
            String out = it.getStringExtra("out");
            if (out != null && !out.isEmpty()) etOut.setText(out);
        }
        if (it.getBooleanExtra("auto", false)) {
            h.postDelayed(() -> { if (!running) new Thread(this::runConvert).start(); }, 300);
        }
        if (it.hasExtra("engine")) {
            int e = it.getIntExtra("engine", -1);
            if (e >= 0 && e < Config.ENGINE_CFG.length) {
                h.postDelayed(() -> applyEngine(e), 400);
            }
        }
        if (it.getBooleanExtra("rt", false)) {
            h.postDelayed(this::startRealTime, 600);
        }
        if (it.getBooleanExtra("stoprec", false)) {
            h.postDelayed(() -> {
                if (ScreenRecorder.isRecording()) {
                    toggleRecord();
                } else {
                    log("stoprec: 当前没有在录屏");
                }
            }, 800);
        }
    }

    // ================================================================== 控件构建小工具
    /** M3 bodyMedium：说明文字。 */
    private TextView label(String s) {
        TextView t = new TextView(this);
        t.setText(s);
        t.setTextAppearance(com.google.android.material.R.style.TextAppearance_Material3_BodyMedium);
        t.setAlpha(0.75f);
        return t;
    }

    /** M3 titleLarge：分区标题，前面加一条主色竖条做视觉锚点。 */
    private void header(LinearLayout root, String s) {
        LinearLayout row = new LinearLayout(this);
        row.setOrientation(LinearLayout.HORIZONTAL);
        int d = (int) getResources().getDisplayMetrics().density;
        android.view.View bar = new android.view.View(this);
        android.util.TypedValue tv = new android.util.TypedValue();
        getTheme().resolveAttribute(com.google.android.material.R.attr.colorPrimary, tv, true);
        bar.setBackgroundColor(tv.data);
        LinearLayout.LayoutParams barLp = new LinearLayout.LayoutParams(d * 3, d * 18);
        barLp.setMargins(0, d * 18, d * 8, d * 4);
        row.addView(bar, barLp);

        TextView t = new TextView(this);
        t.setText(s);
        t.setTextAppearance(com.google.android.material.R.style.TextAppearance_Material3_TitleMedium);
        t.setPadding(0, d * 12, 0, d * 6);
        row.addView(t);
        root.addView(row);
    }

    private interface Fn {
        void run(float v);
    }

    /**
     * M3 滑块：{@link Slider} 直接用真实数值（不再是 SeekBar 的整数 index），
     * 省掉了原来「值 ↔ index」的来回换算。thumb 上带数值气泡。
     */
    private Slider seekRow(LinearLayout root, float min, float max, float step, float val, Fn onChange) {
        Slider sl = new Slider(this);
        sl.setValueFrom(min);
        sl.setValueTo(max);
        sl.setStepSize(step);
        sl.setLabelFormatter(v -> String.format(java.util.Locale.US, "%.2f", v));
        sl.setValue(snap(min, max, step, val));
        sl.addOnChangeListener((slider, value, fromUser) -> onChange.run(value));
        root.addView(sl);
        return sl;
    }

    /** 把任意值吸附到 [min,max] 内的合法 step 格点上，避免 Slider 抛非法值。 */
    private static float snap(float min, float max, float step, float v) {
        if (v < min) v = min;
        if (v > max) v = max;
        if (step > 0f) {
            v = min + Math.round((v - min) / step) * step;
            if (v < min) v = min;
            if (v > max) v = max;
        }
        return v;
    }

    /** 程序化设置滑块值（不触发非法值异常）。 */
    private static void setSliderValue(Slider sl, float v) {
        if (sl == null) return;
        sl.setValue(snap(sl.getValueFrom(), sl.getValueTo(), sl.getStepSize(), v));
    }

    /** 给 EditText 套一层 M3 输入框（浮动标签 + 圆角描边）。 */
    private TextInputLayout wrapInput(android.widget.EditText et, String hint) {
        TextInputLayout til = new TextInputLayout(this);
        til.setHint(hint);
        til.setBoxBackgroundMode(TextInputLayout.BOX_BACKGROUND_OUTLINE);
        til.addView(et, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT));
        return til;
    }

    private Spinner spinnerRow(LinearLayout root, String title) {
        root.addView(label(title));
        Spinner sp = new Spinner(this);
        root.addView(sp);
        return sp;
    }

    private void fillSpinner(Spinner sp, List<String> items, String selected) {
        List<String> list = new ArrayList<>(items);
        if (selected != null && !selected.isEmpty() && !list.contains(selected)) list.add(0, selected);
        ArrayAdapter<String> ad = new ArrayAdapter<>(this, android.R.layout.simple_spinner_item, list);
        ad.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        sp.setAdapter(ad);
        int idx = list.indexOf(selected);
        sp.setSelection(Math.max(0, idx));
    }

    // ================================================================== 预设与配置
    private String pitchLabel() {
        return String.format("变调 %+.1f 半音  (f0 倍率 %.3f)", cfg.pitchSemi, cfg.pitchRatio());
    }

    private String fbLabel() {
        if (cfg.feedbackGateDb <= 0f) return "反馈抑制：关闭（回环会一直挂着，必须戴耳机）";
        return String.format("反馈抑制门限 %.0f dB（逐帧门控，切断“扬声器→麦克风→再合成”的循环尾巴）",
                cfg.feedbackGateDb);
    }

    private int nearestVoicePreset(float semi) {
        int best = 0;
        float bd = Float.MAX_VALUE;
        for (int i = 0; i < Config.VOICE_SEMI.length; i++) {
            float d = Math.abs(Config.VOICE_SEMI[i] - semi);
            if (d < bd) { bd = d; best = i; }
        }
        return best;
    }

    /** 让音色下拉跟随滑块，且不触发它自己的回调。 */
    private void syncVoiceSpinner(int pos) {
        if (spVoice == null || spVoice.getAdapter() == null) return;
        if (spVoice.getSelectedItemPosition() == pos) return;
        syncingVoice = true;
        try { spVoice.setSelection(pos); } finally { syncingVoice = false; }
    }

    private void restorePresetUi() {
        // 音色预设
        ArrayAdapter<String> ad = new ArrayAdapter<>(this, android.R.layout.simple_spinner_item, Config.VOICE_NAMES);
        ad.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        spVoice.setAdapter(ad);
        spVoice.setSelection(Math.max(0, Math.min(Config.VOICE_NAMES.length - 1, cfg.voicePreset)));
        spVoice.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override public void onItemSelected(AdapterView<?> p, android.view.View v, int pos, long id) {
                if (syncingVoice) return;
                cfg.voicePreset = pos;
                cfg.pitchSemi = Config.VOICE_SEMI[pos];
                setSliderValue(sbPitch, cfg.pitchSemi);
                tvPitch.setText(pitchLabel());
                cfg.save(MainActivity.this);
                updateStatus();
                log("音色预设 → " + Config.VOICE_NAMES[pos]
                        + String.format("  (%+.1f 半音)", Config.VOICE_SEMI[pos]));
            }
            @Override public void onNothingSelected(AdapterView<?> p) {}
        });

        // 引擎预设
        ArrayAdapter<String> ad2 = new ArrayAdapter<>(this, android.R.layout.simple_spinner_item, Config.ENGINE_NAMES);
        ad2.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        spEngine.setAdapter(ad2);
        spEngine.setSelection(Math.max(0, Math.min(Config.ENGINE_CFG.length - 1, cfg.enginePreset)));
        spEngine.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override public void onItemSelected(AdapterView<?> p, android.view.View v, int pos, long id) {
                applyEngine(pos);
            }
            @Override public void onNothingSelected(AdapterView<?> p) {}
        });
        tvEngine.setText(engineLabel());
        setSliderValue(sbPitch, cfg.pitchSemi);
        setSliderValue(sbGain, cfg.gain);
        setSliderValue(sbGate, cfg.gate);
        tvPitch.setText(pitchLabel());
        tvGain.setText(String.format("输出增益 %.2fx", cfg.gain));
        tvGate.setText(String.format("静音门限 RMS %.4f（实时用，低于此值输出静音）", cfg.gate));
    }

    private String engineLabel() {
        int[] c = Config.ENGINE_CFG[cfg.enginePreset];
        return String.format("引擎: %s  |  后端 %s  精度 hub=%s ct=%s",
                Config.ENGINE_NAMES[cfg.enginePreset],
                c[0] != 0 ? "Vulkan" : "CPU", precName(c[1]), precName(c[2]));
    }

    private static String precName(int p) {
        return p <= 0 ? "fp16" : (p == 1 ? "normal" : "fp32");
    }

    private void applyEngine(int pos) {
        boolean hubOk = cfg.applyEnginePreset(pos);
        cfg.save(this);
        if (!hubOk) log("提示: 预设偏好的 hubert 权重 " + Config.ENGINE_HUBERT[pos]
                + " 不在模型目录，已保留当前权重 " + cfg.hubert);
        tvEngine.setText(engineLabel());
        refreshModels();
        invalidateNative();
        updateStatus();
        log("引擎预设 → " + Config.ENGINE_NAMES[cfg.enginePreset] + "（下次运行生效）");
    }

    private void resetConfig() {
        SharedPreferences p = getSharedPreferences(Config.PREFS, Context.MODE_PRIVATE);
        p.edit().clear().apply();
        cfg.modelDir = Config.DEF_MODEL_DIR;
        cfg.hubert = "hubert_fp16.mnn";
        cfg.synth = "ct_static.mnn";
        cfg.fcpe = "fcpe_audio.onnx";
        cfg.emb = "emb_pitch.bin";
        cfg.enginePreset = 1;
        cfg.voicePreset = 2;            // 同性转换（不改音高）
        cfg.pitchSemi = 0f;
        cfg.gain = 1f;
        cfg.gate = 0.0005f;
        cfg.feedbackGateDb = 25f;
        cfg.outputMode = Config.OUT_SPEAKER;
        cfg.captureFriendly = false;
        cfg.inWav = "/sdcard/Download/test10s.wav";
        cfg.outWav = "/sdcard/Download/out_mnn.wav";
        cfg.saveRealtime = false;
        cfg.save(this);
        etModels.setText(cfg.modelDir);
        etIn.setText(cfg.inWav);
        etOut.setText(cfg.outWav);
        cbSaveRt.setChecked(false);
        setSliderValue(sbFbDb, cfg.feedbackGateDb);
        spOut.setSelection(Config.OUT_SPEAKER);
        cbCapture.setChecked(false);
        tvFb.setText(fbLabel());
        spVoice.setSelection(2);
        spEngine.setSelection(cfg.enginePreset);
        setSliderValue(sbPitch, 0f);
        tvPitch.setText(pitchLabel());
        refreshModels();
        invalidateNative();
        updateStatus();
        log("已重置为默认配置");
    }

    // ================================================================== 模型扫描
    private void refreshModels() {
        cfg.modelDir = etModels.getText().toString().trim();
        File dir = new File(cfg.modelDir);
        File[] fs = dir.listFiles();
        List<String> hub = new ArrayList<>(), syn = new ArrayList<>(),
                fcp = new ArrayList<>(), emb = new ArrayList<>();
        if (fs != null) {
            for (File f : fs) {
                if (!f.isFile()) continue;
                String n = f.getName(), ln = n.toLowerCase();
                if (ln.endsWith(".mnn")) {
                    if (ln.contains("hubert")) hub.add(n);
                    else if (!ln.startsWith("sub_") && !ln.startsWith("s2_") && !ln.contains("fcpe")) syn.add(n);
                } else if (ln.endsWith(".onnx") && ln.contains("fcpe")) {
                    fcp.add(n);
                } else if (ln.endsWith(".bin") && ln.contains("emb")) {
                    emb.add(n);
                }
            }
        }
        Collections.sort(hub);
        Collections.sort(syn);
        Collections.sort(fcp);
        Collections.sort(emb);

        wireModelSpinner(spHubert, hub, cfg.hubert, v -> cfg.hubert = v);
        wireModelSpinner(spSynth, syn, cfg.synth, v -> cfg.synth = v);
        wireModelSpinner(spFcpe, fcp, cfg.fcpe, v -> cfg.fcpe = v);
        wireModelSpinner(spEmb, emb, cfg.emb, v -> cfg.emb = v);
    }

    private interface StrFn {
        void run(String v);
    }

    private void wireModelSpinner(Spinner sp, List<String> items, String selected, StrFn set) {
        final String[] chosen = {selected};
        sp.setOnItemSelectedListener(null);
        fillSpinner(sp, items, selected);
        sp.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override public void onItemSelected(AdapterView<?> p, android.view.View v, int pos, long id) {
                Object o = p.getItemAtPosition(pos);
                if (o == null) return;
                String n = o.toString();
                if (n.equals(chosen[0])) return;
                chosen[0] = n;
                set.run(n);
                cfg.save(MainActivity.this);
                invalidateNative();
                updateStatus();
                log("已切换权重 → " + n);
            }
            @Override public void onNothingSelected(AdapterView<?> p) {}
        });
    }

    private String checkMark(String title, String name) {
        File f = new File(cfg.modelDir, name);
        return "  " + (f.isFile() ? "✔" : "✘") + " " + title + ": " + name
                + (f.isFile() ? "" : "  (缺失)");
    }

    private void updateStatus() {
        StringBuilder sb = new StringBuilder();
        sb.append("模型目录: ").append(cfg.modelDir).append('\n');
        sb.append(checkMark("hubert", cfg.hubert)).append('\n');
        sb.append(checkMark("合成器", cfg.synth)).append('\n');
        sb.append(checkMark("fcpe", cfg.fcpe)).append('\n');
        sb.append(checkMark("音高编码", cfg.emb)).append('\n');
        if (RvcDsp.srOutKnown) {
            sb.append("输出采样率: ").append(RvcDsp.SR_OUT).append(" Hz")
                    .append(RvcDsp.SR_OUT == 48000 ? "（48k 模型）"
                            : RvcDsp.SR_OUT == 40000 ? "（40k 模型）" : "")
                    .append('\n');
        }
        sb.append("音色: ").append(cfg.voiceTag())
                .append(String.format(" (%+.1f 半音)  ", cfg.pitchSemi))
                .append("增益 ").append(String.format("%.2fx", cfg.gain));
        tvStatus.setText(sb);
    }

    // ================================================================== 模型加载
    private void invalidateNative() {
        // 实时服务在跑时不要释放 native session，否则正在推理的线程会崩（SIGSEGV）
        if (RealtimeService.running) {
            log("实时变声运行中：模型/引擎变更将在停止实时后生效");
            return;
        }
        EngineWarmup.invalidate();
        if (engine != null) engine.invalidate();
    }

    private boolean ensureModels() {
        return engine.ensure(cfg);
    }

    /**
     * 逐个核对模型 SHA-256（跑在后台，338MB 大约 1~2 秒）。
     * 用来确认「装着的模型就是 assets 里那份，且没被拷坏」。
     */
    private void verifyModels() {
        toast("正在校验模型哈希…");
        new Thread(() -> {
            String dirShown = cfg.modelDir;
            String[] lines = ModelBootstrap.verify(this, cfg);
            boolean allOk = true;
            StringBuilder sb = new StringBuilder();
            for (String line : lines) {
                sb.append(line).append('\n');
                if (!line.contains("✓")) allOk = false;
            }
            sb.append('\n').append("目录：").append(dirShown);
            final String msg = sb.toString();
            final boolean ok = allOk;
            runOnUiThread(() -> new com.google.android.material.dialog.MaterialAlertDialogBuilder(this)
                    .setTitle(ok ? "模型校验通过" : "模型有问题")
                    .setMessage(msg)
                    .setPositiveButton("好", null)
                    .setNeutralButton(ok ? null : "重解压", (d, w) -> reinstallBundled())
                    .show());
        }, "model-verify").start();
    }

    // ================================================================== 日志
    private void log(String s) {
        RvcLog.i("RvcApp", s);
    }

    private void clearLog() {
        tvLog.setText("");
        RvcLog.clearFile();
        log("日志已清空");
    }

    private void exportLog() {
        try {
            String p = "/sdcard/Download/rvc_app_log.txt";
            try (FileOutputStream fo = new FileOutputStream(p)) {
                fo.write(tvLog.getText().toString().getBytes("UTF-8"));
            }
            toast("日志已导出: " + p);
        } catch (Exception e) {
            toast("导出失败: " + e);
        }
    }

    private void exportConfig() {
        try {
            String p = "/sdcard/Download/rvc_config.json";
            try (FileOutputStream fo = new FileOutputStream(p)) {
                fo.write(cfg.toJson().toString(2).getBytes("UTF-8"));
            }
            toast("配置已导出: " + p);
            log("配置导出: " + p);
        } catch (Exception e) {
            toast("导出失败: " + e);
        }
    }

    private void toast(String s) { Toast.makeText(this, s, Toast.LENGTH_SHORT).show(); }

    // ================================================================== 后台保活
    private void requestBatteryExempt() {
        try {
            if (Build.VERSION.SDK_INT >= 23) {
                Intent i = new Intent(Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS,
                        Uri.parse("package:" + getPackageName()));
                startActivity(i);
                log("已请求加入电池优化白名单");
                return;
            }
        } catch (Exception ignored) {}
        try {
            startActivity(new Intent(Settings.ACTION_IGNORE_BATTERY_OPTIMIZATION_SETTINGS));
            log("请手动在列表里勾选本应用");
        } catch (Exception e) {
            log("无法打开电池优化设置，请手动到 设置→应用→本应用 关闭电池优化");
        }
    }

    private void showKeepaliveTips() {
        String tips = "保活三层：\n" +
                "1) 前台服务（本 App 已做）：通知栏常驻 → 后台/息屏可继续实时变声\n" +
                "2) 电池优化白名单：点「关电池优化」确认；或 root/ADB 执行\n" +
                "   cmd deviceidle whitelist +com.rvc.mnn\n" +
                "3) 厂商后台限制（MIUI/EMUI…）：允许「后台弹出界面/自启动」，把本应用从\n" +
                "   省电策略里设为「无限制」。\n" +
                "ADB/root 一键脚本见项目 tools/keepalive.sh（可 root 或 Shizuku 跑）。";
        android.app.AlertDialog.Builder b = new android.app.AlertDialog.Builder(this);
        b.setTitle("后台保活说明");
        b.setMessage(tips);
        b.setPositiveButton("知道了", null);
        b.show();
        log("保活说明：前台服务 + 电池白名单 + 厂商后台限制");
    }

    // ================================================================== 实时（交给服务）
    private void syncRtUi() {
        boolean on = RealtimeService.running;
        if (btnRT != null) btnRT.setText(on ? "实时变声 (关)" : "实时变声 (开)");
        if (btnGo != null) btnGo.setEnabled(!on && modelsReady);
        updateRecUi();
    }

    // ------------------------------------------------------------------ 录屏（App 内，mp4 带变声音轨）
    /** 程序化切换输出路由（同步 cfg + 下拉 + 落盘）。 */
    private void setOutputModeUi(int mode) {
        mode = Math.max(0, Math.min(Config.OUTPUT_NAMES.length - 1, mode));
        suppressOutCb = true;
        try {
            cfg.outputMode = mode;
            cfg.save(this);
            if (spOut != null) spOut.setSelection(mode);
        } finally {
            suppressOutCb = false;
        }
    }

    private void updateRecUi() {
        if (btnRec == null) return;
        boolean rec = ScreenRecorder.isRecording();
        btnRec.setText(rec ? "停止录屏" : "开始录屏（App 内，mp4 带变声音轨）");
    }

    /** 开始/停止录屏。开始时要先拿 MediaProjection 授权（系统弹窗）。 */
    private void toggleRecord() {
        if (ScreenRecorder.isRecording()) {
            if (rtStartedForRec) {
                // 实时变声是录屏时一起拉起来的，一起停掉；否则停止录制后还会继续外放
                startService(new Intent(this, RealtimeService.class)
                        .setAction(RealtimeService.ACTION_STOP));
                rtStartedForRec = false;
                log("已请求停止录屏，并同时停止实时变声（本次实时是为录屏启动的）");
            } else {
                startService(new Intent(this, RealtimeService.class)
                        .setAction(RealtimeService.ACTION_STOP_REC));
                log("已请求停止录屏");
            }
            restoreOutputAfterRec();
            updateRecUi();
            return;
        }
        if (needPerms()) {
            pendingRecordAfterPerm = true;    // 授权回来后再自动继续
            return;
        }
        // 录屏场景默认不外放：声音只进 mp4，手机不出声
        if (cfg.outputMode != Config.OUT_MUTE) {
            preRecOutputMode = cfg.outputMode;
            setOutputModeUi(Config.OUT_MUTE);
            log("已把实时输出自动切到「不外放」：录屏时手机不出声，声音只进 mp4。"
                    + "（想边录边听可以手动改回，但会外放）");
        }
        MediaProjectionManager mpm = (MediaProjectionManager)
                getSystemService(Context.MEDIA_PROJECTION_SERVICE);
        if (mpm == null) {
            log("这台设备不支持屏幕录制（没有 MediaProjectionManager）");
            return;
        }
        try {
            startActivityForResult(mpm.createScreenCaptureIntent(), REQ_PROJ);
        } catch (Exception e) {
            log("无法请求录屏权限: " + e);
        }
    }

    /** 录屏授权结果回到这里：把 resultCode + data 交给前台服务去建 VirtualDisplay。 */
    private void onProjectionResult(int resultCode, Intent data) {
        if (resultCode != RESULT_OK || data == null) {
            log("录屏授权被拒绝");
            return;
        }
        String path = "/sdcard/Download/rvc_rec_" + System.currentTimeMillis() + ".mp4";
        if (RealtimeService.running) {
            // Android 14+ 要求 mediaProjection 类型在 startForeground 时就声明；
            // 运行中的服务补改类型不一定被允许，所以先停再带授权重启，最稳。
            log("先停一下实时变声，好让录屏服务以 mediaProjection 类型重新前台…");
            startService(new Intent(this, RealtimeService.class)
                    .setAction(RealtimeService.ACTION_STOP));
            awaitStoppedThenRecord(resultCode, data, path, 0);
        } else {
            launchRecordService(resultCode, data, path);
        }
    }

    /** 停止录屏后把输出路由恢复成录屏前的选择。 */
    private void restoreOutputAfterRec() {
        if (preRecOutputMode < 0) return;
        int mode = preRecOutputMode;
        preRecOutputMode = -1;
        setOutputModeUi(mode);
        log("已恢复录屏前的输出路由 → " + Config.OUTPUT_NAMES[mode]);
    }

    /** 轮询等实时服务完全停下，再用授权重启（最多 ~5s）。 */
    private void awaitStoppedThenRecord(int resultCode, Intent data, String path, int tries) {
        if (!RealtimeService.running) {
            launchRecordService(resultCode, data, path);
            return;
        }
        if (tries > 50) {
            log("实时服务没能停下，已取消录屏");
            return;
        }
        new android.os.Handler(getMainLooper()).postDelayed(
                () -> awaitStoppedThenRecord(resultCode, data, path, tries + 1), 100);
    }

    private void launchRecordService(int resultCode, Intent data, String path) {
        rtStartedForRec = true;      // 这条路径总是由录屏连带启动/重启实时服务
        Intent si = new Intent(this, RealtimeService.class)
                .setAction(RealtimeService.ACTION_START)
                .putExtra(RealtimeService.EXTRA_PROJ_CODE, resultCode)
                .putExtra(RealtimeService.EXTRA_PROJ_DATA, data)
                .putExtra(RealtimeService.EXTRA_REC_PATH, path);
        try {
            if (Build.VERSION.SDK_INT >= 26) startForegroundService(si);
            else startService(si);
            log("录屏授权通过，正在启动（实时变声 + 录屏）");
            log("输出: " + path);
        } catch (Exception e) {
            log("启动录屏服务失败: " + e);
            return;
        }
        updateRecUi();
    }

    /** 再次从 assets 解压内置模型（排查损坏 / 升级后强制刷新）。 */
    private void reinstallBundled() {
        log("开始重新解压内置模型...");
        new Thread(() -> {
            boolean ok = ModelBootstrap.reinstall(this, cfg);
            runOnUiThread(() -> {
                modelsReady = ok;
                etModels.setText(cfg.modelDir);
                refreshModels();
                updateStatus();
                syncRtUi();
                log(ok ? ("内置模型已重新解压: " + cfg.modelDir) : "重解压失败，见日志");
            });
        }, "model-reinstall").start();
    }

    private boolean needPerms() {
        boolean mic = checkSelfPermission(android.Manifest.permission.RECORD_AUDIO)
                == PackageManager.PERMISSION_GRANTED;
        boolean notif = Build.VERSION.SDK_INT < 33
                || checkSelfPermission(android.Manifest.permission.POST_NOTIFICATIONS)
                == PackageManager.PERMISSION_GRANTED;
        return !(mic && notif);
    }

    private void startRealTime() {
        if (needPerms()) {
            if (Build.VERSION.SDK_INT >= 33) {
                requestPermissions(new String[]{
                        android.Manifest.permission.RECORD_AUDIO,
                        android.Manifest.permission.POST_NOTIFICATIONS}, 1);
            } else {
                requestPermissions(new String[]{android.Manifest.permission.RECORD_AUDIO}, 1);
            }
            return;
        }
        if (!new File(cfg.modelDir).isDirectory()) {
            log("模型目录不存在: " + cfg.modelDir);
            return;
        }
        Intent i = new Intent(this, RealtimeService.class).setAction(RealtimeService.ACTION_START);
        if (Build.VERSION.SDK_INT >= 26) {
            startForegroundService(i);
        } else {
            startService(i);
        }
        log("已启动实时变声前台服务（可退后台/息屏）");
        btnRT.setText("实时变声 (关)");
        btnGo.setEnabled(false);
    }

    private void stopRealTime() {
        startService(new Intent(this, RealtimeService.class).setAction(RealtimeService.ACTION_STOP));
        log("已请求停止实时变声");
        restoreOutputAfterRec();      // 停实时会连带停录屏，顺手恢复路由
        rtStartedForRec = false;
        syncRtUi();
    }

    // ================================================================== 文件选择
    private void pickWav() {
        Intent i = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        i.addCategory(Intent.CATEGORY_OPENABLE);
        i.setType("audio/*");
        try { startActivityForResult(i, REQ_PICK_WAV); }
        catch (Exception e) { log("无法打开文件选择器: " + e); }
    }

    @Override
    protected void onActivityResult(int req, int res, Intent data) {
        super.onActivityResult(req, res, data);
        if (req == REQ_PROJ) {
            onProjectionResult(res, data);
            return;
        }
        if (res != RESULT_OK || data == null || data.getData() == null) return;
        if (req == REQ_PICK_WAV) {
            Uri uri = data.getData();
            try {
                String base = "picked_" + System.currentTimeMillis() + ".wav";
                String dst = "/sdcard/Download/" + base;
                try (java.io.InputStream in = getContentResolver().openInputStream(uri);
                     FileOutputStream fo = new FileOutputStream(dst)) {
                    byte[] buf = new byte[65536];
                    int n;
                    while (in != null && (n = in.read(buf)) > 0) fo.write(buf, 0, n);
                }
                etIn.setText(dst);
                String name = base.replace(".wav", "") + "_rvc_" + cfg.voiceTag() + ".wav";
                etOut.setText("/sdcard/Download/" + name);
                cfg.inWav = dst;
                cfg.outWav = etOut.getText().toString();
                cfg.save(this);
                log("已选择输入: " + dst);
            } catch (Exception e) {
                log("导入音频失败: " + e);
            }
        }
    }

    private void play(String path) {
        try {
            File f = new File(path);
            if (!f.isFile()) { toast("文件不存在: " + path); return; }
            android.media.MediaPlayer player = new android.media.MediaPlayer();
            player.setDataSource(path);
            player.setOnCompletionListener(mp -> mp.release());
            player.prepare();
            player.start();
            toast("播放: " + f.getName());
        } catch (Exception e) {
            toast("播放失败: " + e);
        }
    }

    // ================================================================== WAV 读写
    static float[] readWav(String path, int[] outSr) throws Exception {
        RandomAccessFile f = new RandomAccessFile(path, "r");
        byte[] hdr = new byte[12];
        f.readFully(hdr);
        if (!(hdr[0] == 'R' && hdr[1] == 'I' && hdr[2] == 'F' && hdr[3] == 'F')) throw new Exception("not wav");
        int sr = 16000, ch = 1, bits = 16;
        byte[] data = null;
        long pos = 12;
        long flen = f.length();
        while (pos + 8 <= flen) {
            f.seek(pos);
            byte[] ch8 = new byte[8];
            f.readFully(ch8);
            String id = new String(ch8, 0, 4, "ASCII");
            int sz = ByteBuffer.wrap(ch8, 4, 4).order(ByteOrder.LITTLE_ENDIAN).getInt();
            if (sz < 0) break;
            long next = pos + 8 + sz + (sz & 1);   // 每个 chunk 都要推进
            if (id.equals("fmt ") && sz >= 16) {
                byte[] fmt = new byte[Math.min(sz, 32)];
                f.readFully(fmt);
                ch = ByteBuffer.wrap(fmt, 2, 2).order(ByteOrder.LITTLE_ENDIAN).getShort();
                sr = ByteBuffer.wrap(fmt, 4, 4).order(ByteOrder.LITTLE_ENDIAN).getInt();
                bits = ByteBuffer.wrap(fmt, 14, 2).order(ByteOrder.LITTLE_ENDIAN).getShort();
            } else if (id.equals("data")) {
                int avail = (int) Math.min((long) sz, flen - pos - 8);
                data = new byte[Math.max(0, avail)];
                if (avail > 0) f.readFully(data);
                break;
            }
            pos = next;
        }
        f.close();
        if (data == null) throw new Exception("no data chunk");
        int bytesPer = bits / 8;
        int frames = data.length / (bytesPer * ch);
        float[] out = new float[frames];
        ByteBuffer bb = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < frames; i++) {
            float s = 0;
            for (int c = 0; c < ch; c++) {
                if (bits == 16) s += bb.getShort() / 32768f;
                else if (bits == 8) s += (bb.get() - 128) / 128f;
                else s += bb.getInt() / 2147483648f;
            }
            out[i] = s / ch;
        }
        outSr[0] = sr;
        return out;
    }

    static float[] resample(float[] x, int from, int to) {
        if (from == to) return x;
        int n = (int) Math.round(x.length * (double) to / from);
        float[] y = new float[n];
        for (int i = 0; i < n; i++) {
            double idx = i * (double) from / to;
            int i0 = (int) idx, i1 = Math.min(i0 + 1, x.length - 1);
            float fr = (float) (idx - i0);
            y[i] = x[i0] * (1 - fr) + x[i1] * fr;
        }
        return y;
    }

    static void writeWav(String path, float[] y, int sr, float gain) throws Exception {
        int n = y.length;
        ByteBuffer bb = ByteBuffer.allocate(44 + n * 2).order(ByteOrder.LITTLE_ENDIAN);
        bb.put("RIFF".getBytes()); bb.putInt(36 + n * 2); bb.put("WAVE".getBytes());
        bb.put("fmt ".getBytes()); bb.putInt(16); bb.putShort((short) 1); bb.putShort((short) 1);
        bb.putInt(sr); bb.putInt(sr * 2); bb.putShort((short) 2); bb.putShort((short) 16);
        bb.put("data".getBytes()); bb.putInt(n * 2);
        for (float v : y) {
            float g = v * gain;
            int s = (int) Math.max(-32768, Math.min(32767, Math.round(g * 32767f)));
            bb.putShort((short) s);
        }
        try (FileOutputStream fo = new FileOutputStream(path)) { fo.write(bb.array()); }
    }

    // ================================================================== 主流程（文件转换）
    /**
     * 文件转换（③「开始转换」/ 命令行 --ez auto）。
     *
     * <p>实现在 {@link OfflineConvert} —— 和「录音模式」共用同一份离线变声代码。
     * 这里原来有一份复制品，两份容易改出偏差（尤其是切块/接缝这种细节），
     * 所以收敛成一处。
     */
    private void runConvert() {
        if (running) return;
        running = true;
        final String inPath = etIn.getText().toString().trim();
        String outPath = etOut.getText().toString().trim();
        try {
            if (inPath.isEmpty()) { log("请输入输入 WAV 路径"); return; }
            if (outPath.isEmpty()) {
                outPath = "/sdcard/Download/out_mnn.wav";
                final String o = outPath;
                etOut.post(() -> etOut.setText(o));
            }
            if (!new File(inPath).isFile()) { log("输入文件不存在: " + inPath); return; }
            final String fOut = outPath;
            runOnUiThread(() -> btnGo.setEnabled(false));

            if (!ModelBootstrap.prepare(this, cfg)) { log("模型未就绪（内置解压失败？）"); return; }
            etModels.post(() -> etModels.setText(cfg.modelDir));
            if (!ensureModels()) return;
            OfflineConvert.convert(engine, cfg, inPath, fOut, this::log);
        } catch (Exception e) {
            log("错误: " + e);
        } finally {
            running = false;
            runOnUiThread(() -> btnGo.setEnabled(true));
        }
    }

    @Override
    public void onRequestPermissionsResult(int req, String[] perms, int[] res) {
        super.onRequestPermissionsResult(req, perms, res);
        if (req == 1) {
            boolean mic = false;
            for (int i = 0; i < perms.length; i++) {
                if (android.Manifest.permission.RECORD_AUDIO.equals(perms[i])
                        && res[i] == PackageManager.PERMISSION_GRANTED) mic = true;
            }
            if (mic) {
                if (pendingRecordAfterPerm) {
                    pendingRecordAfterPerm = false;
                    toggleRecord();
                } else {
                    startRealTime();
                }
            } else {
                pendingRecordAfterPerm = false;
                log("需要录音权限");
            }
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        RvcLog.setSink(null);
        // 实时服务还在跑时不要释放模型，否则会把后台的实时链路打断
        if (!RealtimeService.running) {
            engine.releaseAll();
            EngineWarmup.invalidate();
        }
    }
}
