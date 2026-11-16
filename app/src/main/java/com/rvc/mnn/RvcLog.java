package com.rvc.mnn;

import android.util.Log;

import java.io.FileOutputStream;
import java.util.concurrent.atomic.AtomicReference;

/**
 * 统一日志：既写 /sdcard/Download/rvc_progress.log，也转给当前界面的监听者。
 * Activity 和 Service 共用，保证后台运行时日志不断。
 */
public final class RvcLog {
    public interface Sink {
        void onLog(String s);
    }

    private static final AtomicReference<Sink> SINK = new AtomicReference<>();

    private RvcLog() {}

    public static void setSink(Sink s) {
        SINK.set(s);
    }

    public static void i(String tag, String msg) {
        Log.i(tag, msg);
        String line = new java.text.SimpleDateFormat("MM-dd HH:mm:ss.SSS", java.util.Locale.US)
                .format(new java.util.Date()) + " " + msg;
        try (FileOutputStream fo = new FileOutputStream("/sdcard/Download/rvc_progress.log", true)) {
            fo.write((line + "\n").getBytes("UTF-8"));
        } catch (Exception ignored) {}
        Sink s = SINK.get();
        if (s != null) s.onLog(msg);
    }

    public static void clearFile() {
        try (FileOutputStream fo = new FileOutputStream("/sdcard/Download/rvc_progress.log")) {
            fo.write(new byte[0]);
        } catch (Exception ignored) {}
    }
}
