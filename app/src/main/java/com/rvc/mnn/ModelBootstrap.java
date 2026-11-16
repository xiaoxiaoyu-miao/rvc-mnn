package com.rvc.mnn;

import android.content.Context;
import android.content.pm.PackageInfo;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.security.MessageDigest;

/**
 * 内置模型自解压 + 完整性校验。
 *
 * <p>模型（hubert_fp16.mnn / ct_static.mnn / fcpe_audio.onnx / emb_pitch.bin）打进 APK 的
 * {@code assets/models/}，首次启动解压到应用私有目录 {@code files/models/}，之后直接用该目录。
 *
 * <p><b>「确保最新」是怎么保证的</b>：
 * <ol>
 *   <li>解压版本号 = {@code versionCode + "-" + lastUpdateTime}；APK 一升级就自动重解压，
 *       所以打进 assets 的新模型一定会落到设备上（换模型时记得同时 bump versionCode）。</li>
 *   <li>每个模型在编译期把 SHA-256 写进 {@link #SHA256} 表；解压时逐个校验，
 *       不一致直接判失败，不会把半截/损坏的文件当可用模型用。</li>
 *   <li>标记文件里同时记录解压时的哈希，{@link #verify(Context, Config)} 可随时复核。</li>
 * </ol>
 *
 * <p>如果用户自己配置的模型目录（默认 /sdcard/models/mm）里四个文件齐全，优先用用户的，
 * 不动内置那份。
 */
public final class ModelBootstrap {

    public static final String[] FILES = {
            "hubert_fp16.mnn", "ct_static.mnn", "fcpe_audio.onnx", "emb_pitch.bin"};

    /**
     * 与 app/src/main/assets/models/ 下四个文件逐一对应的 SHA-256
     * （换模型时：更新文件 → 更新这张表 → bump versionCode）。
     */
    public static final String[] SHA256 = {
            "d53a345949ab066157ffb83c79cd840faa283f590f2f45a923494d0f05d06d40", // hubert_fp16.mnn
            "f53d581d37d674eee01f3a9d567bf78db171705d45e14b20a7ff9d1ee5f2e1fd", // ct_static.mnn
            "b7e4f3871b10641869b7ac5a2d56ed94deb37552c0336d77e17ad6e66760adf0", // fcpe_audio.onnx
            "5d85c7fbb73c7b3324fa855c2636e3fb8d4c4c9e1a42f2662346541bcdf8591e", // emb_pitch.bin
    };

    private static final String ASSET_DIR = "models";
    private static final String MARKER = ".installed";
    private static final Object LOCK = new Object();

    private ModelBootstrap() {
    }

    /** 解压目标：应用私有目录（无权限、随 App 卸载清掉）。 */
    public static File dir(Context c) {
        return new File(c.getFilesDir(), "models");
    }

    /** 目录里四个模型是否齐全（只看存在与大小，不算哈希）。 */
    public static boolean hasAll(String modelDir) {
        if (modelDir == null || modelDir.isEmpty()) return false;
        for (String f : FILES) {
            File x = new File(modelDir, f);
            if (!x.isFile() || x.length() < 1024) return false;
        }
        return true;
    }

    private static String versionTag(Context c) {
        try {
            PackageInfo pi = c.getPackageManager().getPackageInfo(c.getPackageName(), 0);
            return pi.versionCode + "-" + pi.lastUpdateTime;
        } catch (Exception e) {
            return "0-0";
        }
    }

    /** 标记文件内容 = 版本号 + 解压时的四个哈希。 */
    private static String expectedMarker(Context c) {
        StringBuilder sb = new StringBuilder(versionTag(c));
        sb.append('|');
        for (int i = 0; i < SHA256.length; i++) {
            if (i > 0) sb.append(',');
            sb.append(SHA256[i]);
        }
        return sb.toString();
    }

    private static String readMarker(Context c) {
        try {
            File m = new File(dir(c), MARKER);
            if (!m.isFile()) return "";
            return new String(Files.readAllBytes(m.toPath()), StandardCharsets.UTF_8).trim();
        } catch (Exception e) {
            return "";
        }
    }

    public static boolean isInstalled(Context c) {
        return hasAll(dir(c).getAbsolutePath())
                && expectedMarker(c).equals(readMarker(c));
    }

    /** 该模型在编译期声明的 SHA-256。 */
    public static String expectedSha(String name) {
        for (int i = 0; i < FILES.length; i++) {
            if (FILES[i].equals(name)) return SHA256[i];
        }
        return "";
    }

    /** 计算文件 SHA-256（小写十六进制）。大文件按 4MB 块读，内存占用恒定。 */
    public static String sha256(File f) {
        if (f == null || !f.isFile()) return "";
        try (InputStream in = new java.io.FileInputStream(f)) {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            byte[] buf = new byte[4 << 20];
            int n;
            while ((n = in.read(buf)) > 0) {
                md.update(buf, 0, n);
            }
            byte[] d = md.digest();
            StringBuilder sb = new StringBuilder(d.length * 2);
            for (byte b : d) {
                sb.append(Character.forDigit((b >> 4) & 0xF, 16));
                sb.append(Character.forDigit(b & 0xF, 16));
            }
            return sb.toString();
        } catch (Exception e) {
            return "";
        }
    }

    /**
     * 复核当前模型目录：逐个比对 SHA-256。
     *
     * @return 每个文件一行「文件名 ✓/✗ (大小)」，可直接打到界面上
     */
    public static String[] verify(Context c, Config cfg) {
        String dirPath = (cfg != null && cfg.modelDir != null && !cfg.modelDir.isEmpty())
                ? cfg.modelDir : dir(c).getAbsolutePath();
        String[] out = new String[FILES.length];
        for (int i = 0; i < FILES.length; i++) {
            File f = new File(dirPath, FILES[i]);
            if (!f.isFile()) {
                out[i] = FILES[i] + "  ✗ 缺失";
                continue;
            }
            String actual = sha256(f);
            boolean ok = SHA256[i].equalsIgnoreCase(actual);
            out[i] = FILES[i] + "  " + (ok ? "✓ OK" : "✗ 哈希不符")
                    + "  (" + (f.length() >> 20) + "MB)";
        }
        return out;
    }

    /** 指纹摘要，打日志/界面用。 */
    public static String fingerprint() {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < FILES.length; i++) {
            if (i > 0) sb.append(' ');
            sb.append(FILES[i]).append('=').append(SHA256[i].substring(0, 8));
        }
        return sb.toString();
    }

    /**
     * 确保有一份可用模型目录：
     * <ol>
     *   <li>用户目录齐全 → 用它；</li>
     *   <li>内置已解压且版本+哈希匹配 → 用私有目录；</li>
     *   <li>否则从 assets 解压（边解边校验），再把 cfg.modelDir 指过去。</li>
     * </ol>
     */
    public static boolean prepare(Context c, Config cfg) {
        synchronized (LOCK) {
            if (cfg == null) {
                cfg = new Config();
                cfg.load(c);
            }
            final String bundled = dir(c).getAbsolutePath();

            // ① 用户在设置里指定的「外部」目录（不在本 App 私有目录下）：齐全就尊重用户，不动内置。
            //    以前这里写的是 hasAll(cfg.modelDir) 就返回——而 cfg.modelDir 默认就指向内置解压目录，
            //    于是 versionCode 变了也永远短路，新 assets 永远落不到设备上。
            if (hasAll(cfg.modelDir) && !isUnder(cfg.modelDir, c.getFilesDir().getAbsolutePath())) {
                RvcLog.i("ModelBootstrap", "使用用户指定模型目录: " + cfg.modelDir);
                return true;
            }

            // ② 内置目录：必须「版本号 + 四个哈希」全对才算装好，否则重解压
            if (isInstalled(c)) {
                if (!bundled.equals(cfg.modelDir)) {
                    cfg.modelDir = bundled;
                    cfg.save(c);
                }
                RvcLog.i("ModelBootstrap", "内置模型已就绪且校验通过: " + bundled);
                return true;
            }

            // ③ 解压（边解边校验）
            try {
                long t0 = System.currentTimeMillis();
                RvcLog.i("ModelBootstrap", "解压内置模型到 " + bundled
                        + "（指纹 " + fingerprint() + "）");
                extract(c);
                cfg.modelDir = bundled;
                cfg.save(c);
                RvcLog.i("ModelBootstrap", "模型解压+校验完成，用时 "
                        + (System.currentTimeMillis() - t0) + "ms");
                return true;
            } catch (Exception e) {
                RvcLog.i("ModelBootstrap", "解压失败: " + e);
                return false;
            }
        }
    }

    /** path 是否位于 parent 目录下（含相等判断），用规范路径比较，避开 /data/data 与 /data/user/0 的别名。 */
    private static boolean isUnder(String path, String parent) {
        if (path == null || parent == null) return false;
        try {
            String a = new File(path).getCanonicalPath();
            String b = new File(parent).getCanonicalPath();
            return a.equals(b) || a.startsWith(b + File.separator);
        } catch (Exception e) {
            return path.startsWith(parent);
        }
    }

    /**
     * 强制重新解压（资产损坏排查 / 手动「确保最新」用）。
     *
     * <p>注意不能只是删标记再调 {@link #prepare}：prepare 会先判目录齐全而短路，
     * 结果既不重解压也不补标记。这里直接走 extract。
     */
    public static boolean reinstall(Context c, Config cfg) {
        synchronized (LOCK) {
            File m = new File(dir(c), MARKER);
            //noinspection ResultOfMethodCallIgnored
            m.delete();
            try {
                extract(c);
                if (cfg == null) {
                    cfg = new Config();
                    cfg.load(c);
                }
                cfg.modelDir = dir(c).getAbsolutePath();
                cfg.save(c);
                return true;
            } catch (Exception e) {
                RvcLog.i("ModelBootstrap", "强制重解压失败: " + e);
                return false;
            }
        }
    }

    private static void extract(Context c) throws IOException {
        File d = dir(c);
        if (!d.isDirectory() && !d.mkdirs()) throw new IOException("无法创建目录: " + d);
        for (int i = 0; i < FILES.length; i++) {
            String name = FILES[i];
            File out = new File(d, name);
            long t0 = System.currentTimeMillis();
            RvcLog.i("ModelBootstrap", "解压 " + name + " ...");
            try (InputStream in = c.getAssets().open(ASSET_DIR + "/" + name);
                 FileOutputStream fo = new FileOutputStream(out)) {
                byte[] buf = new byte[1 << 20];
                int n;
                long total = 0;
                while ((n = in.read(buf)) > 0) {
                    fo.write(buf, 0, n);
                    total += n;
                }
                fo.getFD().sync();
                RvcLog.i("ModelBootstrap", "  " + name + " " + (total >> 20) + "MB / "
                        + (System.currentTimeMillis() - t0) + "ms");
            }
            // 边解边校验：损坏就抛出，避免把坏模型当可用
            String actual = sha256(out);
            if (!SHA256[i].equalsIgnoreCase(actual)) {
                //noinspection ResultOfMethodCallIgnored
                out.delete();
                throw new IOException(name + " 校验失败：期望 " + SHA256[i]
                        + " 实际 " + actual);
            }
            RvcLog.i("ModelBootstrap", "  " + name + " ✓ sha256 一致");
        }
        try (FileOutputStream fo = new FileOutputStream(new File(d, MARKER))) {
            fo.write(expectedMarker(c).getBytes(StandardCharsets.UTF_8));
        }
        RvcLog.i("ModelBootstrap", "全部模型就绪并已校验: " + d);
    }
}
