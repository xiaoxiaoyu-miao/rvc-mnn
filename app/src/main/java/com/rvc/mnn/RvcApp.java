package com.rvc.mnn;

import android.app.Application;

import com.google.android.material.color.DynamicColors;

/**
 * 应用入口：给全局接上 Material 3 动态取色（Monet）。
 *
 * <p>Android 12+ 会从壁纸里抽出种子色，覆盖 {@code Theme.RvcM3} 里写死的兜底色；
 * 低版本自动跳过，用 XML 里的 static 配色。
 */
public class RvcApp extends Application {

    @Override
    public void onCreate() {
        super.onCreate();
        // 先落一份静态主题，再按可用性叠加动态取色
        DynamicColors.applyToActivitiesIfAvailable(this);
        RvcLog.i("RvcApp", "Material3 动态取色已尝试启用（SDK " + android.os.Build.VERSION.SDK_INT + "）");
    }
}
