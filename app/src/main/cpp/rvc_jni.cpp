// RVC MNN JNI: hubert + 合成器 走 MNN(Vulkan/CPU)，fcpe 由 Java 层 ORT 负责
#include <jni.h>
#include <android/log.h>
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <cstdio>
#include <vector>
#include <cstring>
#include <chrono>
#include <mutex>
#include <memory>
#include <string>
#include <cmath>
#include <thread>
#include <algorithm>
#include <cstdlib>
#include <sys/system_properties.h>

#define LOG_TAG "RvcNative"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {
struct Model {
    std::shared_ptr<MNN::Interpreter> net;
    MNN::Session* session = nullptr;
    MNN::BackendConfig bc;
    std::string inName, outName;
    int outFloats = 0;
    // 预分配复用，避免每块 create/free
    std::shared_ptr<MNN::Tensor> hostIn[3];
    std::shared_ptr<MNN::Tensor> hostOut;
    std::vector<std::string> inNames;
};
Model g_hub, g_ct;
std::mutex g_muHub, g_muCt, g_muInit;
bool g_ready = false;
// 当前已加载的配置指纹；配置变化时自动重建 session
std::string g_cfgKey;

// 线程数：RVC_THREADS 环境变量优先；否则按 CPU 拓扑推算大核数（取 max_freq 最高那档）×2，
// 夹在 2..8。例：2 个 4.32GHz 大核 + 6 个 3.53GHz 中核 → 2×2 = 4 线程。
static int detectNumThreads() {
    if (const char* th = getenv("RVC_THREADS")) {
        int v = atoi(th);
        if (v > 0) return v;
    }
    // App 拿不到 zygote 的环境变量，改用系统属性方便现场调优：
    //   su -c 'setprop debug.rvc.threads 6'
    char prop[PROP_VALUE_MAX];
    if (__system_property_get("debug.rvc.threads", prop) > 0) {
        int v = atoi(prop);
        if (v > 0) return v;
    }
    long best = 0;
    int big = 0;
    for (int i = 0; i < 16; i++) {
        char path[128];
        snprintf(path, sizeof(path),
                 "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", i);
        FILE* f = fopen(path, "r");
        if (!f) continue;
        long v = 0;
        if (fscanf(f, "%ld", &v) == 1) {
            if (v > best) { best = v; big = 1; }
            else if (v == best) big++;
        }
        fclose(f);
    }
    if (big > 0) {
        int t = big * 2;
        return std::max(2, std::min(8, t));
    }
    unsigned n = std::thread::hardware_concurrency();
    return n ? (int)std::min(n, 8u) : 4;
}

static bool setup(Model& m, const char* path, const char* inName, const char* outName,
                  bool useVulkan, int prec, std::vector<int>* dynShape, int dynValue,
                  const char** inNames3 = nullptr, int numThread = 0) {
    auto _t0 = std::chrono::steady_clock::now();
    m.net.reset(MNN::Interpreter::createFromFile(path));
    if (!m.net) { LOGE("load fail %s", path); return false; }
    MNN::ScheduleConfig cfg;
    cfg.type = useVulkan ? MNN_FORWARD_VULKAN : MNN_FORWARD_CPU;
    cfg.numThread = numThread > 0 ? numThread : detectNumThreads();
    if (prec <= 0)      m.bc.precision = MNN::BackendConfig::Precision_Low;
    else if (prec == 1) m.bc.precision = MNN::BackendConfig::Precision_Normal;
    else                m.bc.precision = MNN::BackendConfig::Precision_High;
    m.bc.power = MNN::BackendConfig::Power_High;
    cfg.backendConfig = &m.bc;
    m.session = m.net->createSession(cfg);
    if (!m.session) { LOGE("session fail %s", path); return false; }
    m.inName = inName; m.outName = outName;
    if (dynShape) {
        MNN::Tensor* t = m.net->getSessionInput(m.session, inName);
        if (t) {
            std::vector<int> want = *dynShape;
            bool need = false;
            for (size_t i = 0; i < want.size(); i++) if (want[i] < 0) { want[i] = dynValue; need = true; }
            if (need) { m.net->resizeTensor(t, want); m.net->resizeSession(m.session); }
        }
    }
    MNN::Tensor* o = m.net->getSessionOutput(m.session, outName);
    m.outFloats = o ? (int)o->elementSize() : 0;
    if (o) m.hostOut.reset(MNN::Tensor::create<float>(o->shape(), nullptr, MNN::Tensor::CAFFE));
    for (int i = 0; i < 3 && inNames3; i++) {
        MNN::Tensor* it = m.net->getSessionInput(m.session, inNames3[i]);
        if (it) m.hostIn[i].reset(MNN::Tensor::create<float>(it->shape(), nullptr, MNN::Tensor::CAFFE));
    }
    LOGI("model %s ready, out=%d, vulkan=%d prec=%d threads=%d loadMs=%ld", path, m.outFloats,
         (int)useVulkan, prec, cfg.numThread,
         (long)std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now() - _t0).count());
    return m.outFloats > 0;
}

// 调用方需持有 g_muInit
static void teardownLocked() {
    for (Model* m : {&g_hub, &g_ct}) {
        if (m->net && m->session) { m->net->releaseSession(m->session); m->session = nullptr; }
        m->net.reset();
        for (auto& h : m->hostIn) h.reset();
        m->hostOut.reset();
        m->outFloats = 0;
    }
    g_ready = false;
    g_cfgKey.clear();
}
} // namespace

extern "C" {

/**
 * 初始化：hubert + 合成器。参数变化（后端/精度/模型文件）时自动重建。
 *
 * @param vulkan  true=合成器走 Vulkan，false=CPU
 * @param hubPrec hubert 精度：0=fp16 1=normal 2=fp32
 * @param ctPrec  合成器精度：0=fp16 1=normal 2=fp32
 */
JNIEXPORT jboolean JNICALL
Java_com_rvc_mnn_RvcEngine_init(JNIEnv* env, jclass, jstring hubPath, jstring ctPath,
                                   jboolean vulkan, jint hubPrec, jint ctPrec, jint threads) {
    std::lock_guard<std::mutex> lk(g_muInit);
    const char* hp = env->GetStringUTFChars(hubPath, nullptr);
    const char* cp = env->GetStringUTFChars(ctPath, nullptr);
    bool vk = vulkan == JNI_TRUE;

    const int nThread = threads > 0 ? (int)threads : detectNumThreads();
    std::string key = std::string(hp) + "|" + cp + "|" + (vk ? "1" : "0") + "|" +
                      std::to_string((int)hubPrec) + "|" + std::to_string((int)ctPrec) + "|" +
                      std::to_string(nThread);
    if (g_ready && key == g_cfgKey) {
        env->ReleaseStringUTFChars(hubPath, hp);
        env->ReleaseStringUTFChars(ctPath, cp);
        return JNI_TRUE;   // 配置没变，复用已有 session
    }
    if (g_ready) {
        LOGI("config changed, rebuilding sessions");
        teardownLocked();
    }

    // hubert: input_values [1,-1] -> 7840 ；合成器: 静态 [1,49,768] 等
    std::vector<int> hubDyn = {1, -1};
    bool ok1 = setup(g_hub, hp, "input_values", "features", false, (int)hubPrec, &hubDyn, 7840,
                     nullptr, nThread);
    const char* ctIns[3] = {"phone", "nsff0", "pitch_emb"};
    bool ok2 = setup(g_ct, cp, "phone", "audio", vk, (int)ctPrec, nullptr, 0, ctIns, nThread);
    env->ReleaseStringUTFChars(hubPath, hp);
    env->ReleaseStringUTFChars(ctPath, cp);
    g_ready = ok1 && ok2;
    if (g_ready) g_cfgKey = key;
    LOGE("init: hubert=%d synth=%d threads=%d ready=%d", ok1, ok2, nThread, g_ready);
    return g_ready ? JNI_TRUE : JNI_FALSE;
}

// 返回帧数 T（输出写进 out，最多 64*768）
JNIEXPORT jint JNICALL
Java_com_rvc_mnn_RvcEngine_hubert(JNIEnv* env, jclass, jfloatArray segArr, jfloatArray outArr) {
    std::lock_guard<std::mutex> lk(g_muHub);
    if (!g_ready) return 0;
    jsize n = env->GetArrayLength(segArr);
    jfloat* seg = env->GetFloatArrayElements(segArr, nullptr);
    MNN::Tensor* t = g_hub.net->getSessionInput(g_hub.session, g_hub.inName.c_str());
    // 输入含 -1 时按实际长度 reshape
    std::vector<int> shp = t->shape();
    int dynIdx = -1;
    for (size_t i = 0; i < shp.size(); i++) if (shp[i] < 0) dynIdx = (int)i;
    if (dynIdx >= 0 && shp[dynIdx] != (int)n) {
        std::vector<int> want = shp; want[dynIdx] = (int)n;
        g_hub.net->resizeTensor(t, want); g_hub.net->resizeSession(g_hub.session);
        t = g_hub.net->getSessionInput(g_hub.session, g_hub.inName.c_str());
    }
    auto _a = std::chrono::steady_clock::now();
    if (!g_hub.hostIn[0] || g_hub.hostIn[0]->elementSize() != (size_t)n)
        g_hub.hostIn[0].reset(MNN::Tensor::create<float>(t->shape(), nullptr, MNN::Tensor::CAFFE));
    memcpy(g_hub.hostIn[0]->host<float>(), seg, sizeof(float) * n);
    t->copyFromHostTensor(g_hub.hostIn[0].get());
    env->ReleaseFloatArrayElements(segArr, seg, JNI_ABORT);
    g_hub.net->runSession(g_hub.session);
    MNN::Tensor* o = g_hub.net->getSessionOutput(g_hub.session, g_hub.outName.c_str());
    if (!o) return 0;
    int total = (int)o->elementSize();
    int T = o->shape().size() >= 2 ? o->length(1) : 0;
    if (!g_hub.hostOut) g_hub.hostOut.reset(MNN::Tensor::create<float>(o->shape(), nullptr, MNN::Tensor::CAFFE));
    o->copyToHostTensor(g_hub.hostOut.get());
    jsize cap = env->GetArrayLength(outArr);
    int cpy = total < cap ? total : cap;
    env->SetFloatArrayRegion(outArr, 0, cpy, g_hub.hostOut->host<float>());
    __android_log_print(ANDROID_LOG_INFO, "RvcNative", "hubert %ldms",
        (long)std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-_a).count());
    return T;
}

JNIEXPORT jboolean JNICALL
Java_com_rvc_mnn_RvcEngine_synth(JNIEnv* env, jclass, jfloatArray phoneArr, jfloatArray nsArr,
                                 jfloatArray peArr, jfloatArray outArr) {
    std::lock_guard<std::mutex> lk(g_muCt);
    if (!g_ready) return JNI_FALSE;
    struct F { const char* name; jfloatArray arr; };
    F feeds[3] = {{"phone", phoneArr}, {"nsff0", nsArr}, {"pitch_emb", peArr}};
    for (auto& f : feeds) {
        jsize n = env->GetArrayLength(f.arr);
        jfloat* p = env->GetFloatArrayElements(f.arr, nullptr);
        MNN::Tensor* t = g_ct.net->getSessionInput(g_ct.session, f.name);
        if (t) {
            int idx = (&f - feeds);
            if (!g_ct.hostIn[idx]) g_ct.hostIn[idx].reset(MNN::Tensor::create<float>(t->shape(), nullptr, MNN::Tensor::CAFFE));
            int cpy = (int)g_ct.hostIn[idx]->elementSize() < (int)n ? (int)g_ct.hostIn[idx]->elementSize() : (int)n;
            memcpy(g_ct.hostIn[idx]->host<float>(), p, sizeof(float) * cpy);
            t->copyFromHostTensor(g_ct.hostIn[idx].get());
        }
        env->ReleaseFloatArrayElements(f.arr, p, JNI_ABORT);
    }
    auto _a = std::chrono::steady_clock::now();
    g_ct.net->runSession(g_ct.session);
    MNN::Tensor* o = g_ct.net->getSessionOutput(g_ct.session, g_ct.outName.c_str());
    if (!o) return JNI_FALSE;
    if (!g_ct.hostOut) g_ct.hostOut.reset(MNN::Tensor::create<float>(o->shape(), nullptr, MNN::Tensor::CAFFE));
    o->copyToHostTensor(g_ct.hostOut.get());
    int total = (int)o->elementSize();
    jsize cap = env->GetArrayLength(outArr);
    int cpy = total < cap ? total : cap;
    env->SetFloatArrayRegion(outArr, 0, cpy, g_ct.hostOut->host<float>());
    __android_log_print(ANDROID_LOG_INFO, "RvcNative", "synth %ldms",
        (long)std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-_a).count());
    return JNI_TRUE;
}

JNIEXPORT jint JNICALL
Java_com_rvc_mnn_RvcEngine_audioLen(JNIEnv*, jclass) {
    std::lock_guard<std::mutex> lk(g_muCt);
    return g_ready ? g_ct.outFloats : 0;
}

/** 后端描述串，供 UI 显示，例如 "Vulkan/fp32" 或 "CPU/normal"。 */
JNIEXPORT jstring JNICALL
Java_com_rvc_mnn_RvcEngine_backendInfo(JNIEnv* env, jclass) {
    std::lock_guard<std::mutex> lk(g_muInit);
    const char* backend = (g_ct.session && g_ct.bc.precision == MNN::BackendConfig::Precision_High)
                              ? "High" : "Normal";
    char buf[128];
    snprintf(buf, sizeof(buf), "ready=%d ctOut=%d ctPrec=%s", (int)g_ready, g_ct.outFloats, backend);
    return env->NewStringUTF(buf);
}

JNIEXPORT void JNICALL
Java_com_rvc_mnn_RvcEngine_release(JNIEnv*, jclass) {
    std::lock_guard<std::mutex> lk(g_muInit);
    teardownLocked();
}

} // extern "C"
