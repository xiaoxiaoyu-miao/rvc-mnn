// mnn_run: 通用 MNN 推理 CLI（支持任意输入名/数量，自动按模型输入顺序喂数）
// 用法: mnn_run <model.mnn> <backend:cpu|vulkan|opencl> <out.raw> <reps> [name=file ...]
// 环境: RVC_PREC(0=Low,1=Normal,2=High) RVC_THREADS
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>
#include <map>
#include <chrono>

static std::vector<float> readRaw(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "open fail: %s\n", path); return {}; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<float> v(sz / 4);
    size_t n = fread(v.data(), 4, v.size(), f);
    fclose(f);
    v.resize(n);
    return v;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s model.mnn backend out.raw reps [name=file ...]\n", argv[0]);
        return 1;
    }
    const char* modelPath = argv[1];
    std::string backend = argv[2];
    const char* outPath = argv[3];
    int reps = atoi(argv[4]);
    std::map<std::string, std::string> feedFiles;
    for (int i = 5; i < argc; i++) {
        std::string a = argv[i];
        auto p = a.find('=');
        if (p != std::string::npos) feedFiles[a.substr(0, p)] = a.substr(p + 1);
    }

    std::shared_ptr<MNN::Interpreter> net(MNN::Interpreter::createFromFile(modelPath));
    if (!net) { fprintf(stderr, "load model fail\n"); return 1; }

    MNN::ScheduleConfig cfg;
    if (backend == "vulkan") cfg.type = MNN_FORWARD_VULKAN;
    else if (backend == "opencl") cfg.type = MNN_FORWARD_OPENCL;
    else if (backend == "qnn") cfg.type = MNN_FORWARD_NN;
    else cfg.type = MNN_FORWARD_CPU;
    cfg.numThread = 4;
    if (const char* th = getenv("RVC_THREADS")) cfg.numThread = atoi(th);
    MNN::BackendConfig bc;
    const char* precEnv = getenv("RVC_PREC");
    int prec = precEnv ? atoi(precEnv) : 1;
    if (prec <= 0) bc.precision = MNN::BackendConfig::Precision_Low;
    else if (prec == 1) bc.precision = MNN::BackendConfig::Precision_Normal;
    else bc.precision = MNN::BackendConfig::Precision_High;
    bc.power = MNN::BackendConfig::Power_High;
    cfg.backendConfig = &bc;

    auto t0 = std::chrono::steady_clock::now();
    MNN::Session* session = net->createSession(cfg);
    if (!session) { fprintf(stderr, "create session fail\n"); return 1; }
    auto t1 = std::chrono::steady_clock::now();
    fprintf(stderr, "session create: %.1f ms\n", std::chrono::duration<double, std::milli>(t1 - t0).count());

    // 收集输入
    std::map<std::string, std::vector<float>> datas;
    for (auto& kv : feedFiles) {
        auto v = readRaw(kv.second.c_str());
        datas[kv.first] = v;
        fprintf(stderr, "  feed %s: %zu floats\n", kv.first.c_str(), v.size());
    }
    // 动态 shape：若输入含 -1，按喂入数据量推断（仅支持 [1,N] 形态）
    for (auto& kv : datas) {
        MNN::Tensor* t = net->getSessionInput(session, kv.first.c_str());
        if (!t) continue;
        std::vector<int> shp = t->shape();
        fprintf(stderr, "  input %s shape:", kv.first.c_str());
        for (int d : shp) fprintf(stderr, " %d", d);
        fprintf(stderr, "\n");
        bool dyn = false;
        for (int d : shp) if (d < 0) dyn = true;
        if (!dyn) continue;
        long known = 1;
        for (int d : shp) if (d > 0) known *= d;
        if (known <= 0 || (long)kv.second.size() % known != 0) {
            fprintf(stderr, "  (cannot infer shape for %s)\n", kv.first.c_str());
            continue;
        }
        int nDyn = 0, dynIdx = -1;
        for (size_t i = 0; i < shp.size(); i++) if (shp[i] < 0) { nDyn++; dynIdx = (int)i; }
        if (nDyn != 1) {
            fprintf(stderr, "  (%d dynamic dims in %s, cannot infer)\n", nDyn, kv.first.c_str());
            continue;
        }
        std::vector<int> want = shp;
        want[dynIdx] = (int)((long)kv.second.size() / known);
        net->resizeTensor(t, want);
        fprintf(stderr, "  resize %s ->", kv.first.c_str());
        for (int d : want) fprintf(stderr, " %d", d);
        fprintf(stderr, "\n");
    }
    net->resizeSession(session);

    auto feedAll = [&]() {
        for (auto& kv : datas) {
            MNN::Tensor* t = net->getSessionInput(session, kv.first.c_str());
            if (!t) { fprintf(stderr, "  (no input %s)\n", kv.first.c_str()); continue; }
            MNN::Tensor host(t, t->getDimensionType());
            size_t n = host.elementSize() < kv.second.size() ? host.elementSize() : kv.second.size();
            memcpy(host.host<float>(), kv.second.data(), n * 4);
            t->copyFromHostTensor(&host);
        }
    };
    feedAll();

    double best = 1e9, total = 0;
    for (int i = 0; i < reps; i++) {
        feedAll();   // MNN 会就地复用输入 buffer，每次必须重新喂
        auto s = std::chrono::steady_clock::now();
        net->runSession(session);
        auto e = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(e - s).count();
        total += ms; if (ms < best) best = ms;
    }
    fprintf(stderr, "run: %d reps, avg %.2f ms, best %.2f ms\n", reps, total / reps, best);

    auto outs = net->getSessionOutputAll(session);
    if (outs.empty()) { fprintf(stderr, "no output\n"); return 1; }
    MNN::Tensor* out = outs.begin()->second;
    fprintf(stderr, "  output: %s\n", outs.begin()->first.c_str());
    MNN::Tensor outHost(out, out->getDimensionType());
    out->copyToHostTensor(&outHost);
    size_t n = outHost.elementSize();
    FILE* f = fopen(outPath, "wb");
    if (!f) { fprintf(stderr, "write fail %s\n", outPath); return 1; }
    fwrite(outHost.host<float>(), 4, n, f);
    fclose(f);
    float* p = outHost.host<float>();
    double rms = 0; float mx = 0;
    for (size_t i = 0; i < n; i++) { rms += p[i]*p[i]; if (fabsf(p[i]) > mx) mx = fabsf(p[i]); }
    fprintf(stderr, "out: %zu floats, rms %.4f, max %.4f -> %s\n", n, sqrt(rms/n), mx, outPath);
    return 0;
}
