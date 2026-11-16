// rvc_mnn: 用 MNN（Vulkan/CPU）跑 RVC 合成器，输入/输出均为原始 float32 raw
// 用法: rvc_mnn <model.mnn> <backend> <phone.raw> <nsff0.raw> <pitch_emb.raw> <out.raw>
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>
#include <chrono>

static std::vector<float> readRaw(const char* path, size_t expect) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "open fail: %s\n", path); exit(1); }
    std::vector<float> v(expect);
    size_t n = fread(v.data(), 4, expect, f);
    fclose(f);
    fprintf(stderr, "  read %s: %zu floats\n", path, n);
    return v;
}

int main(int argc, char** argv) {
    if (argc < 7) {
        fprintf(stderr, "usage: %s model.mnn backend(vulkan|opencl|cpu) phone.raw nsff0.raw pitch_emb.raw out.raw [reps]\n", argv[0]);
        return 1;
    }
    const char* modelPath = argv[1];
    std::string backend = argv[2];
    int reps = (argc > 7) ? atoi(argv[7]) : 1;

    std::shared_ptr<MNN::Interpreter> net(MNN::Interpreter::createFromFile(modelPath));
    if (!net) { fprintf(stderr, "load model fail\n"); return 1; }

    MNN::ScheduleConfig cfg;
    if (backend == "vulkan") cfg.type = MNN_FORWARD_VULKAN;
    else if (backend == "opencl") cfg.type = MNN_FORWARD_OPENCL;
    else if (backend == "qnn") cfg.type = MNN_FORWARD_QNN;
    else cfg.type = MNN_FORWARD_CPU;
    cfg.numThread = 4;
    if (const char* th = getenv("RVC_THREADS")) cfg.numThread = atoi(th);
    MNN::BackendConfig bc;
    // 精度：默认 Normal(fp32)。Precision_Low=fp16 在 ARM 上会真实启用 fp16 累加，
    // 768 维 MatMul 会被打坏（实测 corr 0.22）。需 fp16 时显式 RVC_LOW=1。
    const char* precEnv = getenv("RVC_PREC");   // 0=Low(fp16) 1=Normal 2=High(fp32)
    int prec = precEnv ? atoi(precEnv) : 1;
    if (prec <= 0)      bc.precision = MNN::BackendConfig::Precision_Low;
    else if (prec == 1) bc.precision = MNN::BackendConfig::Precision_Normal;
    else                bc.precision = MNN::BackendConfig::Precision_High;
    bc.power = MNN::BackendConfig::Power_High;
    cfg.backendConfig = &bc;

    auto t0 = std::chrono::steady_clock::now();
    MNN::Session* session = net->createSession(cfg);
    if (!session) { fprintf(stderr, "create session fail (backend %s)\n", backend.c_str()); return 1; }
    auto t1 = std::chrono::steady_clock::now();
    fprintf(stderr, "session create: %.1f ms\n", std::chrono::duration<double, std::milli>(t1 - t0).count());

    auto showT = [&](const char* tag, MNN::Tensor* t) {
        if (!t) { fprintf(stderr, "  %s: MISSING\n", tag); return; }
        fprintf(stderr, "  %s dims=", tag);
        for (int i = 0; i < t->dimensions(); i++) fprintf(stderr, "%d ", t->length(i));
        fprintf(stderr, " type=%d\n", t->getDimensionType());
    };
    showT("in.phone", net->getSessionInput(session, "phone"));
    showT("in.nsff0", net->getSessionInput(session, "nsff0"));
    showT("in.pitch_emb", net->getSessionInput(session, "pitch_emb"));
    showT("out.audio", net->getSessionOutput(session, "audio"));
    auto phone = readRaw(argv[3], 1*49*768);
    auto nsff0 = readRaw(argv[4], 1*49);
    auto pe    = readRaw(argv[5], 1*49*192);

    auto feedIfExists = [&](const char* name, std::vector<float>& src) {
        MNN::Tensor* t = net->getSessionInput(session, name);
        if (!t) { fprintf(stderr, "  (skip missing input %s)\n", name); return; }
        MNN::Tensor host(t, t->getDimensionType());
        size_t n = host.elementSize() < src.size() ? host.elementSize() : src.size();
        memcpy(host.host<float>(), src.data(), n * 4);
        t->copyFromHostTensor(&host);
    };
    // 注意：MNN 会在推理中就地复用输入 buffer，因此每次 run 前必须重新喂数，
    // 否则第 2 次起读到的是被覆盖的中间结果（实测 rms 0.1855 -> 0.0425 -> 0.0259）。
    auto feedAll = [&]() {
        feedIfExists("phone", phone); feedIfExists("nsff0", nsff0); feedIfExists("pitch_emb", pe);
    };
    feedAll();

    // warmup + timing
    double best = 1e9, total = 0;
    for (int i = 0; i < reps; i++) {
        feedAll();
        auto s = std::chrono::steady_clock::now();
        net->runSession(session);
        auto e = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(e - s).count();
        total += ms; if (ms < best) best = ms;
    }
    fprintf(stderr, "run: %d reps, avg %.2f ms, best %.2f ms\n", reps, total / reps, best);

    if (getenv("RVC_DUMP_ALL")) {
        system("mkdir -p /tmp/mnndump");
        auto all = net->getSessionOutputAll(session);
        fprintf(stderr, "  total tensors: %zu\n", all.size());
        int cnt = 0;
        for (auto& kv : all) {
            MNN::Tensor* t = kv.second;
            if (!t) continue;
            MNN::Tensor host(t, t->getDimensionType());
            t->copyToHostTensor(&host);
            std::string safe = kv.first;
            for (auto& ch : safe) if (!isalnum((unsigned char)ch) && ch != '_' && ch != '-') ch = '_';
            char path[512]; snprintf(path, sizeof(path), "/tmp/mnndump/%s.raw", safe.c_str());
            FILE* fo = fopen(path, "wb");
            if (fo) {
                fwrite(host.host<float>(), 4, host.elementSize(), fo);
                fclose(fo);
                fprintf(stderr, "  T %s %d", kv.first.c_str(), (int)host.elementSize());
                for (int i = 0; i < t->dimensions(); i++) fprintf(stderr, " %d", t->length(i));
                fprintf(stderr, "\n");
            }
            cnt++;
        }
    }
    // 可选：按名字 dump 中间张量（RVC_DUMP="name1,name2"）
    if (const char* dumps = getenv("RVC_DUMP")) {
        std::string all(dumps), cur;
        size_t pos = 0;
        while (pos <= all.size()) {
            size_t c = all.find(',', pos);
            cur = all.substr(pos, (c == std::string::npos) ? std::string::npos : c - pos);
            if (!cur.empty()) {
                MNN::Tensor* t = net->getSessionOutput(session, cur.c_str());
                if (!t) { fprintf(stderr, "  dump %s: MISSING\n", cur.c_str()); }
                else {
                    MNN::Tensor host(t, t->getDimensionType());
                    t->copyToHostTensor(&host);
                    char path[256]; snprintf(path, sizeof(path), "/tmp/mnn_%s.raw", cur.c_str());
                    FILE* fo = fopen(path, "wb");
                    if (fo) { fwrite(host.host<float>(), 4, host.elementSize(), fo); fclose(fo); }
                    fprintf(stderr, "  dump %s: %zu floats -> %s\n", cur.c_str(), host.elementSize(), path);
                }
            }
            if (c == std::string::npos) break;
            pos = c + 1;
        }
    }
    auto out = net->getSessionOutput(session, "audio");
    if (!out && argc > 6) {
        // 子图模式：按第三个参数之外的名字取第一个可用输出
        auto outs = net->getSessionOutputAll(session);
        if (!outs.empty()) { out = outs.begin()->second; fprintf(stderr, "  using output: %s\n", outs.begin()->first.c_str()); }
    }
    if (!out) { fprintf(stderr, "output missing\n"); return 1; }
    MNN::Tensor outHost(out, out->getDimensionType());
    out->copyToHostTensor(&outHost);
    size_t n = outHost.elementSize();
    FILE* f = fopen(argv[6], "wb");
    fwrite(outHost.host<float>(), 4, n, f);
    fclose(f);
    float* p = outHost.host<float>();
    double rms = 0; float mx = 0;
    for (size_t i = 0; i < n; i++) { rms += p[i]*p[i]; if (fabsf(p[i]) > mx) mx = fabsf(p[i]); }
    fprintf(stderr, "out: %zu floats, rms %.4f, max %.4f -> %s\n", n, sqrt(rms/n), mx, argv[6]);
    return 0;
}
