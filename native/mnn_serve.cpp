// mnn_serve: 常驻 MNN 推理服务，stdin 逐行接收 "out=<path> [name=<path> ...]"，跑一次推理后回一行 "ok <ms>"
// 避免每块重建 session，等价于 APK 内的常驻推理循环
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>
#include <map>
#include <sstream>
#include <chrono>
#include <iostream>

static std::vector<float> readRaw(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return {};
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<float> v(sz / 4);
    size_t n = fread(v.data(), 4, v.size(), f);
    fclose(f); v.resize(n);
    return v;
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s model.mnn backend\n", argv[0]); return 1; }
    std::shared_ptr<MNN::Interpreter> net(MNN::Interpreter::createFromFile(argv[1]));
    if (!net) { fprintf(stderr, "load fail\n"); return 1; }
    MNN::ScheduleConfig cfg;
    std::string be = argv[2];
    if (be == "vulkan") cfg.type = MNN_FORWARD_VULKAN;
    else if (be == "opencl") cfg.type = MNN_FORWARD_OPENCL;
    else if (be == "qnn") cfg.type = MNN_FORWARD_NN;
    else cfg.type = MNN_FORWARD_CPU;
    cfg.numThread = 4;
    if (const char* th = getenv("RVC_THREADS")) cfg.numThread = atoi(th);
    MNN::BackendConfig bc;
    const char* pe = getenv("RVC_PREC"); int prec = pe ? atoi(pe) : 1;
    bc.precision = prec <= 0 ? MNN::BackendConfig::Precision_Low
                 : prec == 1 ? MNN::BackendConfig::Precision_Normal
                             : MNN::BackendConfig::Precision_High;
    bc.power = MNN::BackendConfig::Power_High;
    cfg.backendConfig = &bc;
    MNN::Session* session = net->createSession(cfg);
    if (!session) { fprintf(stderr, "session fail\n"); return 1; }
    fprintf(stderr, "READY\n"); fflush(stderr);

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line == "quit") break;
        std::map<std::string, std::string> kv;
        std::istringstream iss(line); std::string tok;
        while (iss >> tok) { auto p = tok.find('='); if (p != std::string::npos) kv[tok.substr(0,p)] = tok.substr(p+1); }
        if (kv.find("out") == kv.end()) { printf("err noout\n"); fflush(stdout); continue; }
        // 喂输入
        bool feedErr = false;
        std::map<std::string, std::vector<float>> buf;
        for (auto& it : kv) {
            if (it.first == "out") continue;
            auto data = readRaw(it.second);
            if (data.empty()) { feedErr = true; break; }
            buf[it.first] = data;
        }
        if (feedErr) { printf("err feed\n"); fflush(stdout); continue; }
        // 动态 shape：唯一 -1 维按数据量推断
        bool doResize = false;
        for (auto& it : buf) {
            MNN::Tensor* t = net->getSessionInput(session, it.first.c_str());
            if (!t) continue;
            std::vector<int> shp = t->shape();
            int nDyn = 0, dynIdx = -1;
            long known = 1;
            for (size_t i = 0; i < shp.size(); i++) {
                if (shp[i] < 0) { nDyn++; dynIdx = (int)i; } else known *= shp[i];
            }
            if (nDyn != 1 || known <= 0 || (long)it.second.size() % known != 0) continue;
            std::vector<int> want = shp;
            want[dynIdx] = (int)((long)it.second.size() / known);
            net->resizeTensor(t, want);
            doResize = true;
        }
        if (doResize) net->resizeSession(session);
        for (auto& it : buf) {
            MNN::Tensor* t = net->getSessionInput(session, it.first.c_str());
            if (!t) continue;
            auto& data = it.second;
            MNN::Tensor host(t, t->getDimensionType());
            size_t n = host.elementSize() < data.size() ? host.elementSize() : data.size();
            memcpy(host.host<float>(), data.data(), n * 4);
            t->copyFromHostTensor(&host);
        }
        auto s = std::chrono::steady_clock::now();
        net->runSession(session);
        auto e = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(e - s).count();
        auto outs = net->getSessionOutputAll(session);
        if (outs.empty()) { printf("err nooutput\n"); fflush(stdout); continue; }
        MNN::Tensor* out = outs.begin()->second;
        MNN::Tensor outHost(out, out->getDimensionType());
        out->copyToHostTensor(&outHost);
        FILE* f = fopen(kv["out"].c_str(), "wb");
        if (f) { fwrite(outHost.host<float>(), 4, outHost.elementSize(), f); fclose(f); }
        printf("ok %.2f %zu\n", ms, outHost.elementSize()); fflush(stdout);
    }
    return 0;
}
