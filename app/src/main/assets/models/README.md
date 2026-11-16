# RVC 权重不放仓库

`hubert_fp16.mnn`（181 MB）和 `ct_static.mnn`（105 MB）超过 GitHub 单文件限制，
`fcpe_audio.onnx`（43 MB）虽然能传，但为了仓库体积也一起省略了。

## 两种放法

1. **放到手机模型目录**（推荐，App 会优先用这里，不动内置那份）：
   ```bash
   mkdir -p /sdcard/models/mm
   # 把下面四个文件拷进去
   ```
2. **放到本目录随 APK 打包**（需要与 `ModelBootstrap.SHA256` 里的哈希一致）：
   ```
   app/src/main/assets/models/hubert_fp16.mnn
   app/src/main/assets/models/ct_static.mnn
   app/src/main/assets/models/fcpe_audio.onnx
   app/src/main/assets/models/emb_pitch.bin
   ```

## 四个文件

| 文件 | 说明 | 大小 |
| --- | --- | --- |
| `hubert_fp16.mnn` | hubert 内容编码器（fp16） | ~181 MB |
| `ct_static.mnn` | RVC 合成器（静态 49 帧） | ~105 MB |
| `fcpe_audio.onnx` | f0 提取 | ~43 MB |
| `emb_pitch.bin` | 音高编码查找表 | ~192 KB |

DFN3 降噪模型不走这里：它内置在 `app/src/main/assets/dfn/`。
SGMSE+ Diffusion 模型默认找 `<模型目录>/sgmse.onnx`（约 262 MB，生成方法见 `tools/sgmse/README.md`）。
