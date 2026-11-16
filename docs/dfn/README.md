---
tags:
  - speech-enhancement
  - noise-cancellation
  - denoising
  - onnx
license: mit
library_name: onnxruntime
---

# DeepFilterNet3 ONNX

Verified neural-network export of the official
[DeepFilterNet3 v0.5.6](https://github.com/Rikorose/DeepFilterNet/tree/v0.5.6)
checkpoint for native 48 kHz speech enhancement.

This is the neural graph only. It expects normalized spectral features and
returns an ERB mask plus complex deep-filter coefficients. A compatible STFT,
feature-normalization, filtering, and synthesis implementation is required.

## Files

| File | Description |
|---|---|
| `deepfilter.onnx` | FP32 ONNX graph with a dynamic frame axis |
| `deepfilter-auxiliary.bin` | Canonical ERB matrices and Vorbis window for legacy consumers |
| `config.json` | Model and DSP configuration |
| `export_manifest.json` | Source checksum, artifact checksums, and ONNX parity results |

## Tensor contract

| Direction | Name | Shape | Type |
|---|---|---|---|
| input | `feat_erb` | `[1, 1, T, 32]` | float32 |
| input | `feat_spec` | `[1, 2, T, 96]` | float32 |
| output | `erb_mask` | `[1, 1, T, 32]` | float32 |
| output | `df_coefs` | `[1, 5, T, 96, 2]` | float32 |

The `feat_spec` channels are real then imaginary. The coefficient axes are
batch, filter order, frame, frequency, then real/imaginary.

## Required DSP contract

- sample rate: 48,000 Hz
- STFT: 960-point real FFT, 480-sample hop, Vorbis window
- spectrum: 481 bins; deep filtering: first 96 bins
- ERB bank: 32 disjoint bands with at least 2 FFT bins per band
- normalization: `tau=1.0`, rounded alpha `0.99`; ERB mean state linearly
  initialized from -60 dB to -90 dB and complex unit state from 0.001 to 0.0001
- deep filter: order 5, lookahead 2, applied from an immutable copy of the noisy
  spectrum before fusion with the ERB-masked spectrum
- batch/offline parity: append one 960-sample analysis tail and remove the
  480-sample STFT delay after overlap-add synthesis

`deepfilter-auxiliary.bin` is little-endian float32 data containing, in order,
the `[481,32]` forward ERB matrix, `[32,481]` inverse ERB matrix, and `[960]`
Vorbis window. New integrations can generate these deterministic constants.

## Verification

The export script checks the graph with ONNX, then compares ONNX Runtime CPU
outputs with PyTorch for several frame lengths. Exact errors and SHA-256 hashes
are recorded in `export_manifest.json`.

End-to-end validation used 20 VoiceBank-DEMAND clips (101.51 seconds total):

| Runtime | PESQ | STOI | SI-SDR |
|---|---:|---:|---:|
| official DeepFilterNet v0.5.6 | 2.9585 | 0.96064 | 18.228 dB |
| speech-core with this graph | 2.9602 | 0.96064 | 18.225 dB |

The speech-core output matched the official reference at 54.85 dB mean SI-SDR
with 0.000140 mean waveform RMSE. These values validate the tested native DSP
integration; other implementations must follow the contract above.

## Attribution and license

DeepFilterNet is by Hendrik Schröter and contributors and is dual-licensed
under Apache-2.0 or MIT. This repository publishes the converted model under
the MIT option. See the upstream project and papers for architecture and
training details.
