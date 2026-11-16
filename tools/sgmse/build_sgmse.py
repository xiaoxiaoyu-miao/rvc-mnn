#!/usr/bin/env python3
"""Build the SGMSE+ score network as ONNX using only torch + a JSON graph,
then serialize with rvc-onnx-web's dependency-free ONNX serializer.

Replicates sgmse/backbones/ncsnpp.py forward for the pinned NCSN++ config:
nf=128, ch_mult=(1,1,2,2,2,2,2), num_res_blocks=2, attn_resolutions=(16,),
image_size=256, progressive input/output skip, FIR up/down sampling.
Fixed frame count (default 256) and fixed FFT bins (256).
"""
import argparse, json, math, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "sgmse_src"))
from sgmse.backbones.ncsnpp import NCSNpp
from sgmse.backbones import ncsnpp_utils as U
from sgmse.backbones.ncsnpp_utils import layers, layerspp
import sgmse.backbones.ncsnpp_utils.up_or_down_sampling as uds

BINS = 256


class V:
    __slots__ = ("name", "shape")

    def __init__(self, name, shape):
        self.name = name
        self.shape = tuple(int(v) for v in shape)

    def __repr__(self):
        return f"{self.name}{self.shape}"


class Builder:
    def __init__(self, frames):
        self.frames = frames
        self.nodes = []
        self.inits = []          # metadata; weights appended to blob
        self.blob = bytearray()
        self.n = 0
        self.cache = {}

    def name(self, prefix):
        self.n += 1
        return f"{prefix}_{self.n}"

    def init_tensor(self, key, tensor, dtype="float32"):
        """Add an initializer once per (key,dtype); return its name."""
        ck = (key, dtype)
        if ck in self.cache:
            return self.cache[ck]
        t = tensor.detach().cpu().contiguous()
        if dtype == "float32":
            arr = t.to(torch.float32).numpy().astype("<f4", copy=False)
        elif dtype == "int64":
            arr = t.to(torch.int64).numpy().astype("<i8", copy=False)
        else:
            raise ValueError(dtype)
        off = len(self.blob)
        self.blob.extend(arr.tobytes())
        name = self.name("w")
        self.inits.append({"name": name, "shape": list(arr.shape),
                           "dtype": dtype, "offset": off,
                           "count": int(arr.size)})
        self.cache[ck] = name
        return name

    def init_value(self, key, value, dtype="float32"):
        return self.init_tensor(key, torch.tensor(value), dtype)

    def node(self, op, ins, out_shape, **attrs):
        out = self.name(op.lower())
        self.nodes.append({"opType": op, "name": out,
                           "inputs": [i.name if isinstance(i, V) else i for i in ins],
                           "outputs": [out], "attrs": attrs})
        return V(out, out_shape)

    # ------------------------------------------------------------------ ops
    def conv(self, x, w, b=None, stride=1, pad=1, dil=1, groups=1):
        wn = self.init_tensor(("conv", id(w)), w)
        ins = [x, wn]
        if b is not None:
            ins.append(self.init_tensor(("bias", id(b)), b))
        H, W = x.shape[2], x.shape[3]
        kH, kW = w.shape[2], w.shape[3]
        oH = (H + 2 * pad - dil * (kH - 1) - 1) // stride + 1
        oW = (W + 2 * pad - dil * (kW - 1) - 1) // stride + 1
        return self.node("Conv", ins, (x.shape[0], w.shape[0], oH, oW),
                         strides=[stride, stride], pads=[pad, pad, pad, pad],
                         dilations=[dil, dil], group=groups)

    def linear(self, x, w, b):
        return self.node("Gemm", [x, self.init_tensor(("lin", id(w)), w),
                                  self.init_tensor(("linb", id(b)), b)],
                         (x.shape[0], w.shape[0]), transB=1)

    def transpose(self, x, perm):
        return self.node("Transpose", [x], [x.shape[p] for p in perm], perm=perm)

    def reshape(self, x, shape):
        sname = self.init_value(("shape", tuple(int(v) for v in shape)), shape, "int64")
        return self.node("Reshape", [x, sname], shape)

    def add(self, a, b, out_shape):
        return self.node("Add", [a, b], out_shape)

    def mul(self, a, b, out_shape):
        return self.node("Mul", [a, b], out_shape)

    def sub(self, a, b, out_shape):
        return self.node("Sub", [a, b], out_shape)

    def div(self, a, b, out_shape):
        return self.node("Div", [a, b], out_shape)

    def neg(self, a, out_shape):
        return self.node("Neg", [a], out_shape)

    def concat(self, xs, axis, out_shape):
        return self.node("Concat", xs, out_shape, axis=axis)

    def silu(self, x):
        s = self.node("Sigmoid", [x], x.shape)
        return self.mul(x, s, x.shape)

    def group_norm(self, x, m):
        N, C, H, W = x.shape
        G = m.num_groups
        eps = float(m.eps)
        x5 = self.reshape(x, (N, G, C // G, H, W))
        mean = self.node("ReduceMean", [x5], (N, G, 1, 1, 1), axes=[2, 3, 4], keepdims=1)
        d = self.sub(x5, mean, x5.shape)
        d2 = self.mul(d, d, x5.shape)
        var = self.node("ReduceMean", [d2], (N, G, 1, 1, 1), axes=[2, 3, 4], keepdims=1)
        ve = self.add(var, self.init_value(("gn_eps", eps), eps), var.shape)
        sd = self.node("Sqrt", [ve], ve.shape)
        y = self.div(d, sd, x5.shape)
        y = self.reshape(y, (N, C, H, W))
        g = self.reshape(self.init_tensor(("gn_g", id(m.weight)), m.weight), (1, C, 1, 1))
        b = self.reshape(self.init_tensor(("gn_b", id(m.bias)), m.bias), (1, C, 1, 1))
        y = self.mul(y, g, (N, C, H, W))
        return self.add(y, b, (N, C, H, W))

    def nin(self, x, m):
        N, C, H, W = x.shape
        Cout = m.W.shape[1]
        xt = self.transpose(x, (0, 2, 3, 1))
        w = self.init_tensor(("nin", id(m.W)), m.W)
        y = self.node("MatMul", [xt, w], (N, H, W, Cout))
        b = self.init_tensor(("ninb", id(m.b)), m.b)
        y = self.add(y, b, (N, H, W, Cout))
        return self.transpose(y, (0, 3, 1, 2))

    def attention(self, x, m):
        N, C, H, W = x.shape
        h = self.group_norm(x, m.GroupNorm_0)
        q = self.nin(h, m.NIN_0)
        k = self.nin(h, m.NIN_1)
        v = self.nin(h, m.NIN_2)
        q2 = self.reshape(self.transpose(q, (0, 2, 3, 1)), (N, H * W, C))
        k2 = self.transpose(self.reshape(self.transpose(k, (0, 2, 3, 1)), (N, H * W, C)),
                            (0, 2, 1))
        w = self.node("MatMul", [q2, k2], (N, H * W, H * W))
        scale = self.init_value(("attn_scale", C), float(C) ** -0.5)
        w = self.mul(w, scale, w.shape)
        w = self.node("Softmax", [w], w.shape, axis=-1)
        v2 = self.transpose(self.reshape(v, (N, C, H * W)), (0, 2, 1))
        o = self.node("MatMul", [w, v2], (N, H * W, C))
        o = self.reshape(self.transpose(o, (0, 2, 1)), (N, C, H, W))
        o = self.nin(o, m.NIN_3)
        if not m.skip_rescale:
            return self.add(x, o, x.shape)
        s = self.init_value(("sqrt2",), math.sqrt(2.0))
        return self.div(self.add(x, o, x.shape), s, x.shape)

    # ------------------------------------------------------------ fir resample
    def _fir_kernel(self, k, gain):
        k = np.asarray(k, dtype=np.float32)
        k = np.outer(k, k)
        k /= np.sum(k)
        return k * gain

    def _upfirdn(self, x, k, up=1, down=1, pad=(0, 0)):
        N, C, H, W = x.shape
        kH, kW = k.shape
        p0, p1 = pad
        NC = N * C
        # input -> [NC, H, W, 1]
        xv = self.reshape(x, (NC, H, W, 1))
        # view [NC,H,1,W,1,1] + zero insertion
        xv = self.reshape(xv, (NC, H, 1, W, 1, 1))
        pads6 = [0, 0, 0, 0, 0, 0,
                 0, 0, up - 1, 0, up - 1, 0]
        xv = self.node("Pad", [xv, self.init_value(("pad6", tuple(pads6)), pads6, "int64")],
                       (NC, H, up, W, up, 1), mode="constant")
        xv = self.reshape(xv, (NC, H * up, W * up, 1))
        pads4 = [0, max(p0, 0), max(p0, 0), 0,
                 0, max(p1, 0), max(p1, 0), 0]
        xv = self.node("Pad", [xv, self.init_value(("pad4", tuple(pads4)), pads4, "int64")],
                       (NC, H * up + max(p0, 0) + max(p1, 0),
                        W * up + max(p0, 0) + max(p1, 0), 1), mode="constant")
        if p0 < 0 or p1 < 0:
            sy0, sy1 = max(-p0, 0), max(-p1, 0)
            st = self.init_value(("sl_st", sy0), [sy0, sy0], "int64")
            hbig = H * up + max(p0, 0) + max(p1, 0)
            wbig = W * up + max(p0, 0) + max(p1, 0)
            en = self.init_value(("sl_en", hbig, wbig, sy1),
                                 [hbig - sy1, wbig - sy1], "int64")
            ax = self.init_value(("sl_ax",), [1, 2], "int64")
            xv = self.node("Slice", [xv, st, en, ax],
                           (NC, H * up + p0 + p1, W * up + p0 + p1, 1))
        Hp = H * up + p0 + p1
        Wp = W * up + p0 + p1
        xv = self.transpose(xv, (0, 3, 1, 2))
        xv = self.reshape(xv, (NC, 1, Hp, Wp))
        kf = np.ascontiguousarray(np.flip(k, (0, 1))).reshape(1, 1, kH, kW)
        xv = self.node("Conv", [xv, self.init_tensor(("fir", tuple(np.asarray(k).reshape(-1).tolist())), torch.from_numpy(kf))],
                       (NC, 1, Hp - kH + 1, Wp - kW + 1))
        Hc, Wc = Hp - kH + 1, Wp - kW + 1
        xv = self.reshape(xv, (NC, 1, Hc, Wc))
        xv = self.transpose(xv, (0, 2, 3, 1))
        if down > 1:
            st = self.init_value(("ds_st",), [0, 0], "int64")
            en = self.init_value(("ds_en", Hc, Wc), [Hc, Wc], "int64")
            ax = self.init_value(("ds_ax",), [1, 2], "int64")
            sp = self.init_value(("ds_step", down), [down, down], "int64")
            ho = (Hc - 1) // down + 1
            wo = (Wc - 1) // down + 1
            xv = self.node("Slice", [xv, st, en, ax, sp], (NC, ho, wo, 1))
            Hc, Wc = ho, wo
        return self.reshape(xv, (N, C, Hc, Wc))

    def upsample2d(self, x):
        k = self._fir_kernel([1, 3, 3, 1], 4.0)
        p = k.shape[0] - 2
        return self._upfirdn(x, k, up=2, down=1, pad=((p + 1) // 2 + 1, p // 2))

    def downsample2d(self, x):
        k = self._fir_kernel([1, 3, 3, 1], 1.0)
        p = k.shape[0] - 2
        return self._upfirdn(x, k, up=1, down=2, pad=((p + 1) // 2, p // 2))

# ------------------------------------------------------------------ forward
def resblock(b, m, x, temb):
    h = b.silu(b.group_norm(x, m.GroupNorm_0))
    x2 = x
    if m.up:
        if not m.fir:
            raise NotImplementedError("naive upsample not needed for pinned config")
        h = b.upsample2d(h)
        x2 = b.upsample2d(x2)
    elif m.down:
        if not m.fir:
            raise NotImplementedError("naive downsample not needed for pinned config")
        h = b.downsample2d(h)
        x2 = b.downsample2d(x2)
    h = b.conv(h, m.Conv_0.weight, m.Conv_0.bias, pad=1)
    if temb is not None:
        t = b.linear(b.silu(temb), m.Dense_0.weight, m.Dense_0.bias)
        t = b.reshape(t, (t.shape[0], t.shape[1], 1, 1))
        h = b.add(h, t, h.shape)
    h = b.silu(b.group_norm(h, m.GroupNorm_1))
    h = b.conv(h, m.Conv_1.weight, m.Conv_1.bias, pad=1)
    if hasattr(m, "Conv_2"):
        x2 = b.conv(x2, m.Conv_2.weight, m.Conv_2.bias, pad=0)
    s = b.add(x2, h, h.shape)
    if m.skip_rescale:
        s = b.div(s, b.init_value(("sqrt2",), math.sqrt(2.0)), s.shape)
    return s


def combine(b, m, x, y):
    h = b.conv(x, m.Conv_0.weight, m.Conv_0.bias, pad=0)
    if m.method == "cat":
        return b.concat([h, y], 1, (h.shape[0], h.shape[1] + y.shape[1], h.shape[2], h.shape[3]))
    return b.add(h, y, h.shape)


def build_forward(m, frames):
    b = Builder(frames)
    x = V("features", (1, 4, BINS, frames))
    t = V("time", (1,))
    modules = list(m.all_modules)
    idx = 0

    # Gaussian fourier embedding of log(t)
    W = modules[idx].W
    idx += 1
    logt = b.node("Log", [t], (1,))
    wn = b.init_tensor(("gauss_w", id(W)), W)
    xp = b.mul(b.reshape(logt, (1, 1)), b.reshape(wn, (1, W.numel())), (1, W.numel()))
    xp = b.mul(xp, b.init_value(("2pi",), 2 * math.pi), xp.shape)
    temb = b.concat([b.node("Sin", [xp], xp.shape),
                     b.node("Cos", [xp], xp.shape)], 1, (1, 2 * W.numel()))
    temb = b.linear(temb, modules[idx].weight, modules[idx].bias)
    idx += 1
    temb = b.silu(temb)
    temb = b.linear(temb, modules[idx].weight, modules[idx].bias)
    idx += 1

    hs = [b.conv(x, modules[idx].weight, modules[idx].bias, pad=1)]
    idx += 1
    input_pyramid = x

    # down path
    for i_level in range(m.num_resolutions):
        for _ in range(m.num_res_blocks):
            h = resblock(b, modules[idx], hs[-1], temb)
            idx += 1
            if h.shape[2] in m.attn_resolutions:
                h = b.attention(h, modules[idx])
                idx += 1
            hs.append(h)
        if i_level != m.num_resolutions - 1:
            h = resblock(b, modules[idx], hs[-1], temb)
            idx += 1
            if m.progressive_input == "input_skip":
                input_pyramid = b.downsample2d(input_pyramid)
                h = combine(b, modules[idx], input_pyramid, h)
                idx += 1
            elif m.progressive_input != "none":
                raise NotImplementedError(m.progressive_input)
            hs.append(h)

    h = resblock(b, modules[idx], hs[-1], temb)
    idx += 1
    h = b.attention(h, modules[idx])
    idx += 1
    h = resblock(b, modules[idx], h, temb)
    idx += 1

    # up path
    pyramid = None
    for i_level in reversed(range(m.num_resolutions)):
        for _ in range(m.num_res_blocks + 1):
            skip = hs.pop()
            h = b.concat([h, skip], 1, (h.shape[0], h.shape[1] + skip.shape[1],
                                        h.shape[2], h.shape[3]))
            h = resblock(b, modules[idx], h, temb)
            idx += 1
        if h.shape[2] in m.attn_resolutions:
            h = b.attention(h, modules[idx])
            idx += 1
        if m.progressive == "output_skip":
            gn = modules[idx]
            idx += 1
            conv = modules[idx]
            idx += 1
            if i_level == m.num_resolutions - 1:
                pyramid = b.conv(b.silu(b.group_norm(h, gn)), conv.weight, conv.bias, pad=1)
            else:
                pyramid = b.upsample2d(pyramid)
                pyramid = b.add(pyramid,
                                b.conv(b.silu(b.group_norm(h, gn)), conv.weight, conv.bias, pad=1),
                                pyramid.shape)
        else:
            raise NotImplementedError(m.progressive)
        if i_level != 0:
            h = resblock(b, modules[idx], h, temb)
            idx += 1

    h = pyramid
    t4 = b.reshape(t, (1, 1, 1, 1))
    h = b.div(h, t4, h.shape)
    h = b.conv(h, m.output_layer.weight, m.output_layer.bias, pad=0)
    score = b.neg(h, h.shape)
    if idx != len(modules):
        raise SystemExit(f"module index mismatch: consumed {idx}, have {len(modules)}")
    return b, x, t, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=256)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-weights", required=True)
    args = ap.parse_args()

    m = NCSNpp().eval()
    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    m.load_state_dict({k[4:]: v for k, v in sd.items() if k.startswith("dnn.")}, strict=True)
    del sd

    b, x, t, score = build_forward(m, args.frames)
    meta = {
        "frames": args.frames,
        "inputs": [{"name": "features", "shape": list(x.shape)},
                   {"name": "time", "shape": list(t.shape)}],
        "outputs": [{"name": score.name, "shape": list(score.shape)}],
        "initializers": b.inits,
        "nodes": b.nodes,
    }
    with open(args.out_json, "w") as f:
        json.dump(meta, f)
    with open(args.out_weights, "wb") as f:
        f.write(b.blob)
    print(f"nodes={len(b.nodes)} initializers={len(b.inits)} weights={len(b.blob)/1e6:.1f}MB")


if __name__ == "__main__":
    main()
