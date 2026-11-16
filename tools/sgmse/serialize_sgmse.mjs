import fs from 'node:fs';

// 指向本机 rvc-onnx-web 的 dist/onnx-serializer.js（可用环境变量覆盖）
const serializerPath = process.env.RVC_ONNX_WEB
  || '/data/data/com.termux/files/home/rvc-onnx-web/dist/onnx-serializer.js';
const { serializeOnnx } = await import(serializerPath);

const [,, jsonPath, weightsPath, outPath] = process.argv;
const meta = JSON.parse(fs.readFileSync(jsonPath, 'utf8'));
const buf = fs.readFileSync(weightsPath);

function typed(init) {
  const bytes = buf.subarray(init.offset, init.offset + init.count * (init.dtype === 'int64' ? 8 : 4));
  const copy = new Uint8Array(bytes); // own aligned ArrayBuffer
  if (init.dtype === 'int64') {
    return { data: new BigInt64Array(copy.buffer), shape: init.shape, dtype: 'int64' };
  }
  return { data: new Float32Array(copy.buffer), shape: init.shape, dtype: 'float32' };
}

function attrs(obj) {
  const out = [];
  for (const [name, v] of Object.entries(obj || {})) {
    if (Array.isArray(v)) {
      out.push({ name, type: 'INTS', intsValue: v.map(x => BigInt(x)) });
    } else if (typeof v === 'string') {
      out.push({ name, type: 'STRING', stringValue: v });
    } else if (Number.isInteger(v)) {
      out.push({ name, type: 'INT', intValue: BigInt(v) });
    } else {
      out.push({ name, type: 'FLOAT', floatValue: v });
    }
  }
  return out;
}

const model = {
  irVersion: 8n,
  opsetImports: [{ domain: '', version: 17n }],
  producerName: 'sgmse-onnx-builder',
  producerVersion: '1.0',
  graph: {
    name: 'sgmse_plus_score',
    nodes: meta.nodes.map(n => ({
      opType: n.opType, name: n.name, inputs: n.inputs, outputs: n.outputs,
      attributes: attrs(n.attrs),
    })),
    inputs: meta.inputs.map(i => ({
      name: i.name, elemType: 1,
      shape: i.shape.map(d => ({ dimValue: BigInt(d) })),
    })),
    outputs: meta.outputs.map(o => ({
      name: o.name, elemType: 1,
      shape: o.shape.map(d => ({ dimValue: BigInt(d) })),
    })),
    initializers: meta.initializers.map(i => ({ name: i.name, data: typed(i) })),
  },
};

fs.writeFileSync(outPath, serializeOnnx(model));
console.log('wrote', outPath, fs.statSync(outPath).size);
