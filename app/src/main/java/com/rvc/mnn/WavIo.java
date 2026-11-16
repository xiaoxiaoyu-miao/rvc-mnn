package com.rvc.mnn;

import java.io.Closeable;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * 离线录音模式用的 WAV 读写。
 *
 * <p>从 MainActivity 里把 {@code readWav/resample/writeWav} 抽出来，供
 * 「录音模式」复用，避免两处各写一份解析代码走偏。
 */
public final class WavIo {

    private WavIo() {
    }

    /** 读 WAV（支持 8/16/32-bit，多声道自动平均成单声道）。结果范围 -1..1。 */
    public static float[] readWav(String path, int[] outSr) throws Exception {
        try (RandomAccessFile f = new RandomAccessFile(path, "r")) {
            byte[] hdr = new byte[12];
            f.readFully(hdr);
            if (!(hdr[0] == 'R' && hdr[1] == 'I' && hdr[2] == 'F' && hdr[3] == 'F')) {
                throw new Exception("not wav");
            }
            int sr = 16000, ch = 1, bits = 16;
            byte[] data = null;
            long pos = 12;
            long flen = f.length();
            while (pos + 8 <= flen) {
                f.seek(pos);
                byte[] ch8 = new byte[8];
                f.readFully(ch8);
                String id = new String(ch8, 0, 4, "ASCII");
                int sz = ByteBuffer.wrap(ch8, 4, 4).order(ByteOrder.LITTLE_ENDIAN).getInt();
                if (sz < 0) break;
                long next = pos + 8 + sz + (sz & 1);
                if (id.equals("fmt ") && sz >= 16) {
                    byte[] fmt = new byte[Math.min(sz, 32)];
                    f.readFully(fmt);
                    ch = ByteBuffer.wrap(fmt, 2, 2).order(ByteOrder.LITTLE_ENDIAN).getShort();
                    sr = ByteBuffer.wrap(fmt, 4, 4).order(ByteOrder.LITTLE_ENDIAN).getInt();
                    bits = ByteBuffer.wrap(fmt, 14, 2).order(ByteOrder.LITTLE_ENDIAN).getShort();
                } else if (id.equals("data")) {
                    int avail = (int) Math.min((long) sz, flen - pos - 8);
                    data = new byte[Math.max(0, avail)];
                    if (avail > 0) f.readFully(data);
                    break;
                }
                pos = next;
            }
            if (data == null) throw new Exception("no data chunk");
            int bytesPer = bits / 8;
            if (bytesPer <= 0) throw new Exception("bad bits: " + bits);
            int frames = data.length / (bytesPer * ch);
            float[] out = new float[frames];
            ByteBuffer bb = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
            for (int i = 0; i < frames; i++) {
                float s = 0;
                for (int c = 0; c < ch; c++) {
                    if (bits == 16) s += bb.getShort() / 32768f;
                    else if (bits == 8) s += (bb.get() - 128) / 128f;
                    else s += bb.getInt() / 2147483648f;
                }
                out[i] = s / ch;
            }
            outSr[0] = sr;
            return out;
        }
    }

    /** 线性插值重采样。 */
    public static float[] resample(float[] x, int from, int to) {
        if (from == to) return x;
        int n = (int) Math.round(x.length * (double) to / from);
        float[] y = new float[n];
        for (int i = 0; i < n; i++) {
            double idx = i * (double) from / to;
            int i0 = (int) idx, i1 = Math.min(i0 + 1, x.length - 1);
            float fr = (float) (idx - i0);
            y[i] = x[i0] * (1 - fr) + x[i1] * fr;
        }
        return y;
    }

    /** 写 16-bit 单声道 WAV，顺便乘上增益。 */
    public static void writeWav(String path, float[] y, int sr, float gain) throws Exception {
        int n = y.length;
        ByteBuffer bb = ByteBuffer.allocate(44 + n * 2).order(ByteOrder.LITTLE_ENDIAN);
        bb.put("RIFF".getBytes()); bb.putInt(36 + n * 2); bb.put("WAVE".getBytes());
        bb.put("fmt ".getBytes()); bb.putInt(16); bb.putShort((short) 1); bb.putShort((short) 1);
        bb.putInt(sr); bb.putInt(sr * 2); bb.putShort((short) 2); bb.putShort((short) 16);
        bb.put("data".getBytes()); bb.putInt(n * 2);
        for (float v : y) {
            float g = v * gain;
            int s = (int) Math.max(-32768, Math.min(32767, Math.round(g * 32767f)));
            bb.putShort((short) s);
        }
        try (FileOutputStream fo = new FileOutputStream(path)) {
            fo.write(bb.array());
        }
    }

    /**
     * 边录边写 WAV：先占位 44 字节头，收尾时回填 RIFF/data 长度。
     * 这样录制过程中断电也只丢最后一点，不会整段作废。
     */
    public static final class Writer implements Closeable {
        private final RandomAccessFile raf;
        private final int sr;
        private long dataBytes;
        private boolean closed;

        public Writer(File f, int sr) throws IOException {
            this.sr = sr;
            this.raf = new RandomAccessFile(f, "rw");
            this.raf.setLength(0);
            this.raf.write(new byte[44]);
            this.raf.seek(44);
        }

        public void write(short[] s, int n) throws IOException {
            if (closed || n <= 0) return;
            byte[] b = new byte[n * 2];
            for (int i = 0; i < n; i++) {
                b[2 * i] = (byte) (s[i] & 0xFF);
                b[2 * i + 1] = (byte) ((s[i] >> 8) & 0xFF);
            }
            raf.write(b, 0, n * 2);
            dataBytes += n * 2L;
        }

        /** 已录秒数。 */
        public double seconds() {
            return dataBytes / 2.0 / sr;
        }

        @Override
        public void close() {
            if (closed) return;
            closed = true;
            try {
                ByteBuffer h = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN);
                h.put("RIFF".getBytes()); h.putInt((int) (36 + dataBytes)); h.put("WAVE".getBytes());
                h.put("fmt ".getBytes()); h.putInt(16); h.putShort((short) 1); h.putShort((short) 1);
                h.putInt(sr); h.putInt(sr * 2); h.putShort((short) 2); h.putShort((short) 16);
                h.put("data".getBytes()); h.putInt((int) dataBytes);
                raf.seek(0);
                raf.write(h.array());
            } catch (Exception ignored) {
            }
            try {
                raf.close();
            } catch (Exception ignored) {
            }
        }
    }
}
