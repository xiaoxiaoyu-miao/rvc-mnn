package com.rvc.mnn;

import android.content.Context;
import android.media.AudioFormat;
import android.media.MediaCodec;
import android.media.MediaExtractor;
import android.media.MediaFormat;
import android.net.Uri;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * 把「随便一个音频文件」变成 RVC 能吃的 WAV。
 *
 * <p>录音模式原来只有麦克风一条输入路径，用户想拿一段现成的语音（微信/QQ 保存的
 * aac、m4a、mp3、amr、flac…）来变声就没门。这里用系统自带的
 * {@link MediaExtractor} + {@link MediaCodec} 解成 16-bit 单声道 WAV：
 * <ul>
 *   <li>输入本来就是 WAV 时**直接原样拷贝**，不经过解码器（零损失、最快）；</li>
 *   <li>其它格式解成**源采样率**的单声道 WAV，后面的 {@link OfflineConvert}
 *       照旧做 16 kHz 重采样，链路其余部分不用动；</li>
 *   <li>多声道按算术平均下混成单声道（RVC 只吃单声道）。</li>
 * </ul>
 */
public final class AudioImport {

    private AudioImport() {
    }

    /** 极简进度回调。 */
    public interface Progress {
        void onProgress(int percent);
    }

    /**
     * @param uri 文件选择器给的 Uri（content:// 或 file://）
     * @param dst 目标 WAV（会被覆盖）
     * @return 目标文件时长（秒）
     */
    public static double importToWav(Context ctx, Uri uri, File dst, Progress cb) throws Exception {
        return importToWav(ctx, (Object) uri, dst, cb);
    }

    /** 同上，但直接给文件路径（命令行 / 内部调用用，免去 SAF 的 Uri）。 */
    public static double importToWav(Context ctx, String path, File dst, Progress cb) throws Exception {
        return importToWav(ctx, (Object) path, dst, cb);
    }

    private static double importToWav(Context ctx, Object src, File dst, Progress cb) throws Exception {
        if (copyIfWav(ctx, src, dst)) {
            int[] sr = new int[1];
            float[] a = WavIo.readWav(dst.getAbsolutePath(), sr);
            if (cb != null) cb.onProgress(100);
            return a.length / (double) Math.max(1, sr[0]);
        }
        return decodeToWav(ctx, src, dst, cb);
    }

    /** 打开来源（Uri 或路径）。 */
    private static InputStream open(Context ctx, Object src) throws Exception {
        if (src instanceof Uri) return ctx.getContentResolver().openInputStream((Uri) src);
        return new FileInputStream((String) src);
    }

    /** 文件是 RIFF/WAVE 就直接整段拷贝到 dst，返回 true。 */
    private static boolean copyIfWav(Context ctx, Object src, File dst) throws Exception {
        try (InputStream in = open(ctx, src)) {
            if (in == null) throw new Exception("打不开这个文件");
            byte[] hdr = new byte[12];
            int got = 0;
            while (got < 12) {
                int n = in.read(hdr, got, 12 - got);
                if (n <= 0) break;
                got += n;
            }
            boolean isWav = got == 12
                    && hdr[0] == 'R' && hdr[1] == 'I' && hdr[2] == 'F' && hdr[3] == 'F'
                    && hdr[8] == 'W' && hdr[9] == 'A' && hdr[10] == 'V' && hdr[11] == 'E';
            if (!isWav) return false;
            try (FileOutputStream fo = new FileOutputStream(dst)) {
                fo.write(hdr, 0, got);
                byte[] buf = new byte[65536];
                int n;
                while ((n = in.read(buf)) > 0) fo.write(buf, 0, n);
            }
            return true;
        }
    }

    /** 用系统解码器把任意音频解成 16-bit 单声道 WAV（采样率保持源采样率）。 */
    private static double decodeToWav(Context ctx, Object src, File dst, Progress cb) throws Exception {
        MediaExtractor ex = new MediaExtractor();
        MediaCodec codec = null;
        WavIo.Writer w = null;
        try {
            if (src instanceof Uri) {
                ex.setDataSource(ctx, (Uri) src, null);
            } else {
                ex.setDataSource((String) src);
            }
            int track = -1;
            MediaFormat fmt = null;
            for (int i = 0; i < ex.getTrackCount(); i++) {
                MediaFormat f = ex.getTrackFormat(i);
                String mime = f.getString(MediaFormat.KEY_MIME);
                if (mime != null && mime.startsWith("audio/")) {
                    track = i;
                    fmt = f;
                    break;
                }
            }
            if (track < 0 || fmt == null) throw new Exception("这个文件里没有音频轨");
            ex.selectTrack(track);

            String mime = fmt.getString(MediaFormat.KEY_MIME);
            int srcSr = fmt.containsKey(MediaFormat.KEY_SAMPLE_RATE)
                    ? fmt.getInteger(MediaFormat.KEY_SAMPLE_RATE) : 16000;
            int srcCh = fmt.containsKey(MediaFormat.KEY_CHANNEL_COUNT)
                    ? fmt.getInteger(MediaFormat.KEY_CHANNEL_COUNT) : 1;
            long durUs = fmt.containsKey(MediaFormat.KEY_DURATION)
                    ? fmt.getLong(MediaFormat.KEY_DURATION) : 0L;

            codec = MediaCodec.createDecoderByType(mime);
            codec.configure(fmt, null, null, 0);
            codec.start();

            int sr = srcSr > 0 ? srcSr : 16000;
            int ch = Math.max(1, srcCh);
            int pcmEnc = AudioFormat.ENCODING_PCM_16BIT;
            if (fmt.containsKey(MediaFormat.KEY_PCM_ENCODING)) {
                pcmEnc = fmt.getInteger(MediaFormat.KEY_PCM_ENCODING);
            }
            w = new WavIo.Writer(dst, sr);

            MediaCodec.BufferInfo info = new MediaCodec.BufferInfo();
            boolean inDone = false, outDone = false;
            long fed = 0;
            final short[] mono = new short[8192];

            while (!outDone) {
                if (!inDone) {
                    int inI = codec.dequeueInputBuffer(10000);
                    if (inI >= 0) {
                        ByteBuffer ib = codec.getInputBuffer(inI);
                        int size = ib == null ? -1 : ex.readSampleData(ib, 0);
                        if (size < 0) {
                            codec.queueInputBuffer(inI, 0, 0, 0,
                                    MediaCodec.BUFFER_FLAG_END_OF_STREAM);
                            inDone = true;
                        } else {
                            long pts = ex.getSampleTime();
                            codec.queueInputBuffer(inI, 0, size, pts, 0);
                            ex.advance();
                            fed++;
                        }
                    }
                }
                int outI = codec.dequeueOutputBuffer(info, 10000);
                if (outI == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                    MediaFormat of = codec.getOutputFormat();
                    if (of.containsKey(MediaFormat.KEY_CHANNEL_COUNT)) {
                        ch = Math.max(1, of.getInteger(MediaFormat.KEY_CHANNEL_COUNT));
                    }
                    if (of.containsKey(MediaFormat.KEY_SAMPLE_RATE)) {
                        int nsr = of.getInteger(MediaFormat.KEY_SAMPLE_RATE);
                        if (nsr > 0 && nsr != sr) {
                            sr = nsr;
                            w.close();
                            w = new WavIo.Writer(dst, sr);
                        }
                    }
                    if (of.containsKey(MediaFormat.KEY_PCM_ENCODING)) {
                        pcmEnc = of.getInteger(MediaFormat.KEY_PCM_ENCODING);
                    }
                } else if (outI >= 0) {
                    ByteBuffer ob = codec.getOutputBuffer(outI);
                    if (ob != null && info.size > 0) {
                        ob.position(info.offset);
                        ob.limit(info.offset + info.size);
                        int wrote = 0;
                        if (pcmEnc == AudioFormat.ENCODING_PCM_FLOAT) {
                            java.nio.FloatBuffer fb = ob.order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
                            int frames = fb.remaining() / ch;
                            int off = 0;
                            while (off < frames) {
                                int cnt = Math.min(frames - off, mono.length);
                                for (int i = 0; i < cnt; i++) {
                                    float v = 0f;
                                    for (int c = 0; c < ch; c++) v += fb.get((off + i) * ch + c);
                                    v /= ch;
                                    if (v > 1f) v = 1f; else if (v < -1f) v = -1f;
                                    mono[i] = (short) (v * 32767f);
                                }
                                w.write(mono, cnt);
                                wrote += cnt;
                                off += cnt;
                            }
                        } else if (pcmEnc == AudioFormat.ENCODING_PCM_8BIT) {
                            int frames = ob.remaining() / ch;
                            int off = 0;
                            while (off < frames) {
                                int cnt = Math.min(frames - off, mono.length);
                                for (int i = 0; i < cnt; i++) {
                                    int acc = 0;
                                    for (int c = 0; c < ch; c++) acc += (ob.get() & 0xFF) - 128;
                                    mono[i] = (short) (acc / ch * 256);
                                }
                                w.write(mono, cnt);
                                wrote += cnt;
                                off += cnt;
                            }
                        } else {
                            java.nio.ShortBuffer sb = ob.order(ByteOrder.LITTLE_ENDIAN).asShortBuffer();
                            int frames = sb.remaining() / ch;
                            int off = 0;
                            while (off < frames) {
                                int cnt = Math.min(frames - off, mono.length);
                                for (int i = 0; i < cnt; i++) {
                                    int acc = 0;
                                    for (int c = 0; c < ch; c++) acc += sb.get((off + i) * ch + c);
                                    mono[i] = (short) (acc / ch);
                                }
                                w.write(mono, cnt);
                                wrote += cnt;
                                off += cnt;
                            }
                        }
                        if (cb != null && durUs > 0 && wrote > 0) {
                            double sec = w.seconds();
                            cb.onProgress((int) Math.min(99, sec * 1e6 / durUs * 100));
                        }
                    }
                    codec.releaseOutputBuffer(outI, false);
                    if ((info.flags & MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) outDone = true;
                } else if (outI == MediaCodec.INFO_TRY_AGAIN_LATER && inDone) {
                    // 解码器不给数据也不结束：保守退出，避免死循环
                    if (fed == 0) throw new Exception("解码器没吐出任何数据");
                    outDone = true;
                }
            }
            w.close();
            w = null;
            double sec = 0;
            if (cb != null) cb.onProgress(100);
            int[] srOut = new int[1];
            // 用最终文件长度算时长（比累加可靠）
            try {
                float[] check = WavIo.readWav(dst.getAbsolutePath(), srOut);
                sec = check.length / (double) Math.max(1, srOut[0]);
            } catch (Exception ignored) {
            }
            return sec;
        } finally {
            if (w != null) w.close();
            if (codec != null) {
                try {
                    codec.stop();
                } catch (Exception ignored) {
                }
                try {
                    codec.release();
                } catch (Exception ignored) {
                }
            }
            try {
                ex.release();
            } catch (Exception ignored) {
            }
        }
    }
}
