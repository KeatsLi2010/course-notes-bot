#!/usr/bin/env python3
"""
课堂笔记自动生成 — 视频批量处理（视频帧 + 语音转录）
用法：
  python course_notes.py --video ./courses/
  python course_notes.py --video ./courses/ --fps 1 --max-frames 120 --whisper large-v3

环境：
  pip install "transformers[torch]>=5.7.0" torchvision av openai-whisper
"""

# 必须在所有 import 之前设置
import os
os.environ["GPTQMODEL_NOGIL"] = "0"
os.environ["TRITON_CACHE_DIR"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".triton_cache")

# 禁用 gptqmodel 的 triton nogil patcher（与新 triton 不兼容）
try:
    import gptqmodel.utils.nogil_patcher as _np
    if hasattr(_np, 'patch'):
        _np.patch = lambda: None
except ImportError:
    pass

import argparse
import re
from pathlib import Path


def strip_thinking(text: str) -> str:
    """移除模型的  `  ` 思考块"""
    return re.sub(r'`[^`]*`', '', text).strip()


SYSTEM_PROMPT = """你是一个专业的课堂笔记助手。根据课件截图/板书和同步的语音转录，生成结构化的课堂笔记。

⚠️ 重要：以语音转录内容为主，图片为辅。因为板书或课件截图可能有滞后性（老师口头讲了但还没写出来）。

要求：
1. 优先依据「语音转录」判断老师在讲什么知识点，图片仅用于参考公式、图表、板书细节
2. 提取关键知识点，用 Markdown 格式组织
3. 数学公式用 LaTeX 格式（行内 $...$，独立 $$...$$）
4. 保持简洁，不添加没有的信息"""

NOTE_PROMPT = "请根据这些图片和语音转录，生成这段时段的课堂笔记。"


def find_videos(path: str) -> list[str]:
    p = Path(path)
    if p.is_file():
        return [str(p)]
    exts = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".ts"}
    return sorted([str(f) for f in p.rglob("*") if f.suffix.lower() in exts])


def load_sensevoice():
    """加载 SenseVoice — 中文识别比 Whisper 更准更快"""
    from funasr import AutoModel
    from funasr.utils.postprocess_utils import rich_transcription_postprocess
    print("⏳ 加载 SenseVoice Small ...")
    model = AutoModel(
        model="iic/SenseVoiceSmall",
        trust_remote_code=True,
        device="cuda:0",
    )
    return model, rich_transcription_postprocess


def process_one(model, processor, video_path: str, output_dir: str,
                fps: float, max_frames: int, downsample: str,
                asr_model, transcribe_fn, audio_buffer: float = 5.0):
    import av

    name = Path(video_path).stem
    output_path = os.path.join(output_dir, f"{name}.txt")
    audio_dir = os.path.join(output_dir, "_audio")
    os.makedirs(audio_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"🎬 {name}")
    print(f"{'='*60}")

    container = av.open(video_path)
    video_stream = container.streams.video[0]
    audio_stream = container.streams.audio[0] if container.streams.audio else None

    duration = float(video_stream.duration * video_stream.time_base)
    orig_fps = float(video_stream.average_rate)
    total_frames = video_stream.frames or int(duration * orig_fps)

    interval = max(1, int(orig_fps / fps))
    if total_frames // interval > max_frames:
        interval = max(1, total_frames // max_frames)

    actual_frames = min(total_frames // interval, max_frames)
    has_audio = audio_stream is not None and asr_model is not None
    print(f"  时长: {duration:.0f}s | {orig_fps:.1f}fps | ~{actual_frames}帧 | 音频: {'✓' if has_audio else '✗'}")

    # 抽帧（若已存在则跳过）
    frame_count = 0
    saved_frames = []
    frame_timestamps = []
    frame_dir = os.path.join(output_dir, "_frames", name)

    existing = sorted(Path(frame_dir).glob(f"{name}_*.jpg")) if os.path.isdir(frame_dir) else []
    if existing:
        saved_frames = [str(p) for p in existing]
        for p in existing:
            ts = int(p.stem.split("_")[-1]) / orig_fps
            frame_timestamps.append(ts)
        print(f"  📷 复用 {len(saved_frames)} 帧 (已缓存)")
    else:
        os.makedirs(frame_dir, exist_ok=True)
        for frame in container.decode(video=0):
            if frame_count % interval == 0:
                img = frame.to_image()
                path = os.path.join(frame_dir, f"{name}_{frame_count:06d}.jpg")
                img.save(path, quality=85)
                saved_frames.append(path)
                frame_timestamps.append(frame_count / orig_fps)
            frame_count += 1
            if len(saved_frames) >= max_frames:
                break
        print(f"  📷 抽了 {len(saved_frames)} 帧")

    # 提取音频片段
    audio_segments = {}
    if audio_stream is None:
        print(f"  ⚠️ 视频无音频轨道，跳过语音转录")
    elif asr_model is None:
        print(f"  ⚠️ ASR 未加载，跳过语音转录")
    else:
        print(f"  🎤 提取音频片段 (ffmpeg) ...", end=" ", flush=True)
        import subprocess
        ffmpeg_ok = False
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=3)
            ffmpeg_ok = True
        except Exception:
            pass

        if ffmpeg_ok:
            for idx, ts in enumerate(frame_timestamps):
                start = max(0, ts - audio_buffer)
                dur = (interval / orig_fps) + 2 * audio_buffer
                seg_path = os.path.join(audio_dir, f"{name}_{idx:04d}_{ts:.1f}s.wav")
                try:
                    subprocess.run([
                        "ffmpeg", "-y", "-loglevel", "error",
                        "-ss", str(start), "-t", str(dur),
                        "-i", video_path,
                        "-ac", "1", "-ar", "16000",
                        seg_path
                    ], check=True, timeout=10)
                    audio_segments[idx] = seg_path
                except Exception:
                    pass
        else:
            print(f"(PyAV fallback) ...", end=" ", flush=True)
            import av as _av
            ac = _av.open(video_path)
            astr = ac.streams.audio[0]
            for idx, ts in enumerate(frame_timestamps):
                start_t = max(0, ts - audio_buffer)
                end_t = min(duration, ts + (interval / orig_fps) + audio_buffer)
                seg_path = os.path.join(audio_dir, f"{name}_{idx:04d}_{ts:.1f}s.wav")
                try:
                    out_c = _av.open(seg_path, "w")
                    out_s = out_c.add_stream("pcm_s16le")
                    out_s.rate = astr.rate
                    out_s.channels = astr.channels
                    ac.seek(int(start_t / astr.time_base))
                    for packet in ac.demux(astr):
                        if packet.pts is None:
                            continue
                        pkt_ts = float(packet.pts * packet.time_base)
                        if pkt_ts >= end_t:
                            break
                        if pkt_ts >= start_t:
                            packet.stream = out_s
                            out_c.mux(packet)
                    out_c.close()
                    audio_segments[idx] = seg_path
                except Exception:
                    pass
            ac.close()
        print(f"✓ ({len(audio_segments)} 段)")

    container.close()
    print(f"  📷 {'复用' if existing else '抽了'} {len(saved_frames)} 帧")

    # 逐批推理
    all_notes = []
    frame_batch_size = 4

    for batch_start in range(0, len(saved_frames), frame_batch_size):
        batch = saved_frames[batch_start:batch_start + frame_batch_size]
        batch_idx = list(range(batch_start, batch_start + len(batch)))
        batch_num = batch_start // frame_batch_size + 1
        total = (len(saved_frames) + frame_batch_size - 1) // frame_batch_size

        ts_list = [f"{int(frame_timestamps[i] // 60)}:{int(frame_timestamps[i] % 60):02d}" for i in batch_idx]

        content = []
        for img_path in batch:
            content.append({"type": "image", "url": img_path})

        # 附上对应音频的转录
        transcript_lines = []
        if has_audio:
            for i in batch_idx:
                if i in audio_segments:
                    try:
                        txt = transcribe_fn(audio_segments[i])
                        t = ts_list[i - batch_start] if (i - batch_start) < len(ts_list) else ""
                        transcript_lines.append(f"[{t}] {txt}")
                    except Exception:
                        pass

        text_content = SYSTEM_PROMPT
        if transcript_lines:
            text_content += "\n\n--- 语音转录（含上下文缓冲）---\n" + "\n".join(transcript_lines)
        text_content += f"\n\n[视频时间: {ts_list[0]} ~ {ts_list[-1]}]\n" + NOTE_PROMPT
        content.append({"type": "text", "text": text_content})

        print(f"  [批 {batch_num}/{total}] {ts_list[0]} ~ {ts_list[-1]} "
              f"{'🎤' if transcript_lines else ''} ...", end=" ", flush=True)

        messages = [{"role": "user", "content": content}]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
            downsample_mode=downsample,
            max_slice_nums=36,
        ).to(model.device)

        generated = model.generate(**inputs, downsample_mode=downsample, max_new_tokens=4096)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
        note = processor.batch_decode(trimmed, skip_special_tokens=True,
                                       clean_up_tokenization_spaces=False)[0]
        note = strip_thinking(note)
        all_notes.append(f"## [{ts_list[0]} ~ {ts_list[-1]}]\n\n{note}\n")
        print("✓")

    # 清理临时音频（帧图片保留以便复用）
    for p in audio_segments.values():
        try: os.remove(p)
        except OSError: pass
    try: os.rmdir(audio_dir)
    except OSError: pass

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(all_notes))
    print(f"  ✅ {output_path}")


def main():
    parser = argparse.ArgumentParser(description="课堂笔记 — 视频+语音")
    parser.add_argument("--video", required=True, help="视频文件或文件夹")
    parser.add_argument("--model", default="./model", help="模型本地路径")
    parser.add_argument("--output", default="./notes", help="输出目录")
    parser.add_argument("--downsample", default="16x", choices=["4x", "16x"])
    parser.add_argument("--fps", type=float, default=0.5)
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument("--audio-buffer", type=float, default=5.0,
                        help="每帧前后多取几秒音频做上下文 (默认 5s)")
    parser.add_argument("--test-audio", action="store_true",
                        help="单独测试音频提取+转录，不做笔记")
    args = parser.parse_args()

    videos = find_videos(args.video)
    if not videos:
        print(f"❌ 未找到视频: {args.video}")
        return

    # --test-audio: 跳过模型，只测音频
    if args.test_audio:
        test_video = videos[0]
        print(f"🧪 音频测试: {test_video}")
        import av as _av
        c = _av.open(test_video)
        astr = c.streams.audio[0] if c.streams.audio else None
        dur = float(c.streams.video[0].duration * c.streams.video[0].time_base)
        fps_v = float(c.streams.video[0].average_rate)
        c.close()
        if astr is None:
            print("❌ 视频无音频轨道")
            return
        interval = max(1, int(fps_v / args.fps))
        ts = interval / fps_v
        start = max(0, ts - args.audio_buffer)
        seg_dur = (interval / fps_v) + 2 * args.audio_buffer
        seg_path = os.path.join(args.output, "_test_audio.wav")
        os.makedirs(args.output, exist_ok=True)
        import subprocess
        try:
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error",
                "-ss", str(start), "-t", str(seg_dur),
                "-i", test_video, "-ac", "1", "-ar", "16000", seg_path],
                check=True, timeout=10)
            print(f"✅ 音频提取成功: {start:.1f}s ~ {start+seg_dur:.1f}s")
            # 测试 SenseVoice
            try:
                from funasr import AutoModel
                from funasr.utils.postprocess_utils import rich_transcription_postprocess
                model = AutoModel(model="iic/SenseVoiceSmall", trust_remote_code=True,
                                  device="cuda:0")
                res = model.generate(input=seg_path, cache={}, language="zh", use_itn=True)
                text = rich_transcription_postprocess(res[0]["text"])
                print(f"📝 SenseVoice: {text}")
            except Exception as e:
                print(f"⚠️ SenseVoice 失败: {e}")
            try: os.remove(seg_path)
            except: pass
        except FileNotFoundError:
            print("❌ ffmpeg 未安装/不在 PATH")
        except Exception as e:
            print(f"❌ ffmpeg 失败: {e}")
        return

    print(f"🎯 找到 {len(videos)} 个视频")
    os.makedirs(args.output, exist_ok=True)

    # 先加载 VLM（需要 CUDA）
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"⏳ 加载 VLM {args.model} ...")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    print("✅ VLM 就绪")

    # 加载 SenseVoice
    asr_model = None
    transcribe_fn = None
    try:
        asr_model, rich_postprocess = load_sensevoice()
        transcribe_fn = lambda path: rich_postprocess(
            asr_model.generate(input=path, cache={}, language="zh", use_itn=True)[0]["text"]
        )
        print("✅ SenseVoice 就绪")
    except Exception as e:
        print(f"⚠️ SenseVoice 加载失败: {e}")

    for video_path in videos:
        process_one(model, processor, video_path, args.output,
                    args.fps, args.max_frames, args.downsample,
                    asr_model, transcribe_fn, args.audio_buffer)

    for d in ["_audio"]:
        dp = os.path.join(args.output, d)
        try: os.rmdir(dp)
        except OSError: pass

    print(f"\n🏁 完成！{args.output}/")


if __name__ == "__main__":
    main()
