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

import argparse
from pathlib import Path


SYSTEM_PROMPT = """你是一个专业的课堂笔记助手。根据课件截图/板书和同步的语音转录，生成结构化的课堂笔记。

要求：
1. 结合图片内容和下方「语音转录」理解老师在讲什么
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


def load_whisper(model_name: str):
    """加载 Whisper"""
    import whisper
    print(f"⏳ 加载 Whisper {model_name} ...")
    return whisper.load_model(model_name)


def transcribe_segment(whisper_model, audio_path: str) -> str:
    """用 Whisper 转录一段音频"""
    result = whisper_model.transcribe(audio_path, language="zh", fp16=False)
    return result["text"].strip()


def process_one(model, processor, video_path: str, output_dir: str,
                fps: float, max_frames: int, downsample: str,
                whisper_model, audio_buffer: float = 5.0):
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
    has_audio = audio_stream is not None and whisper_model is not None
    print(f"  时长: {duration:.0f}s | {orig_fps:.1f}fps | ~{actual_frames}帧 | 音频: {'✓' if has_audio else '✗'}")

    # 抽帧 + 抽对应音频片段
    frame_count = 0
    saved_frames = []
    frame_timestamps = []  # (frame_index, timestamp_seconds)
    tmp_dir = os.path.join(output_dir, "_frames")
    os.makedirs(tmp_dir, exist_ok=True)

    for frame in container.decode(video=0):
        if frame_count % interval == 0:
            img = frame.to_image()
            path = os.path.join(tmp_dir, f"{name}_{frame_count:06d}.jpg")
            img.save(path, quality=85)
            saved_frames.append(path)
            frame_timestamps.append(frame_count / orig_fps)
        frame_count += 1
        if len(saved_frames) >= max_frames:
            break

    # 提取音频片段（每帧对应一个音频块）
    audio_segments = {}
    if has_audio:
        print(f"  🎤 提取音频片段 ...", end=" ", flush=True)
        for idx, ts in enumerate(frame_timestamps):
            start = max(0, ts - audio_buffer)
            end = min(duration, ts + (interval / orig_fps) + audio_buffer)
            seg_path = os.path.join(audio_dir, f"{name}_{idx:04d}_{ts:.1f}s.wav")

            try:
                out_container = av.open(seg_path, "w")
                out_stream = out_container.add_stream("pcm_s16le")
                out_stream.rate = audio_stream.rate
                out_stream.channels = audio_stream.channels

                container.seek(int(start * av.time_base))
                for packet in container.demux(audio_stream):
                    if packet.pts is None:
                        continue
                    pkt_ts = float(packet.pts * packet.time_base)
                    if pkt_ts >= end:
                        break
                    if pkt_ts >= start:
                        packet.stream = out_stream
                        out_container.mux(packet)
                out_container.close()
                audio_segments[idx] = seg_path
            except Exception:
                pass
        print(f"✓ ({len(audio_segments)} 段)")

    container.close()

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
                        txt = transcribe_segment(whisper_model, audio_segments[i])
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
        all_notes.append(f"## [{ts_list[0]} ~ {ts_list[-1]}]\n\n{note}\n")
        print("✓")

    # 清理临时文件
    for p in saved_frames:
        try: os.remove(p)
        except OSError: pass
    for p in audio_segments.values():
        try: os.remove(p)
        except OSError: pass
    for d in [tmp_dir, audio_dir]:
        try: os.rmdir(d)
        except OSError: pass

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(all_notes))
    print(f"  ✅ {output_path}")


def main():
    parser = argparse.ArgumentParser(description="课堂笔记 — 视频+语音")
    parser.add_argument("--video", required=True, help="视频文件或文件夹")
    parser.add_argument("--model", default="./model", help="模型本地路径")
    parser.add_argument("--whisper", default="base", help="Whisper 模型 (tiny/base/small/medium/large-v3)")
    parser.add_argument("--output", default="./notes", help="输出目录")
    parser.add_argument("--downsample", default="16x", choices=["4x", "16x"])
    parser.add_argument("--fps", type=float, default=0.5)
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument("--audio-buffer", type=float, default=5.0,
                        help="每帧前后多取几秒音频做上下文 (默认 5s)")
    args = parser.parse_args()

    videos = find_videos(args.video)
    if not videos:
        print(f"❌ 未找到视频: {args.video}")
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

    # 再加载 Whisper
    whisper_model = None
    try:
        whisper_model = load_whisper(args.whisper)
    except Exception as e:
        print(f"⚠️ Whisper 加载失败，跳过语音转录: {e}")

    for video_path in videos:
        process_one(model, processor, video_path, args.output,
                    args.fps, args.max_frames, args.downsample,
                    whisper_model, args.audio_buffer)

    for d in ["_frames", "_audio"]:
        dp = os.path.join(args.output, d)
        try: os.rmdir(dp)
        except OSError: pass

    print(f"\n🏁 完成！{args.output}/")


if __name__ == "__main__":
    main()
