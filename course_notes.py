#!/usr/bin/env python3
"""
课堂笔记自动生成 — 视频批量处理
用法：
  python course_notes.py --video lecture.mp4
  python course_notes.py --video ./videos/
  python course_notes.py --video ./videos/ --fps 1 --max-frames 120

环境：
  pip install "transformers[torch]>=5.7.0" torchvision av
"""

import argparse
import os
from pathlib import Path


SYSTEM_PROMPT = """你是一个专业的课堂笔记助手。你的任务是根据课件截图或板书图片，生成清晰、结构化的课堂笔记。

要求：
1. 先判断图片内容类型（课件PPT、手写板书、教材页面等）
2. 提取所有关键知识点，用 Markdown 格式组织
3. 数学公式用 LaTeX 格式（行内 $...$，独立 $$...$$）
4. 如果有图表，用文字描述其内容
5. 保持简洁，不添加图片中没有的额外信息"""

NOTE_PROMPT = "请根据这张图片生成课堂笔记。"


def find_videos(path: str) -> list[str]:
    """查找视频文件：单文件直接返回，文件夹递归搜索"""
    p = Path(path)
    if p.is_file():
        return [str(p)]
    exts = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".ts"}
    videos = sorted([
        str(f) for f in p.rglob("*") if f.suffix.lower() in exts
    ])
    return videos


def process_one(model, processor, video_path: str, output_dir: str,
                fps: float, max_frames: int, downsample: str):
    import av

    name = Path(video_path).stem
    output_path = os.path.join(output_dir, f"{name}.txt")

    print(f"\n{'='*60}")
    print(f"🎬 {name}")
    print(f"{'='*60}")

    container = av.open(video_path)
    video_stream = container.streams.video[0]
    duration = float(video_stream.duration * video_stream.time_base)
    orig_fps = float(video_stream.average_rate)
    total_frames = video_stream.frames or int(duration * orig_fps)

    interval = max(1, int(orig_fps / fps))
    if total_frames // interval > max_frames:
        interval = max(1, total_frames // max_frames)

    actual_frames = min(total_frames // interval, max_frames)
    print(f"  时长: {duration:.0f}s | 原始 {orig_fps:.1f}fps | 每{interval}帧抽一帧 → ~{actual_frames}帧")

    frame_count = 0
    saved_frames = []
    tmp_dir = os.path.join(output_dir, "_frames")
    os.makedirs(tmp_dir, exist_ok=True)

    for frame in container.decode(video=0):
        if frame_count % interval == 0:
            img = frame.to_image()
            path = os.path.join(tmp_dir, f"{name}_{frame_count:06d}.jpg")
            img.save(path, quality=85)
            saved_frames.append(path)
        frame_count += 1
        if len(saved_frames) >= max_frames:
            break
    container.close()

    all_notes = []
    for i, img_path in enumerate(saved_frames, 1):
        ts = int(Path(img_path).stem.split("_")[-1]) / orig_fps
        ts_str = f"{int(ts // 60)}:{int(ts % 60):02d}"
        print(f"  [{i}/{len(saved_frames)}] {ts_str} ...", end=" ", flush=True)

        messages = [{"role": "user", "content": [
            {"type": "image", "url": img_path},
            {"type": "text", "text": SYSTEM_PROMPT + f"\n\n[视频时间: {ts_str}]\n" + NOTE_PROMPT},
        ]}]

        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
            downsample_mode=downsample,
            max_slice_nums=36,
        ).to(model.device)

        generated = model.generate(**inputs, downsample_mode=downsample, max_new_tokens=2048)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
        note = processor.batch_decode(trimmed, skip_special_tokens=True,
                                       clean_up_tokenization_spaces=False)[0]
        all_notes.append(f"## [{ts_str}]\n\n{note}\n")
        print("✓")

    for p in saved_frames:
        try:
            os.remove(p)
        except OSError:
            pass

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(all_notes))
    print(f"  ✅ {output_path}")


def main():
    parser = argparse.ArgumentParser(description="课堂笔记 — 视频批量处理")
    parser.add_argument("--video", required=True, help="视频文件或文件夹")
    parser.add_argument("--model", default="openbmb/MiniCPM-V-4.6-Thinking-AWQ",
                        help="模型 ID 或本地路径")
    parser.add_argument("--output", default="./notes", help="输出目录")
    parser.add_argument("--downsample", default="16x", choices=["4x", "16x"])
    parser.add_argument("--fps", type=float, default=0.5)
    parser.add_argument("--max-frames", type=int, default=60)
    args = parser.parse_args()

    videos = find_videos(args.video)
    if not videos:
        print(f"❌ 未找到视频: {args.video}")
        return

    print(f"🎯 找到 {len(videos)} 个视频")
    os.makedirs(args.output, exist_ok=True)

    # 加载模型（只加载一次）
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"⏳ 加载模型 {args.model} ...")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    print("✅ 模型加载完成")

    # 批量处理
    for video_path in videos:
        process_one(model, processor, video_path, args.output,
                    args.fps, args.max_frames, args.downsample)

    # 清理临时帧目录
    tmp_dir = os.path.join(args.output, "_frames")
    try:
        os.rmdir(tmp_dir)
    except OSError:
        pass

    print(f"\n🏁 全部完成！笔记保存在 {args.output}/")


if __name__ == "__main__":
    main()
