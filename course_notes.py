#!/usr/bin/env python3
"""
课堂笔记自动生成脚本
用法：
  python course_notes.py --images ./slides/
  python course_notes.py --video lecture.mp4

环境：
  pip install "transformers[torch]>=5.7.0" torchvision av
"""

import argparse
import os
from pathlib import Path


# ── Prompt 模板 ──────────────────────────────────────────
SYSTEM_PROMPT = """你是一个专业的课堂笔记助手。你的任务是根据课件截图或板书图片，生成清晰、结构化的课堂笔记。

要求：
1. 先判断图片内容类型（课件PPT、手写板书、教材页面等）
2. 提取所有关键知识点，用 Markdown 格式组织
3. 数学公式用 LaTeX 格式（行内 $...$，独立 $$...$$）
4. 如果有图表，用文字描述其内容
5. 保持简洁，不添加图片中没有的额外信息"""

NOTE_PROMPT = "请根据这些图片生成课堂笔记。"

SUMMARY_PROMPT = """以上是本节课所有批次的笔记片段。请将它们整合为一份完整的课堂笔记，要求：
- 按主题/小节组织，有清晰的标题层级
- 合并重复内容，补全上下文关联
- 保持 LaTeX 公式格式
- 末尾添加「关键要点」总结"""


# ── 图片处理 ─────────────────────────────────────────────
def process_images(image_dir: str, output_path: str, model_id: str,
                   downsample: str = "16x", batch_size: int = 10):
    """本地加载模型，批量处理图片"""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"⏳ 加载模型 {model_id} ...")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    print("✅ 模型加载完成\n")

    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    images = sorted([
        os.path.join(image_dir, f) for f in os.listdir(image_dir)
        if Path(f).suffix.lower() in exts
    ])
    if not images:
        print(f"❌ {image_dir} 中没有图片")
        return

    total_batches = (len(images) + batch_size - 1) // batch_size
    print(f"📷 {len(images)} 张图片，每批 {batch_size} 张，共 {total_batches} 批\n")
    all_notes = []

    for batch_start in range(0, len(images), batch_size):
        batch = images[batch_start:batch_start + batch_size]
        batch_num = batch_start // batch_size + 1
        names = [os.path.basename(p) for p in batch]
        print(f"[批次 {batch_num}/{total_batches}] {', '.join(names)} ...", end=" ", flush=True)

        # 构建多图消息
        content = [{"type": "image", "url": p} for p in batch]
        content.append({"type": "text", "text": SYSTEM_PROMPT + "\n\n" + NOTE_PROMPT})

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

        for name in names:
            all_notes.append(f"## {name}\n")
        all_notes.append(note + "\n")
        print("✓")

    # 汇总
    if total_batches >= 2:
        print("📝 汇总整合中...", end=" ", flush=True)
        combined = "\n".join(all_notes)
        messages = [{"role": "user", "content": [{"type": "text", "text": combined + "\n\n" + SUMMARY_PROMPT}]}]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(model.device)
        generated = model.generate(**inputs, max_new_tokens=4096)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
        summary = processor.batch_decode(trimmed, skip_special_tokens=True,
                                          clean_up_tokenization_spaces=False)[0]
        all_notes.append("\n---\n# 汇总笔记\n\n" + summary)
        print("✓")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(all_notes))
    print(f"\n✅ 保存到 {output_path}")


# ── 视频处理 ─────────────────────────────────────────────
def process_video(video_path: str, output_path: str, model_id: str,
                  fps: float = 0.5, max_frames: int = 60, downsample: str = "16x"):
    """本地加载模型，抽帧 → 逐帧生成笔记 → 汇总"""
    import torch
    import av
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"⏳ 加载模型 {model_id} ...")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    print("✅ 模型加载完成\n")

    print(f"🎬 分析视频 {video_path} ...")
    container = av.open(video_path)
    video_stream = container.streams.video[0]
    duration = float(video_stream.duration * video_stream.time_base)
    orig_fps = float(video_stream.average_rate)
    total_frames = video_stream.frames or int(duration * orig_fps)

    interval = max(1, int(orig_fps / fps))
    if total_frames // interval > max_frames:
        interval = max(1, total_frames // max_frames)

    print(f"  时长: {duration:.0f}s, 原始 {orig_fps:.1f} fps, 每 {interval} 帧抽一帧\n")

    frame_count = 0
    saved_frames = []
    tmp_dir = os.path.join(os.path.dirname(output_path) or ".", "_video_frames")
    os.makedirs(tmp_dir, exist_ok=True)

    for frame in container.decode(video=0):
        if frame_count % interval == 0:
            img = frame.to_image()
            path = os.path.join(tmp_dir, f"frame_{frame_count:06d}.jpg")
            img.save(path, quality=85)
            saved_frames.append(path)
        frame_count += 1
        if len(saved_frames) >= max_frames:
            break
    container.close()

    print(f"✅ 抽了 {len(saved_frames)} 帧\n")
    all_notes = []

    for i, img_path in enumerate(saved_frames, 1):
        name = os.path.basename(img_path)
        timestamp = int(name.replace("frame_", "").replace(".jpg", "")) / orig_fps
        ts_str = f"{int(timestamp // 60)}:{int(timestamp % 60):02d}"
        print(f"  [{i}/{len(saved_frames)}] {name} ({ts_str}) ...", end=" ", flush=True)

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
        all_notes.append(f"## [{ts_str}] {name}\n\n{note}\n")
        print("✓")

    # 汇总
    if len(all_notes) >= 2:
        print("📝 汇总整合中...", end=" ", flush=True)
        combined = "\n\n".join(all_notes)
        messages = [{"role": "user", "content": [{"type": "text", "text": combined + "\n\n" + SUMMARY_PROMPT}]}]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(model.device)
        generated = model.generate(**inputs, max_new_tokens=4096)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
        summary = processor.batch_decode(trimmed, skip_special_tokens=True,
                                          clean_up_tokenization_spaces=False)[0]
        all_notes.append("\n---\n# 汇总笔记\n\n" + summary)
        print("✓")

    # 清理临时帧
    for p in saved_frames:
        try:
            os.remove(p)
        except OSError:
            pass
    try:
        os.rmdir(tmp_dir)
    except OSError:
        pass

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(all_notes))
    print(f"✅ 保存到 {output_path}")


# ── CLI ────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="课堂笔记自动生成")
    parser.add_argument("--images", help="图片文件夹路径")
    parser.add_argument("--video", help="视频文件路径")
    parser.add_argument("--model", default="openbmb/MiniCPM-V-4.6",
                        help="模型 ID 或本地路径")
    parser.add_argument("--output", default="notes.md", help="输出文件")
    parser.add_argument("--downsample", default="16x", choices=["4x", "16x"])
    parser.add_argument("--batch-size", type=int, default=10,
                        help="每批图片数 (默认 10)")
    parser.add_argument("--fps", type=float, default=0.5,
                        help="视频抽帧速率 (默认 0.5)")
    parser.add_argument("--max-frames", type=int, default=60,
                        help="视频最大帧数 (默认 60)")
    args = parser.parse_args()

    if not args.images and not args.video:
        parser.error("需要 --images 或 --video")

    if args.images:
        process_images(args.images, args.output, args.model,
                       args.downsample, args.batch_size)
    elif args.video:
        process_video(args.video, args.output, args.model,
                      args.fps, args.max_frames, args.downsample)


if __name__ == "__main__":
    main()
