#!/usr/bin/env python3
"""
课堂笔记自动生成 — 视频模式
用法：
  python course_notes.py --video lecture.mp4
  python course_notes.py --video lecture.mp4 --model Qwen/Qwen2.5-VL-7B-Instruct

环境：
  pip install "transformers[torch]>=5.7.0" torchvision av
"""

import argparse
import os


SYSTEM_PROMPT = """你是一个专业的课堂笔记助手。你的任务是根据课件截图或板书图片，生成清晰、结构化的课堂笔记。

要求：
1. 先判断图片内容类型（课件PPT、手写板书、教材页面等）
2. 提取所有关键知识点，用 Markdown 格式组织
3. 数学公式用 LaTeX 格式（行内 $...$，独立 $$...$$）
4. 如果有图表，用文字描述其内容
5. 保持简洁，不添加图片中没有的额外信息"""

NOTE_PROMPT = "请根据这张图片生成课堂笔记。"

SUMMARY_PROMPT = """以上是本视频所有帧的笔记片段。请将它们整合为一份完整的课堂笔记，要求：
- 按主题/小节组织，有清晰的标题层级
- 合并重复内容，补全上下文关联
- 保持 LaTeX 公式格式
- 末尾添加「关键要点」总结"""


def process_video(video_path: str, output_path: str, model_id: str,
                  fps: float = 0.5, max_frames: int = 60, downsample: str = "16x"):
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


def main():
    parser = argparse.ArgumentParser(description="课堂笔记自动生成 — 视频模式")
    parser.add_argument("--video", required=True, help="视频文件路径")
    parser.add_argument("--model", default="openbmb/MiniCPM-V-4.6",
                        help="模型 ID 或本地路径")
    parser.add_argument("--output", default="notes.md", help="输出文件")
    parser.add_argument("--downsample", default="16x", choices=["4x", "16x"])
    parser.add_argument("--fps", type=float, default=0.5,
                        help="抽帧速率 (默认 0.5)")
    parser.add_argument("--max-frames", type=int, default=60,
                        help="最大帧数 (默认 60)")
    args = parser.parse_args()

    process_video(args.video, args.output, args.model,
                  args.fps, args.max_frames, args.downsample)


if __name__ == "__main__":
    main()
