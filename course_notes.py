#!/usr/bin/env python3
"""
MiniCPM-V 4.6 课堂笔记生成脚本
用法：
  # 图片文件夹 → 逐张生成笔记
  python notes.py --images ./slides/

  # 单个视频 → 逐帧生成笔记
  python notes.py --video lecture.mp4

  # 连接 llama.cpp server（先启动 llama-server）
  python notes.py --api http://localhost:8080 --images ./slides/

环境：
  pip install "transformers[torch]>=5.7.0" torchvision av
"""

import argparse
import base64
import json
import os
import sys
from pathlib import Path


# ── Prompt 模板 ──────────────────────────────────────────
SYSTEM_PROMPT = """你是一个专业的课堂笔记助手。你的任务是根据课件截图或板书图片，生成清晰、结构化的课堂笔记。

要求：
1. 先判断图片内容类型（课件PPT、手写板书、教材页面等）
2. 提取所有关键知识点，用 Markdown 格式组织
3. 数学公式用 LaTeX 格式（行内 $...$，独立 $$...$$）
4. 如果有图表，用文字描述其内容
5. 保持简洁，不添加图片中没有的额外信息
6. 每张图片独立生成一段笔记，最后汇总"""

NOTE_PROMPT = "请根据这张图片生成课堂笔记。"

SUMMARY_PROMPT = """以上是本节课所有图片的笔记片段。请将它们整合为一份完整的课堂笔记，要求：
- 按主题/小节组织，有清晰的标题层级
- 合并重复内容，补全上下文关联
- 保持 LaTeX 公式格式
- 末尾添加「关键要点」总结"""


# ── llama.cpp API 模式 ────────────────────────────────────
def chat_api(api_url: str, messages: list[dict], max_tokens: int = 2048) -> str:
    """调用 OpenAI 兼容 API"""
    import urllib.request

    body = json.dumps({
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }).encode()

    req = urllib.request.Request(
        f"{api_url}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())["choices"][0]["message"]["content"]


def image_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def process_video_api(api_url: str, video_path: str, output_path: str, fps: float = 0.5, max_frames: int = 60):
    """通过 llama.cpp API 处理视频：抽帧 → 逐帧生成笔记 → 汇总"""
    import av

    print(f"🎬 分析视频 {video_path} ...")
    container = av.open(video_path)
    video_stream = container.streams.video[0]
    duration = float(video_stream.duration * video_stream.time_base)
    orig_fps = float(video_stream.average_rate)
    total_frames = video_stream.frames or int(duration * orig_fps)

    # 计算抽帧间隔
    interval = max(1, int(orig_fps / fps))
    estimated = total_frames // interval
    if estimated > max_frames:
        interval = max(1, total_frames // max_frames)

    print(f"  时长: {duration:.0f}s, 原始 {orig_fps:.1f} fps, 每 {interval} 帧抽一帧, 预计 ~{total_frames // interval} 帧")

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
        timestamp = int(img_path.rsplit("_", 1)[1].split(".")[0]) / orig_fps
        ts_str = f"{int(timestamp // 60)}:{int(timestamp % 60):02d}"
        print(f"  [{i}/{len(saved_frames)}] {name} ({ts_str}) ...", end=" ", flush=True)

        b64 = image_to_base64(img_path)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": f"[视频时间: {ts_str}] " + NOTE_PROMPT},
            ]},
        ]
        try:
            note = chat_api(api_url, messages)
            all_notes.append(f"## [{ts_str}] {name}\n\n{note}\n")
            print("✓")
        except Exception as e:
            print(f"✗ ({e})")

    # 汇总
    if len(all_notes) >= 2:
        print("📝 汇总整合中...", end=" ", flush=True)
        combined = "\n\n".join(all_notes)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": combined + "\n\n" + SUMMARY_PROMPT},
        ]
        try:
            summary = chat_api(api_url, messages, max_tokens=4096)
            all_notes.append("\n---\n# 汇总笔记\n\n" + summary)
            print("✓")
        except Exception as e:
            print(f"✗ ({e})")

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
    print(f"✅ 笔记已保存到 {output_path}")


def process_images_api(api_url: str, image_dir: str, output_path: str):
    """通过 llama.cpp API 处理图片文件夹"""
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    images = sorted(
        [os.path.join(image_dir, f) for f in os.listdir(image_dir)
         if Path(f).suffix.lower() in exts]
    )
    if not images:
        print(f"❌ {image_dir} 中没有图片")
        return

    print(f"📷 找到 {len(images)} 张图片，开始生成笔记...")
    all_notes = []

    for i, img_path in enumerate(images, 1):
        name = os.path.basename(img_path)
        print(f"  [{i}/{len(images)}] {name} ...", end=" ", flush=True)

        b64 = image_to_base64(img_path)
        ext = Path(img_path).suffix.lower().replace(".", "")
        mime = f"image/{ext}" if ext != "jpg" else "image/jpeg"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                {"type": "text", "text": NOTE_PROMPT},
            ]},
        ]
        try:
            note = chat_api(api_url, messages)
            all_notes.append(f"## {name}\n\n{note}\n")
            print("✓")
        except Exception as e:
            print(f"✗ ({e})")

    # 汇总
    if len(all_notes) >= 2:
        print("📝 汇总整合中...", end=" ", flush=True)
        combined = "\n\n".join(all_notes)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": combined + "\n\n" + SUMMARY_PROMPT},
        ]
        try:
            summary = chat_api(api_url, messages, max_tokens=4096)
            all_notes.append("\n---\n# 汇总笔记\n\n" + summary)
            print("✓")
        except Exception as e:
            print(f"✗ ({e})")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(all_notes))
    print(f"✅ 笔记已保存到 {output_path}")


# ── 本地 transformers 模式 ──────────────────────────────────
def process_images_local(image_dir: str, output_path: str, downsample: str = "16x"):
    """本地加载模型处理图片"""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model_id = "openbmb/MiniCPM-V-4.6"
    print(f"⏳ 加载模型 {model_id} ...")

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    print("✅ 模型加载完成")

    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    images = sorted([
        os.path.join(image_dir, f) for f in os.listdir(image_dir)
        if Path(f).suffix.lower() in exts
    ])
    if not images:
        print(f"❌ {image_dir} 中没有图片")
        return

    print(f"📷 找到 {len(images)} 张图片\n")
    all_notes = []

    for i, img_path in enumerate(images, 1):
        name = os.path.basename(img_path)
        print(f"  [{i}/{len(images)}] {name} ...", end=" ", flush=True)

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "url": img_path},
                {"type": "text", "text": SYSTEM_PROMPT + "\n\n" + NOTE_PROMPT},
            ],
        }]

        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
            downsample_mode=downsample,
            max_slice_nums=36,
        ).to(model.device)

        generated = model.generate(
            **inputs,
            downsample_mode=downsample,
            max_new_tokens=2048,
        )
        trimmed = [
            out[len(inp):]
            for inp, out in zip(inputs.input_ids, generated)
        ]
        note = processor.batch_decode(
            trimmed, skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        all_notes.append(f"## {name}\n\n{note}\n")
        print("✓")

    # 汇总
    if len(all_notes) >= 2:
        print("📝 汇总整合中...", end=" ", flush=True)
        combined = "\n\n".join(all_notes)
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": combined + "\n\n" + SUMMARY_PROMPT}],
        }]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(model.device)
        generated = model.generate(**inputs, max_new_tokens=4096)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
        summary = processor.batch_decode(
            trimmed, skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        all_notes.append("\n---\n# 汇总笔记\n\n" + summary)
        print("✓")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(all_notes))
    print(f"✅ 笔记已保存到 {output_path}")


# ── CLI ────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="MiniCPM-V 4.6 课堂笔记生成")
    parser.add_argument("--images", help="图片文件夹路径")
    parser.add_argument("--video", help="视频文件路径")
    parser.add_argument("--api", help="llama.cpp API 地址 (如 http://localhost:8080)")
    parser.add_argument("--local", action="store_true", help="本地加载模型（需 GPU）")
    parser.add_argument("--output", default="notes.md", help="输出文件路径")
    parser.add_argument("--downsample", default="16x", choices=["4x", "16x"],
                        help="视觉 token 压缩率 (4x=精细, 16x=快速)")
    parser.add_argument("--fps", type=float, default=0.5,
                        help="视频抽帧速率 (默认 0.5, 即每 2 秒一帧)")
    parser.add_argument("--max-frames", type=int, default=60,
                        help="视频最大抽帧数 (默认 60)")
    args = parser.parse_args()

    if not args.images and not args.video:
        parser.error("至少需要 --images 或 --video")

    if args.images:
        if args.api:
            process_images_api(args.api, args.images, args.output)
        elif args.local:
            process_images_local(args.images, args.output, args.downsample)
        else:
            parser.error("需要 --api 或 --local")
    elif args.video:
        if args.api:
            process_video_api(args.api, args.video, args.output, args.fps, args.max_frames)
        elif args.local:
            print("⚠️ 本地视频模式需先装 transformers，请用 --api 模式连接 llama-server")
        else:
            parser.error("视频模式需要 --api")


if __name__ == "__main__":
    main()
