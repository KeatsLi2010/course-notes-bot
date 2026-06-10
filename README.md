# Course Notes Bot 📝

基于视觉语言模型 + Whisper 语音转录的课堂笔记自动生成工具。

视频 → 抽帧 + 语音转录 → VLM 生成笔记 → 输出 txt

## 环境

```bash
pip install -r requirements.txt
```

模型文件放到 `./model/` 目录下（默认使用 MiniCPM-V-4.6-Thinking-AWQ）。

## 使用方法

```bash
# 单个视频
python course_notes.py --video lecture.mp4

# 整个文件夹（递归搜索视频）
python course_notes.py --video ./courses/

# 完整参数
python course_notes.py --video ./courses/ \
    --model ./model \
    --whisper large-v3 \
    --output ./notes \
    --fps 1 \
    --max-frames 120 \
    --audio-buffer 10
```

## 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--video` | (必填) | 视频文件或文件夹 |
| `--model` | `./model` | VLM 模型本地路径 |
| `--whisper` | `base` | Whisper 模型：tiny / base / small / medium / large-v3 |
| `--output` | `./notes` | 输出目录 |
| `--fps` | `0.5` | 抽帧速率（帧/秒） |
| `--max-frames` | `60` | 最大抽帧数 |
| `--downsample` | `16x` | 视觉 token 压缩率（4x 更精细，16x 更快） |
| `--audio-buffer` | `5.0` | 每帧前后多取几秒音频做上下文 |

## 工作流程

1. **抽帧**：按 fps 从视频均匀抽取帧
2. **语音转录**：每帧对应的音频段用 Whisper 转文字（含前后 buffer 秒数做上下文）
3. **批量推理**：每 4 帧一批，图片 + 语音文字一起发给 VLM
4. **输出**：每个视频生成一个 `.txt` 文件，含所有批次的笔记片段
