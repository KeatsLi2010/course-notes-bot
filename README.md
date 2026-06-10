# Course Notes Bot 📝

基于视觉语言模型的课堂笔记自动生成工具。支持图片和视频输入，逐帧/逐图生成结构化笔记，最后汇总整合。

## 依赖

```bash
pip install transformers[torch]>=5.7.0 torchvision av
```

## 使用方法

### 1. 启动模型服务

先启动 llama.cpp server 加载多模态模型：

```bash
llama-server -m model.gguf --mmproj mmproj.gguf --host 0.0.0.0 --port 8080 -ngl 99 -c 8192
```

### 2. 生成笔记

**图片模式：**
```bash
python course_notes.py --api http://localhost:8080 --images ./slides/ --output notes.md
```

**视频模式：**
```bash
# 默认每 2 秒一帧，最多 60 帧
python course_notes.py --api http://localhost:8080 --video lecture.mp4 --output notes.md

# 精细抽帧
python course_notes.py --api http://localhost:8080 --video lecture.mp4 --fps 1 --max-frames 120
```

**本地模式（直接加载模型）：**
```bash
python course_notes.py --local --images ./slides/ --output notes.md
```

### 3. 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--images` | - | 图片文件夹 |
| `--video` | - | 视频文件 |
| `--api` | - | llama.cpp API 地址 |
| `--local` | - | 本地加载模型 |
| `--output` | `notes.md` | 输出文件 |
| `--fps` | `0.5` | 视频抽帧速率 |
| `--max-frames` | `60` | 最大抽帧数 |
| `--downsample` | `16x` | 视觉 token 压缩率 |

## 工作流程

1. **输入**：图片文件夹或视频文件
2. **逐帧处理**：每张图/每帧发送给模型，生成笔记片段（含时间戳）
3. **汇总整合**：将所有片段合并，由模型整合为完整课堂笔记
4. **输出**：Markdown 格式，含 LaTeX 公式

## 推荐模型

| 模型 | 显存 | 质量 |
|------|------|------|
| MiniCPM-V 4.6 | ~1.5 GB | 轻量测试 |
| Qwen2.5-VL 7B | ~5.5 GB | 推荐 |
| Gemma 4 E4B | ~4 GB (估) | 平衡之选 |
