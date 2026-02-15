# AURORA: Annotation-free Unification of Reinforcement and Opponent Reasoning for Anti-hallucination

## 🚀 Overview
AURORA is a high-performance reinforcement learning framework designed to improve Vision-Language Models (VLMs) like **Qwen3-VL** using unannotated data. By employing an adversarial Verifier and ground-truth tools (Grounding DINO, EasyOCR), it autonomously reduces hallucinations.

**Designed for High-Performance Compute Clusters (e.g., 8x H200).**

## 📂 Project Structure & Key Locations

| Component | Path | Description |
| :--- | :--- | :--- |
| **Manage CLI** | `manage.py` | **Start Here!** The visual entry point for all tasks. |
| **Training Script** | `aurora_train.py` | Distributed training logic (GRPO) using **YFCC100M**. |
| **Evaluation** | `eval.py` | Inference and simple benchmarking. |
| **Dataset** | `./data/yfcc100m/` | **Auto-Downloaded** YFCC100M Images. |
| **Checkpoints** | `./output/checkpoints/` | **MODEL SAVED HERE**. Automated saving every 50 steps. |

## ⚡ Quick Start (Visual CLI)

The easiest way to run the project is via the interactive management script. **No manual dataset download is required.**

1.  **Install Requirements**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Launch the Manager**:
    ```bash
    python manage.py
    ```
    *Select "🚀 Train AURORA Model". The script will automatically download a subset of YFCC100M (default: 5000 images) if the data folder is empty.*

## 🔧 Configuration

### 1. Data Setup (Auto-Magic)
The project handles YFCC100M automatically.
*   **Location**: `./data/yfcc100m`
*   **Mechanism**: If the folder is empty, the script streams metadata from Hugging Face (`limingcv/YFCC100M_OpenAI_subset`) and downloads images in parallel using `aiohttp`.
*   **Customize Count**: Edit `aurora_train.py` -> `DOWNLOAD_COUNT = 5000` to download more/less.

### 2. Output & Checkpoints
Models are automatically saved to `./output/checkpoints`.
*   **Format**: Hugging Face compatible (config.json, pytorch_model.bin).
*   **Frequency**: Every 50 steps + Final epoch.

### 3. Distributed Training (8x H200)
The script uses `accelerate` for multi-GPU orchestration.
*   **Default**: `python manage.py` invokes `accelerate launch aurora_train.py`.
*   **Custom Launch**:
    ```bash
    accelerate launch --multi_gpu --num_processes=8 aurora_train.py
    ```

## 📊 Evaluation
To test your trained model:
1.  Place test images in `./data/test_images`.
2.  Run:
    ```bash
    python eval.py --model_path ./output/checkpoints/vlm_final
    ```

## 🛠️ Architecture
-   **VLM**: `Qwen/Qwen3-VL-8B-Instruct` (BF16, Flash Attn 2)
-   **Verifier**: `DeepSeek-R1-Distill-Qwen-7B`
-   **Tools**: Grounding DINO (Object Detection) + EasyOCR (Text Reading)
-   **Dataset**: YFCC100M (Auto-Download Stream)
-   **Method**: Group Relative Policy Optimization (GRPO)

## 📝 License
[MIT License](LICENSE)
