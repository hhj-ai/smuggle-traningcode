"""
Post-training evaluation: AURORA internal eval, POPE/MMHal benchmarks,
training curve plots, and base-vs-trained comparison charts.
"""

import json
import os
import gc
import random
import glob
import re
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# AuroraEvaluator: run full AURORA pipeline (generate → extract → verify)
# ---------------------------------------------------------------------------
class AuroraEvaluator:
    """Evaluate a VLM+Verifier+Tools stack on a set of images."""

    def __init__(self, vlm, verifier, tools):
        self.vlm = vlm
        self.verifier = verifier
        self.tools = tools

    def evaluate(self, image_dir, num_images=50, num_generations=4,
                 label="trained", seed=42):
        """
        Run AURORA pipeline and return structured metrics dict.
        Uses fixed seed so base and trained models evaluate the same images.
        """
        # Collect image paths
        exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
        all_paths = []
        for ext in exts:
            all_paths.extend(glob.glob(os.path.join(image_dir, "**", ext),
                                       recursive=True))
        all_paths.sort()
        if not all_paths:
            print(f"[AuroraEval] No images found in {image_dir}")
            return None

        rng = random.Random(seed)
        sample = rng.sample(all_paths, min(num_images, len(all_paths)))

        total_correct, total_incorrect, total_uncertain = 0, 0, 0
        total_claims = 0
        vlm_rewards = []
        desc_lengths = []
        images_evaluated = 0

        from models import _unwrap_model
        vlm_model = _unwrap_model(self.vlm.model)

        for img_path in sample:
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception:
                continue

            # Generate descriptions
            with torch.no_grad():
                results = self.vlm.generate_description_batch(
                    [image], num_generations=num_generations)
            descs = results[0] if results else []
            if not descs:
                continue

            images_evaluated += 1
            for desc in descs:
                desc_lengths.append(len(desc.split()))
                # Extract claims
                with torch.no_grad():
                    claims_list, _ = self.verifier.verify_claims_batch(
                        [desc], max_batch=1)
                claims = claims_list[0] if claims_list else []
                total_claims += len(claims)

                # Tool verification
                c, inc, unc = 0, 0, 0
                for claim in claims:
                    try:
                        verdict, _, _ = self.tools.verify_claim(
                            claim, img_path)
                    except Exception:
                        verdict = "uncertain"
                    if verdict == "correct":
                        c += 1
                    elif verdict == "incorrect":
                        inc += 1
                    else:
                        unc += 1
                total_correct += c
                total_incorrect += inc
                total_uncertain += unc

                # Simple reward proxy
                tc = c + inc + unc
                if tc > 0:
                    vlm_rewards.append((c - inc) / tc)

        if total_claims == 0:
            print(f"[AuroraEval-{label}] No claims extracted")
            return None

        import numpy as np
        result = {
            "label": label,
            "claim_accuracy": total_correct / total_claims,
            "incorrect_rate": total_incorrect / total_claims,
            "uncertain_rate": total_uncertain / total_claims,
            "vlm_reward_mean": float(np.mean(vlm_rewards)) if vlm_rewards else 0.0,
            "vlm_reward_std": float(np.std(vlm_rewards)) if vlm_rewards else 0.0,
            "avg_desc_length": float(np.mean(desc_lengths)) if desc_lengths else 0.0,
            "total_claims": total_claims,
            "total_images_evaluated": images_evaluated,
        }
        print(f"[AuroraEval-{label}] accuracy={result['claim_accuracy']:.3f} "
              f"incorrect={result['incorrect_rate']:.3f} "
              f"images={images_evaluated} claims={total_claims}")
        return result


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def _try_import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("[PostEval] matplotlib not installed, skipping plots")
        return None


def plot_training_curves(metrics, output_dir):
    """3 subplots: reward curves, claim stats, phase timings → training_curves.png"""
    plt = _try_import_matplotlib()
    if plt is None or not metrics:
        return

    steps = [m.get("global_step", i) for i, m in enumerate(metrics)]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1) Reward curves
    ax = axes[0]
    ax.plot(steps, [m.get("vlm_reward_mean", 0) for m in metrics],
            label="VLM reward", alpha=0.8)
    ax.plot(steps, [m.get("verifier_reward_mean", 0) for m in metrics],
            label="Verifier reward", alpha=0.8)
    ax.set_xlabel("Global step")
    ax.set_ylabel("Reward")
    ax.set_title("Reward Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2) Claim statistics
    ax = axes[1]
    ax.plot(steps, [m.get("claim_correct", 0) for m in metrics],
            label="correct", alpha=0.8)
    ax.plot(steps, [m.get("claim_incorrect", 0) for m in metrics],
            label="incorrect", alpha=0.8)
    ax.plot(steps, [m.get("claim_uncertain", 0) for m in metrics],
            label="uncertain", alpha=0.8)
    ax.set_xlabel("Global step")
    ax.set_ylabel("Count")
    ax.set_title("Claim Statistics")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3) Phase timings
    ax = axes[2]
    for key, label in [("phase1_time", "P1-Gen"), ("phase2_time", "P2-Verify"),
                        ("phase3_time", "P3-VLM train"), ("phase4_time", "P4-Ver train")]:
        vals = [m.get(key, 0) for m in metrics]
        if any(v > 0 for v in vals):
            ax.plot(steps, vals, label=label, alpha=0.8)
    ax.set_xlabel("Global step")
    ax.set_ylabel("Seconds")
    ax.set_title("Phase Timings")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    eval_dir = os.path.join(output_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    save_path = os.path.join(eval_dir, "training_curves.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[PostEval] Training curves saved to {save_path}")


def plot_comparison(trained, base, trained_bench, base_bench, output_dir):
    """2 subplots: AURORA metrics comparison + benchmark comparison → comparison.png"""
    plt = _try_import_matplotlib()
    if plt is None:
        return

    import numpy as np
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1) AURORA internal metrics
    ax = axes[0]
    if trained and base:
        keys = ["claim_accuracy", "incorrect_rate", "uncertain_rate",
                "vlm_reward_mean"]
        labels = ["Accuracy", "Incorrect", "Uncertain", "VLM Reward"]
        x = np.arange(len(keys))
        w = 0.35
        base_vals = [base.get(k, 0) for k in keys]
        trained_vals = [trained.get(k, 0) for k in keys]
        ax.bar(x - w / 2, base_vals, w, label="Base", alpha=0.8)
        ax.bar(x + w / 2, trained_vals, w, label="Trained", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15)
        ax.set_title("AURORA Internal Metrics")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
    else:
        ax.text(0.5, 0.5, "AURORA eval data unavailable",
                ha="center", va="center", transform=ax.transAxes)

    # 2) POPE / MMHal benchmarks
    ax = axes[1]
    bench_keys, bench_labels = [], []
    base_bvals, trained_bvals = [], []
    if trained_bench.get("pope") and base_bench.get("pope"):
        bench_keys.append("pope_acc")
        bench_labels.append("POPE Acc")
        base_bvals.append(base_bench["pope"].get("accuracy", 0))
        trained_bvals.append(trained_bench["pope"].get("accuracy", 0))
        bench_keys.append("pope_f1")
        bench_labels.append("POPE F1")
        base_bvals.append(base_bench["pope"].get("f1", 0))
        trained_bvals.append(trained_bench["pope"].get("f1", 0))
    if trained_bench.get("mmhal") and base_bench.get("mmhal"):
        bench_keys.append("mmhal_acc")
        bench_labels.append("MMHal KW-Acc")
        base_bvals.append(base_bench["mmhal"].get("keyword_accuracy", 0))
        trained_bvals.append(trained_bench["mmhal"].get("keyword_accuracy", 0))

    if bench_labels:
        x = np.arange(len(bench_labels))
        w = 0.35
        ax.bar(x - w / 2, base_bvals, w, label="Base", alpha=0.8)
        ax.bar(x + w / 2, trained_bvals, w, label="Trained", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(bench_labels, rotation=15)
        ax.set_title("Benchmark Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
    else:
        ax.text(0.5, 0.5, "Benchmark data unavailable",
                ha="center", va="center", transform=ax.transAxes)

    plt.tight_layout()
    eval_dir = os.path.join(output_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    save_path = os.path.join(eval_dir, "comparison.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[PostEval] Comparison chart saved to {save_path}")


# ---------------------------------------------------------------------------
# Benchmark runner helper (reuses eval.py evaluators)
# ---------------------------------------------------------------------------
def _run_benchmarks(model, processor, eval_dir, prefix):
    """Run POPE + MMHal, return dict with results. Gracefully skip if data missing."""
    from eval import POPEEvaluator, MMHalEvaluator, POPE_DATA_PATH, MMHAL_DATA_PATH

    bench = {"pope": None, "mmhal": None}

    # POPE
    try:
        pope = POPEEvaluator(POPE_DATA_PATH, "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/smuggle-traningcode/data/test_images")
        if pope.data:
            pope_prefix = os.path.join(eval_dir, f"{prefix}")
            pope.evaluate(model, processor, prefix=pope_prefix)
            pope_file = f"{pope_prefix}pope_score.json"
            if os.path.exists(pope_file):
                with open(pope_file) as f:
                    bench["pope"] = json.load(f)
    except Exception as e:
        print(f"[PostEval] POPE skipped: {e}")

    # MMHal
    try:
        mmhal = MMHalEvaluator(MMHAL_DATA_PATH, "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/smuggle-traningcode/data/test_images")
        if mmhal.data:
            mmhal_prefix = os.path.join(eval_dir, f"{prefix}")
            mmhal.evaluate(model, processor, prefix=mmhal_prefix)
            mmhal_file = f"{mmhal_prefix}mmhal_results.json"
            if os.path.exists(mmhal_file):
                with open(mmhal_file) as f:
                    bench["mmhal"] = json.load(f)
    except Exception as e:
        print(f"[PostEval] MMHal skipped: {e}")

    return bench


# ---------------------------------------------------------------------------
# Main entry point: called from aurora_train.py after training
# ---------------------------------------------------------------------------
def run_post_training_eval(vlm, verifier, tools, vlm_path, output_dir,
                           data_dir, training_metrics=None):
    """
    Full post-training evaluation pipeline:
    1. AURORA eval with trained model
    2. POPE/MMHal with trained model
    3. Unload trained VLM, load base VLM
    4. AURORA eval with base model
    5. POPE/MMHal with base model
    6. Plot training curves + comparison charts
    """
    eval_dir = os.path.join(output_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    from models import _unwrap_model, VLMModel

    device = next(_unwrap_model(vlm.model).parameters()).device

    # --- 1. Trained model: AURORA eval ---
    print("[PostEval] === Evaluating TRAINED model ===", flush=True)
    evaluator = AuroraEvaluator(vlm, verifier, tools)
    trained_aurora = evaluator.evaluate(data_dir, label="trained")
    if trained_aurora:
        with open(os.path.join(eval_dir, "aurora_trained.json"), "w") as f:
            json.dump(trained_aurora, f, indent=2)

    # --- 2. Trained model: POPE/MMHal ---
    trained_bench = {"pope": None, "mmhal": None}
    try:
        raw_model = _unwrap_model(vlm.model)
        trained_bench = _run_benchmarks(
            raw_model, vlm.processor, eval_dir, "trained_")
    except Exception as e:
        print(f"[PostEval] Trained benchmarks skipped: {e}")

    # --- 3. Unload trained VLM, load base ---
    print("[PostEval] Unloading trained VLM...", flush=True)
    del evaluator
    del vlm
    gc.collect()
    torch.cuda.empty_cache()

    print(f"[PostEval] Loading base VLM from {vlm_path}...", flush=True)
    base_vlm = VLMModel(model_name=vlm_path, device=device)
    _unwrap_model(base_vlm.model).eval()

    # --- 4. Base model: AURORA eval ---
    print("[PostEval] === Evaluating BASE model ===", flush=True)
    base_evaluator = AuroraEvaluator(base_vlm, verifier, tools)
    base_aurora = base_evaluator.evaluate(data_dir, label="base")
    if base_aurora:
        with open(os.path.join(eval_dir, "aurora_base.json"), "w") as f:
            json.dump(base_aurora, f, indent=2)

    # --- 5. Base model: POPE/MMHal ---
    base_bench = {"pope": None, "mmhal": None}
    try:
        raw_base = _unwrap_model(base_vlm.model)
        base_bench = _run_benchmarks(
            raw_base, base_vlm.processor, eval_dir, "base_")
    except Exception as e:
        print(f"[PostEval] Base benchmarks skipped: {e}")

    # --- 6. Unload base VLM ---
    del base_evaluator, base_vlm
    gc.collect()
    torch.cuda.empty_cache()

    # --- 7. Plot training curves ---
    if training_metrics:
        plot_training_curves(training_metrics, output_dir)

    # --- 8. Plot comparison ---
    plot_comparison(trained_aurora, base_aurora,
                    trained_bench, base_bench, output_dir)

    print("[PostEval] All evaluations complete.", flush=True)
