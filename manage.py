#!/usr/bin/env python3
import os
import sys
import subprocess
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt
import questionary

console = Console()

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_banner():
    clear_screen()
    banner_text = """
    AURORA CLI (H200 Optimized)
    ===========================
    Annotation-free Unification of Reinforcement 
    and Opponent Reasoning for Anti-hallucination
    
    Cluster: 8x H200 Detected
    """
    console.print(Panel(banner_text, style="bold cyan"))

def train_mode():
    console.print("\n[bold green]🚀 Starting Distributed Training (8x H200)...[/bold green]")
    cmd = ["accelerate", "launch", "aurora_train.py"]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        console.print("[bold red]Training failed![/bold red]")
    except KeyboardInterrupt:
        pass

def eval_mode():
    console.print("\n[bold blue]🔍 Starting Evaluation (Trained Model)...[/bold blue]")
    model_path = questionary.path("Path to model checkpoint:", default="./output/checkpoints/vlm_final").ask()
    
    # 询问是否跑 Benchmark
    run_bench = questionary.confirm("Run Benchmarks (POPE/MMHal)?", default=True).ask()
    
    cmd = ["python", "eval.py", "--model_path", model_path]
    if run_bench:
        cmd.append("--benchmarks")
        
    subprocess.run(cmd)

def baseline_mode():
    console.print("\n[bold purple]📉 Starting Baseline Evaluation (Original Qwen3-VL)...[/bold purple]")
    console.print("This will download/load the base model and run benchmarks to establish a baseline score.")
    
    run_bench = questionary.confirm("Run Benchmarks (POPE/MMHal)?", default=True).ask()
    
    # 直接调用 eval.py 带 --baseline 参数
    cmd = ["python", "eval.py", "--baseline"]
    if run_bench:
        cmd.append("--benchmarks")
        
    subprocess.run(cmd)

def setup_mode():
    console.print("\n[bold yellow]⚙️  Environment Setup[/bold yellow]")
    if questionary.confirm("Run setup_force_hf.sh to download data via HF Mirror?").ask():
        subprocess.run(["sh", "setup_force_hf.sh"])

def main():
    print_banner()
    
    while True:
        action = questionary.select(
            "What would you like to do?",
            choices=[
                "🚀 Train AURORA Model (Distributed)",
                "🔍 Evaluate Trained Model",
                "📉 Evaluate Baseline (Original Qwen3-VL)",
                "⚙️  Run Data Setup (HF Mirror)",
                "❌ Exit"
            ]
        ).ask()
        
        if "Train" in action:
            train_mode()
        elif "Evaluate Trained" in action:
            eval_mode()
        elif "Evaluate Baseline" in action:
            baseline_mode()
        elif "Setup" in action:
            setup_mode()
        elif "Exit" in action:
            console.print("Goodbye!")
            break
        
        input("\nPress Enter to continue...")
        print_banner()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[bold red]Interrupted by user.[/bold red]")
        sys.exit(0)
