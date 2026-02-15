#!/usr/bin/env python3
import os
import sys
import subprocess
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm, IntPrompt

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
    # 使用 accelerate 启动
    cmd = ["accelerate", "launch", "aurora_train.py"]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        console.print("[bold red]Training failed![/bold red]")
    except KeyboardInterrupt:
        pass

def eval_mode():
    console.print("\n[bold blue]🔍 Starting Evaluation (Trained Model)...[/bold blue]")
    # 替换 questionary.path 为 Prompt.ask
    model_path = Prompt.ask("Path to model checkpoint", default="./output/checkpoints/vlm_final")
    
    # 替换 questionary.confirm 为 Confirm.ask
    run_bench = Confirm.ask("Run Benchmarks (POPE/MMHal)?", default=True)
    
    cmd = ["python", "eval.py", "--model_path", model_path]
    if run_bench:
        cmd.append("--benchmarks")
        
    subprocess.run(cmd)

def baseline_mode():
    console.print("\n[bold purple]📉 Starting Baseline Evaluation (Original Qwen3-VL)...[/bold purple]")
    console.print("This will download/load the base model and run benchmarks to establish a baseline score.")
    
    run_bench = Confirm.ask("Run Benchmarks (POPE/MMHal)?", default=True)
    
    cmd = ["python", "eval.py", "--baseline"]
    if run_bench:
        cmd.append("--benchmarks")
        
    subprocess.run(cmd)

def setup_mode():
    console.print("\n[bold yellow]⚙️  Environment Setup[/bold yellow]")
    console.print("Will run the resource downloader script.")
    
    # 检测当前目录下有哪些脚本可用
    scripts = []
    if os.path.exists("setup_aliyun_mirror.sh"): scripts.append("setup_aliyun_mirror.sh")
    if os.path.exists("setup_sg.sh"): scripts.append("setup_sg.sh")
    if os.path.exists("setup_force_hf.sh"): scripts.append("setup_force_hf.sh")
    
    if not scripts:
        console.print("[red]No setup script found (setup_aliyun_mirror.sh / setup_sg.sh)![/red]")
        return

    # 让用户选一个脚本跑
    console.print(f"Found scripts: {scripts}")
    script_to_run = Prompt.ask("Enter script name to run", default=scripts[0], choices=scripts)
    
    subprocess.run(["bash", script_to_run])

def main():
    while True:
        print_banner()
        
        console.print("\n[bold]Main Menu:[/bold]")
        options = [
            "🚀 Train AURORA Model (Distributed)",
            "🔍 Evaluate Trained Model",
            "📉 Evaluate Baseline (Original Qwen3-VL)",
            "⚙️  Run Data Setup (Download Data/Models)",
            "❌ Exit"
        ]
        
        # 手动打印菜单
        for i, opt in enumerate(options, 1):
            console.print(f"  [cyan]{i}.[/cyan] {opt}")
        
        console.print("")
        # 使用 rich 的 IntPrompt 替代 questionary.select
        choice = IntPrompt.ask("Select an option", choices=[str(i) for i in range(1, len(options)+1)], show_choices=False)
        
        if choice == 1:
            train_mode()
        elif choice == 2:
            eval_mode()
        elif choice == 3:
            baseline_mode()
        elif choice == 4:
            setup_mode()
        elif choice == 5:
            console.print("Goodbye!")
            break
        
        Prompt.ask("\nPress Enter to continue...", default="", show_default=False)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[bold red]Interrupted by user.[/bold red]")
        sys.exit(0)
