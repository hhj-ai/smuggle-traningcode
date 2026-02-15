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
    AURORA CLI
    ==========
    Annotation-free Unification of Reinforcement 
    and Opponent Reasoning for Anti-hallucination
    
    Current Environment: 8x H200 Cluster (Auto-Detected)
    Dataset: YFCC100M (Auto-Download Enabled)
    """
    console.print(Panel(banner_text, style="bold cyan"))

def check_dependencies():
    try:
        import torch
        import transformers
        import accelerate
        import datasets
        import aiohttp
    except ImportError:
        console.print("[bold red]Missing dependencies![/bold red]")
        console.print("Please run: pip install -r requirements.txt")
        sys.exit(1)

def train_mode():
    console.print("\n[bold green]🚀 Starting Distributed Training...[/bold green]")
    console.print("Using `accelerate launch` for multi-GPU support.")
    
    # Configure accelerate if needed, but assuming user has run `accelerate config` or we use defaults
    # For H200 cluster, simple launch usually works.
    
    cmd = ["accelerate", "launch", "aurora_train.py"]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        console.print("[bold red]Training failed![/bold red]")
    except FileNotFoundError:
        console.print("[bold red]Error: `accelerate` command not found. Ensure it is installed.[/bold red]")

def eval_mode():
    console.print("\n[bold blue]🔍 Starting Evaluation...[/bold blue]")
    
    model_path = questionary.path("Path to model checkpoint:", default="./output/checkpoints/vlm_final").ask()
    image_dir = questionary.path("Path to test images:", default="./data/test_images").ask()
    
    cmd = ["python", "eval.py", "--model_path", model_path, "--image_dir", image_dir]
    subprocess.run(cmd)

def setup_mode():
    console.print("\n[bold yellow]⚙️  Setup Wizard[/bold yellow]")
    
    # Check data directories
    data_dir = "./data/yfcc100m"
    if not os.path.exists(data_dir):
        if questionary.confirm(f"Create YFCC100M data directory at {data_dir}?").ask():
            os.makedirs(data_dir, exist_ok=True)
            console.print(f"[green]Created {data_dir}.[/green]")
            console.print("[blue]Note: The training script will automatically download images here if empty.[/blue]")
    else:
        console.print(f"[green]Data directory found: {data_dir}[/green]")
        
    # Check output directories
    output_dir = "./output/checkpoints"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        console.print(f"[green]Created {output_dir}[/green]")

def main():
    print_banner()
    check_dependencies()
    
    while True:
        action = questionary.select(
            "What would you like to do?",
            choices=[
                "🚀 Train AURORA Model (Distributed + Auto-Download)",
                "🔍 Evaluate Model",
                "⚙️  Run Setup / Check Paths",
                "❌ Exit"
            ]
        ).ask()
        
        if "Train" in action:
            train_mode()
        elif "Evaluate" in action:
            eval_mode()
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
