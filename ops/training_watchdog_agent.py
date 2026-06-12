#!/usr/bin/env python3
"""
Autonomous Auditor / "Self-Repairing Brain" for Monterey Bay AI Lab.

This watchdog monitors the training `metrics.jsonl` firehose in real-time.
If it detects a structural anomaly (Loss explosion, NaN gradients, or an 
I/O throughput collapse), it pauses the environment and autonomously 
dispatches a frontier model (Gemini CLI) to investigate and fix the root cause.
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] WATCHDOG: %(message)s', datefmt='%H:%M:%S')

def get_gemini_cli_command() -> str:
    """Detect the correct command to invoke the Gemini CLI."""
    # This might be 'gemini' or 'gemini-cli' depending on the environment.
    # For this lab, we assume 'gemini' is available in the path.
    return "gemini"

def trigger_autonomous_repair(anomaly_type: str, details: str, metrics_file: Path):
    """The core agentic handover: Spawn an LLM to fix the system."""
    logging.error(f"ANOMALY DETECTED: {anomaly_type}. Triggering Autonomous Repair.")
    logging.info(f"Details: {details}")
    
    prompt = (
        f"CRITICAL ALERT from Autonomous Watchdog: A '{anomaly_type}' anomaly occurred "
        f"during production training. Details: {details}. "
        f"Please analyze the end of '{metrics_file}' to understand the failure state, "
        f"investigate the recent changes in 'trainer.py' or 'data.py', and apply a architectural fix. "
        f"Do not ask for permission, just fix the pipeline and exit."
    )
    
    # We use YOLO mode so the agent can work without human intervention
    cmd = [
        get_gemini_cli_command(),
        "run",
        "--yolo",
        prompt
    ]
    
    logging.info(f"Spawning Agent: {' '.join(cmd)}")
    try:
        # We spawn the agent and wait for it to complete the repair
        # In a fully detached environment, we might run this asynchronously
        subprocess.run(cmd, check=True)
        logging.info("Agent repair cycle completed. You may want to restart training.")
    except Exception as e:
        logging.critical(f"Failed to spawn repair agent: {e}")

def monitor_metrics(metrics_file: Path, poll_interval: int = 5):
    """Tail the metrics file and watch for systems defects."""
    if not metrics_file.exists():
        logging.info(f"Waiting for {metrics_file} to be created...")
        while not metrics_file.exists():
            time.sleep(poll_interval)
            
    logging.info(f"Tailing {metrics_file} for anomalies...")
    
    # Track state
    baseline_throughput = None
    baseline_loss = None
    
    with open(metrics_file, 'r', encoding='utf-8') as f:
        # Seek to the end if we only want to watch new events
        # f.seek(0, os.SEEK_END) 
        
        while True:
            line = f.readline()
            if not line:
                time.sleep(poll_interval)
                continue
                
            try:
                record = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
                
            # 1. Check for explicit Safety Shutdowns
            if record.get('event') == 'safety_shutdown':
                trigger_autonomous_repair(
                    "Safety Monitor Shutdown",
                    f"Reasons: {record.get('reasons')}",
                    metrics_file
                )
                break
                
            # 2. Check for NaN / Loss Explosion
            loss = record.get('loss')
            if loss is not None:
                if str(loss).lower() == 'nan' or record.get('nan_skipped') == True:
                    trigger_autonomous_repair(
                        "Gradient NaN/Divergence",
                        "The model produced NaN losses or gradients. Possible learning rate explosion or unscaled target.",
                        metrics_file
                    )
                    break
                    
                if baseline_loss is None:
                    baseline_loss = loss
                elif loss > baseline_loss * 5.0 and step > 100:
                    trigger_autonomous_repair(
                        "Loss Explosion",
                        f"Loss spiked from {baseline_loss:.2f} to {loss:.2f}.",
                        metrics_file
                    )
                    break
                else:
                    # Update baseline slowly (EMA)
                    baseline_loss = 0.9 * baseline_loss + 0.1 * loss
            
            # 3. Check for I/O Throughput Collapse
            speed = record.get('speed_bps')
            step = record.get('step', 0)
            if speed is not None and step > 500:
                if baseline_throughput is None:
                    baseline_throughput = speed
                
                # If throughput drops by 80% compared to historical baseline
                if speed < (baseline_throughput * 0.2):
                    trigger_autonomous_repair(
                        "Throughput Collapse",
                        f"I/O Speed dropped from {baseline_throughput:.1f} to {speed:.1f} batches/sec. Possible prefetcher stall, disk bottleneck, or CPU thread deadlock.",
                        metrics_file
                    )
                    break
                    
                # Update baseline
                baseline_throughput = 0.95 * baseline_throughput + 0.05 * speed

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Autonomous Training Auditor")
    parser.add_argument("--metrics", type=str, default="sota_continual_learning/output_production/metrics/metrics.jsonl")
    args = parser.parse_args()
    
    monitor_metrics(Path(args.metrics))

