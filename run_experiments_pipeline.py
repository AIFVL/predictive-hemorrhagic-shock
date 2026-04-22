#!/usr/bin/env python3
"""
Simple experiment pipeline runner - validates all experiment YAMLs by running DAG.

No git commits, just sequential execution to check if configs are correct.

Usage:
    python3 run_experiments_pipeline.py
"""

import sys
import yaml
import shutil
import subprocess
from pathlib import Path
from datetime import datetime


class Colors:
    BLUE = '\033[0;34m'
    GREEN = '\033[0;32m'
    RED = '\033[0;31m'
    YELLOW = '\033[1;33m'
    RESET = '\033[0m'


def log_section(title):
    print(f"\n{Colors.BLUE}{'='*90}{Colors.RESET}")
    print(f"{Colors.BLUE}  {title}{Colors.RESET}")
    print(f"{Colors.BLUE}{'='*90}{Colors.RESET}\n")


def log_success(msg):
    print(f"{Colors.GREEN}[✓]{Colors.RESET} {msg}")


def log_error(msg):
    print(f"{Colors.RED}[✗]{Colors.RESET} {msg}")


def log_info(msg):
    print(f"{Colors.YELLOW}[*]{Colors.RESET} {msg}")


def run_cmd(cmd):
    """Run command and return exit code."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr


def check_airflow():
    """Verify Docker and Airflow are running."""
    log_info("Checking Docker & Airflow...")
    code, _, _ = run_cmd("docker compose ps --services | grep airflow-webserver")
    if code == 0:
        log_success("Airflow is running\n")
        return True
    log_error("Airflow not running. Start with: docker compose up -d\n")
    return False


def extract_metadata(yaml_file):
    """Extract version, stage, description from YAML."""
    with open(yaml_file) as f:
        cfg = yaml.safe_load(f)
    version = cfg.get('general_config', {}).get('version', '?')
    stage = cfg.get('experiment_metadata', {}).get('stage', '?')
    desc = cfg.get('experiment_metadata', {}).get('description', '?')[:60]
    return version, stage, desc


def run_experiment(exp_file, pipeline_config, dag_id, run_date):
    """Deploy config and run DAG."""
    version, stage, desc = extract_metadata(exp_file)
    
    print(f"{Colors.YELLOW}Version:{Colors.RESET} {version}")
    print(f"{Colors.YELLOW}Stage:{Colors.RESET} {stage}")
    print(f"{Colors.YELLOW}Description:{Colors.RESET} {desc}...\n")
    
    # Deploy config
    log_info("Deploying configuration...")
    shutil.copy(exp_file, pipeline_config)
    log_success("Config deployed")
    
    # Run DAG
    log_info(f"Running DAG (this may take 5-30 minutes)...\n")
    cmd = f"docker compose exec -T airflow-scheduler airflow dags test {dag_id} {run_date}"
    code, stdout, stderr = run_cmd(cmd)
    
    if code != 0:
        log_error("DAG FAILED!")
        if stderr:
            print(f"\n{Colors.RED}Error:{Colors.RESET}")
            print(stderr[-1000:])  # Last 1000 chars
        return False
    
    log_success("DAG PASSED!")
    return True


def main():
    # Setup
    exp_dir = Path("config/experiments_curated")
    pipeline_config = Path("config/pipeline_config.yaml")
    dag_id = "shock_prediction_pipeline"
    
    # Validate
    if not exp_dir.exists():
        log_error(f"Experiments dir not found: {exp_dir}")
        return 1
    if not pipeline_config.exists():
        log_error(f"Pipeline config not found: {pipeline_config}")
        return 1
    
    # Check Airflow
    if not check_airflow():
        return 1
    
    # Get experiments
    exps = sorted([f for f in exp_dir.glob("*.yaml") if f.name not in {"index.yaml", "README.md"}])
    if not exps:
        log_error(f"No experiments found in {exp_dir}")
        return 1
    
    log_section("RUNNING EXPERIMENTS - YAML VALIDATION")
    print(f"Found {len(exps)} experiments:\n")
    for e in exps:
        print(f"  • {e.name}")
    print()
    
    # Run each experiment
    passed = 0
    failed = 0
    run_date = datetime.now().strftime("%Y-%m-%d")
    
    for idx, exp_file in enumerate(exps, 1):
        print(f"\n{Colors.BLUE}{'─'*90}{Colors.RESET}")
        print(f"{Colors.BLUE}[{idx}/{len(exps)}] {exp_file.name}{Colors.RESET}")
        print(f"{Colors.BLUE}{'─'*90}{Colors.RESET}\n")
        
        if run_experiment(exp_file, pipeline_config, dag_id, run_date):
            passed += 1
        else:
            failed += 1
        
        # Wait before next
        if idx < len(exps):
            print(f"\n{Colors.YELLOW}Waiting 3 seconds before next experiment...{Colors.RESET}\n")
            import time
            time.sleep(3)
    
    # Summary
    log_section("SUMMARY")
    print(f"{Colors.GREEN}✓ Passed:{Colors.RESET} {passed}/{len(exps)}")
    print(f"{Colors.RED}✗ Failed:{Colors.RESET} {failed}/{len(exps)}\n")
    
    if failed == 0:
        print(f"{Colors.GREEN}All experiments validated successfully!{Colors.RESET}")
        return 0
    else:
        print(f"{Colors.RED}Some experiments failed. Review errors above.{Colors.RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
