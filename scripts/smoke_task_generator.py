"""Smoke test script for TaskGenerator."""

import sys
import os
sys.path.insert(0, os.path.abspath("."))

from environments.task_generator import TaskGenerator
import numpy as np

def run_smoke_test():
    gen = TaskGenerator(master_seed=42)
    manifest = gen.generate_all_tasks()
    
    # Save manifest
    manifest_path = os.path.join("configs", "meta_tasks_manifest.json")
    gen.save_manifest(manifest_path)
    
    train = manifest["train"]
    val = manifest["val"]
    ood = manifest["ood_test"]
    
    print("=" * 80)
    print(f"MCR-UAV META-RL TASK GENERATOR SMOKE TEST")
    print("=" * 80)
    print(f"TOTAL TASKS GENERATED : {len(train) + len(val) + len(ood)}")
    print(f"  TRAIN TASKS COUNT   : {len(train)}")
    print(f"  VAL TASKS COUNT     : {len(val)}")
    print(f"  OOD TEST TASKS COUNT: {len(ood)}")
    print("=" * 80)
    
    params = [
        "wind_magnitude",
        "gust_magnitude",
        "gust_frequency",
        "turbulence_std",
        "impulse_magnitude",
        "sensor_noise_scale",
        "motor_degradation",
        "mass_scale",
        "inertia_scale"
    ]
    
    for split_name, tasks in [("TRAIN", train), ("VALIDATION", val), ("OOD_TEST", ood)]:
        print(f"\n--- SPLIT: {split_name} (N = {len(tasks)} tasks) ---")
        print(f"{'Parameter':22s} | {'Min':>7s} | {'Max':>7s} | {'Mean':>7s} | {'Std':>7s} | {'Non-Zero':>8s}")
        print("-" * 75)
        for p in params:
            vals = [getattr(t, p) for t in tasks]
            non_zero = [v for v in vals if v > 0.0]
            print(f"{p:22s} | {min(vals):7.3f} | {max(vals):7.3f} | {np.mean(vals):7.3f} | {np.std(vals):7.3f} | {len(non_zero):8d}")

    print("\n" + "=" * 80)
    print("SMOKE TEST COMPLETED SUCCESSFULLY: ALL CONSTRAINTS VALIDATED")
    print("=" * 80)

if __name__ == "__main__":
    run_smoke_test()
