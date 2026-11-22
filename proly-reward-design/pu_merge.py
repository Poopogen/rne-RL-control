#!/usr/bin/env python3
"""Run after training: merges all TensorBoard runs into a single sequential log.
Usage: python merge_tb.py
"""
import glob
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

logdir = "tensorboard"
output_dir = os.path.join(logdir, "merged")

# Collect all run subdirs, skip the merged output itself
run_dirs = sorted([
    d for d in glob.glob(f"{logdir}/*/")
    if not d.rstrip("/").endswith("merged")
])

print(f"Found {len(run_dirs)} runs to merge...")

writer = SummaryWriter(output_dir)
step_offset = 0

for run_dir in run_dirs:
    ea = EventAccumulator(run_dir)
    ea.Reload()
    scalar_tags = ea.Tags().get("scalars", [])

    run_max_step = 0
    for tag in scalar_tags:
        events = ea.Scalars(tag)
        for ev in events:
            writer.add_scalar(tag, ev.value, step_offset + ev.step)
        if events:
            run_max_step = max(run_max_step, events[-1].step)

    step_offset += run_max_step + 1

writer.close()
print(f"Done. Merged → {output_dir}  (total steps: {step_offset})")
print(f"View with: tensorboard --logdir {output_dir}")