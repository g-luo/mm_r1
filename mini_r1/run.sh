#!/bin/bash

# accelerate launch --num_processes 4 --config_file configs/deepspeed_zero3.yaml scripts/task_countdown.py --config configs/task_countdown.yaml
accelerate launch --num_processes 4 --config_file configs/deepspeed_zero3.yaml scripts/task_visual.py --config configs/task_visual.yaml