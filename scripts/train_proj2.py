import os
import sys
import runpy
import gymnasium as gym

import proj2

spec = gym.spec("Isaac-HuanPJ2-v0")
print("STAND REGISTRY KWARGS:", spec.kwargs)


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(THIS_DIR, "scripts", "reinforcement_learning", "rsl_rl")

if TRAIN_DIR not in sys.path:
    sys.path.insert(0, TRAIN_DIR)

sys.argv = ["train.py"] + sys.argv[1:]
runpy.run_path(os.path.join(TRAIN_DIR, "train.py"), run_name="__main__")
