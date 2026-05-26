import os
os.environ["PYTORCH_JIT"] = "0"
from runner.cmes_ai_runner import run_ai

if __name__ == '__main__':
    run_ai("inference.opt")