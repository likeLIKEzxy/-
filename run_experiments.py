import os
import subprocess

def run_experiment(config_path, tag):
    print(f"\n🚀 开始运行实验：{tag}")
    env = os.environ.copy()
    env["EXPERIMENT_TAG"] = tag
    subprocess.run(["python", "-m", "src.train", "--config", config_path], env=env)

def main():
    configs = {
        "base": "src/configs/base.yaml",
        "no_posenc": "src/configs/no_posenc.yaml",
        "single_head": "src/configs/single_head.yaml",
        "no_residual": "src/configs/no_residual.yaml",
        "small_ffn": "src/configs/small_ffn.yaml",
        "lr_1e-4": "src/configs/lr_1e-4.yaml",
        "lr_1e-3": "src/configs/lr_1e-3.yaml",
    }

    for tag, path in configs.items():
        run_experiment(path, tag)

if __name__ == "__main__":
    main()
