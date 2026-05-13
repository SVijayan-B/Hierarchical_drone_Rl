import argparse
import os

from hierarchical_drone.rl.train_ppo import train
from hierarchical_drone.rl.evaluate import evaluate


def main():
    parser = argparse.ArgumentParser(description="Hierarchical autonomous drone stack")
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--vecnorm_path", type=str, default="")
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    if args.mode == "train":
        train(total_timesteps=args.timesteps, run_name=args.run_name)
    else:
        if not args.model_path or not args.vecnorm_path:
            raise ValueError("For eval mode provide --model_path and --vecnorm_path")
        evaluate(args.model_path, args.vecnorm_path, episodes=args.episodes, gui=True)


if __name__ == "__main__":
    main()
