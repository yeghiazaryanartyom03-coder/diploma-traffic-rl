import pandas as pd
import glob
from sumo_rl import SumoEnvironment
from stable_baselines3 import PPO

base = "sumo-rl/sumo_rl/nets/single-intersection/"

def make_env(prefix, fixed=False):
    return SumoEnvironment(
        net_file=base + "single-intersection.net.xml",
        route_file=base + "single-intersection.rou.xml",
        use_gui=False,
        num_seconds=3600,
        single_agent=True,
        fixed_ts=fixed,
        out_csv_name=f"results/eval/{prefix}",
    )

def run_fixed(episodes=5):
    env = make_env("fixed", fixed=True)
    for _ in range(episodes):
        env.reset()
        done = False
        while not done:
            _, _, term, trunc, _ = env.step(0)
            done = term or trunc
    env.close()

def run_ppo(episodes=5):
    env = make_env("ppo")
    model = PPO.load("ppo_single")
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = env.step(action)
            done = term or trunc
    env.close()

def summarize(prefix):
    files = sorted(glob.glob(f"results/eval/{prefix}_conn*_ep*.csv"))
    rows = []
    for f in files:
        df = pd.read_csv(f)
        rows.append({
            "wait_time": df["system_total_waiting_time"].mean(),
            "stopped": df["system_total_stopped"].mean(),
            "mean_speed": df["system_mean_speed"].mean(),
        })
    return pd.DataFrame(rows).mean()

if __name__ == "__main__":
    run_fixed()
    run_ppo()
    print("Фиксированный цикл:\n", summarize("fixed"))
    print("PPO:\n", summarize("ppo"))
    pd.DataFrame({
        "fixed": summarize("fixed"),
        "ppo": summarize("ppo"),
    }).to_csv("results/summary/single_intersection.csv")
