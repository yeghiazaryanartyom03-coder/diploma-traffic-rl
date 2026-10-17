import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sumo_rl import SumoEnvironment
from stable_baselines3 import PPO

BASE = "sumo-rl/sumo_rl/nets/single-intersection/"
NET_STATIC = BASE + "single-intersection.net.xml"
NET_ACT = "networks/single-intersection-actuated.net.xml"
LOADS = {"low": 0.5, "medium": 1.0, "high": 1.3}
SEEDS = [0, 1, 2, 3, 4]
METHODS = ["fixed", "actuated", "ppo"]


def prepare():
    os.makedirs("networks", exist_ok=True)
    os.makedirs("results/summary", exist_ok=True)
    xml = open(NET_STATIC).read()
    actuated = (
        '<tlLogic id="t" type="actuated" programID="0" offset="0">\n'
        '        <phase duration="30" minDur="5" maxDur="50" state="GGrr"/>\n'
        '        <phase duration="2" state="yyrr"/>\n'
        '        <phase duration="30" minDur="5" maxDur="50" state="rrGG"/>\n'
        '        <phase duration="2" state="rryy"/>\n'
        '    </tlLogic>'
    )
    new = re.sub(r"<tlLogic.*?</tlLogic>", actuated, xml, flags=re.S)
    assert new != xml, "tlLogic не найден"
    open(NET_ACT, "w").write(new)
    for name, k in LOADS.items():
        with open(f"networks/routes_{name}.rou.xml", "w") as f:
            f.write(
                "<routes>\n"
                '    <route id="route_ns" edges="n_t t_s"/>\n'
                '    <route id="route_we" edges="w_t t_e"/>\n'
                f'    <flow id="flow_ns" route="route_ns" begin="0" end="100000" probability="{0.2*k:.3f}" departSpeed="max" departPos="base" departLane="best"/>\n'
                f'    <flow id="flow_we" route="route_we" begin="0" end="100000" probability="{0.5*k:.3f}" departSpeed="max" departPos="base" departLane="best"/>\n'
                "</routes>\n"
            )


def run_episode(method, load, seed, model=None):
    net = NET_ACT if method == "actuated" else NET_STATIC
    env = SumoEnvironment(
        net_file=net,
        route_file=f"networks/routes_{load}.rou.xml",
        use_gui=False,
        num_seconds=3600,
        single_agent=True,
        fixed_ts=(method != "ppo"),
        sumo_seed=seed,
        sumo_warnings=False,
    )
    obs, _ = env.reset()
    wait, wait_mean, stopped, speed = [], [], [], []
    done = False
    while not done:
        if method == "ppo":
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = 0
        obs, _, term, trunc, info = env.step(action)
        done = term or trunc
        wait.append(info["system_total_waiting_time"])
        wait_mean.append(info["system_mean_waiting_time"])
        stopped.append(info["system_total_stopped"])
        speed.append(info["system_mean_speed"])
    env.close()
    return {
        "method": method, "load": load, "seed": seed,
        "wait_time": np.mean(wait),
        "mean_wait_per_vehicle": np.mean(wait_mean),
        "stopped": np.mean(stopped),
        "mean_speed": np.mean(speed),
    }


def plot(summary):
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.25
    x = np.arange(len(LOADS))
    for i, m in enumerate(METHODS):
        means = [summary.loc[(l, m), ("wait_time", "mean")] for l in LOADS]
        stds = [summary.loc[(l, m), ("wait_time", "std")] for l in LOADS]
        ax.bar(x + (i - 1) * width, means, width, yerr=stds, capsize=3, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(list(LOADS))
    ax.set_xlabel("Нагрузка")
    ax.set_ylabel("Суммарное время ожидания (среднее по эпизоду)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/summary/comparison_wait_time.png", dpi=200)


if __name__ == "__main__":
    prepare()
    model = PPO.load("ppo_single")
    rows = []
    for load in LOADS:
        for method in METHODS:
            for seed in SEEDS:
                rows.append(run_episode(method, load, seed, model))
                print(rows[-1])
    df = pd.DataFrame(rows)
    df.to_csv("results/summary/single_raw.csv", index=False)
    summary = df.groupby(["load", "method"]).agg(["mean", "std"]).drop(columns="seed")
    summary.to_csv("results/summary/single_summary.csv")
    print(summary.round(2))
    plot(summary)
