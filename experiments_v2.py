import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sumo_rl import SumoEnvironment
from stable_baselines3 import PPO
from experiments import prepare, NET_STATIC, NET_ACT  # prepare() создаёт actuated-сеть

# (вероятность появления машины в секунду: север-юг, запад-восток)
LOADS = {
    "low": (0.10, 0.25),
    "medium": (0.20, 0.50),
    "high": (0.26, 0.65),
    "symmetric": (0.35, 0.35),
}
SEEDS = [0, 1, 2, 3, 4]
METHODS = ["fixed", "actuated", "maxpressure", "ppo"]
LABELS = {"fixed": "Фиксированный", "actuated": "Actuated",
          "maxpressure": "Max-Pressure", "ppo": "PPO"}
LOAD_LABELS = {"low": "Низкая", "medium": "Средняя",
               "high": "Высокая", "symmetric": "Симметричная"}
NS_LANES = ["n_t_0", "n_t_1"]  # фаза 0 (GGrr)
WE_LANES = ["w_t_0", "w_t_1"]  # фаза 1 (rrGG)


def write_routes():
    for name, (ns, we) in LOADS.items():
        with open(f"networks/routes_{name}.rou.xml", "w") as f:
            f.write(
                "<routes>\n"
                '    <route id="route_ns" edges="n_t t_s"/>\n'
                '    <route id="route_we" edges="w_t t_e"/>\n'
                f'    <flow id="flow_ns" route="route_ns" begin="0" end="100000" probability="{ns}" departSpeed="max" departPos="base" departLane="best"/>\n'
                f'    <flow id="flow_we" route="route_we" begin="0" end="100000" probability="{we}" departSpeed="max" departPos="base" departLane="best"/>\n'
                "</routes>\n"
            )


def max_pressure_action(env):
    ns = sum(env.sumo.lane.getLastStepVehicleNumber(l) for l in NS_LANES)
    we = sum(env.sumo.lane.getLastStepVehicleNumber(l) for l in WE_LANES)
    if ns > we:
        return 0
    if we > ns:
        return 1
    return env.traffic_signals["t"].green_phase  # при равенстве оставляем текущую фазу


def run_episode(method, load, seed, model=None):
    env = SumoEnvironment(
        net_file=NET_ACT if method == "actuated" else NET_STATIC,
        route_file=f"networks/routes_{load}.rou.xml",
        use_gui=False,
        num_seconds=3600,
        single_agent=True,
        fixed_ts=method in ("fixed", "actuated"),
        sumo_seed=seed,
        sumo_warnings=False,
    )
    obs, _ = env.reset()
    seen = set()
    wait, wait_mean, stopped, speed = [], [], [], []
    queue_total = [] 
    done = False
    while not done:
        if method == "ppo":
            action, _ = model.predict(obs, deterministic=True)
        elif method == "maxpressure":
            action = max_pressure_action(env)
        else:
            action = 0
        obs, _, term, trunc, info = env.step(action)
        queue_total.append(info["system_total_stopped"] + len(env.sumo.simulation.getPendingVehicles()))
        done = term or trunc
        seen.update(env.sumo.vehicle.getIDList())
        wait.append(info["system_total_waiting_time"])
        wait_mean.append(info["system_mean_waiting_time"])
        stopped.append(info["system_total_stopped"])
        speed.append(info["system_mean_speed"])
    throughput = len(seen) - len(env.sumo.vehicle.getIDList())
    backlog = len(env.sumo.simulation.getPendingVehicles())
    env.close()
    row = {
        "method": method, "load": load, "seed": seed,
        "wait_time": np.mean(wait),
        "mean_wait_per_vehicle": np.mean(wait_mean),
        "stopped": np.mean(stopped),
        "mean_speed": np.mean(speed),
        "throughput": throughput,
        "backlog": backlog,
        "queue_total": np.mean(queue_total),
    }
    return row, queue_total


def bar_plot(summary, metric, ylabel, fname):
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.2
    x = np.arange(len(LOADS))
    for i, m in enumerate(METHODS):
        means = [summary.loc[(l, m), (metric, "mean")] for l in LOADS]
        stds = [summary.loc[(l, m), (metric, "std")] for l in LOADS]
        bars = ax.bar(x + (i - 1.5) * width, means, width, yerr=stds,
                      capsize=3, label=LABELS[m])
        ax.bar_label(bars, fmt="%.1f", fontsize=7, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels([LOAD_LABELS[l] for l in LOADS])
    ax.set_xlabel("Нагрузка")
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"results/summary/{fname}", dpi=200)
    plt.close(fig)


def queue_plot(series, load, fname):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for m in METHODS:
        y = series[(m, load)]
        ax.plot(np.arange(len(y)) * 5, y, label=LABELS[m], linewidth=1.2)
    ax.set_xlabel("Время симуляции, с")
    ax.set_ylabel("Остановившиеся + ожидающие въезда, машин")
    ax.set_title(f"Нагрузка: {LOAD_LABELS[load].lower()}, seed 0")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"results/summary/{fname}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    prepare()
    write_routes()  # перезаписывает маршруты новыми уровнями нагрузки
    os.makedirs("results/summary", exist_ok=True)
    model = PPO.load("ppo_single")
    rows, series = [], {}
    for load in LOADS:
        for method in METHODS:
            for seed in SEEDS:
                row, stopped = run_episode(method, load, seed, model)
                rows.append(row)
                if seed == 0:
                    series[(method, load)] = stopped
                print(row)
    df = pd.DataFrame(rows)
    df.to_csv("results/summary/v2_raw.csv", index=False)
    metrics = ["wait_time", "mean_wait_per_vehicle", "stopped",
               "mean_speed", "throughput", "backlog","queue_total"]
    summary = df.groupby(["load", "method"])[metrics].agg(["mean", "std"])
    summary.to_csv("results/summary/v2_summary.csv")
    print(summary.round(2))
    bar_plot(summary, "mean_wait_per_vehicle",
             "Среднее ожидание на одну машину, с", "v2_wait_per_vehicle.png")
    bar_plot(summary, "throughput",
             "Пропускная способность, машин за час", "v2_throughput.png")
    bar_plot(summary, "mean_speed",
             "Средняя скорость, м/с", "v2_speed.png")
    queue_plot(series, "high", "v2_queue_high.png")
    queue_plot(series, "symmetric", "v2_queue_symmetric.png")
