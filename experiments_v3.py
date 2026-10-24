"""Эксперименты v3: честное сравнение 7 методов на 4 нагрузках, 5 seed.

Методы: fixed, webster, actuated, maxpressure,
        ppo_single (старый), ppo_mixed (смесь нагрузок),
        ppo_mixed_backlog (смесь нагрузок + награда с учётом очереди на въезде).
Модели ищутся в models/*.zip; отсутствующие пропускаются.
"""
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from sumo_rl import SumoEnvironment

from experiments import NET_ACT, NET_STATIC, prepare
from experiments_v2 import (LOAD_LABELS, LOADS, SEEDS, max_pressure_action,
                            write_routes)
from webster import webster_plan, webster_tl_xml

METHODS = ["fixed", "webster", "actuated", "maxpressure",
           "ppo_single", "ppo_mixed", "ppo_mixed_backlog"]
LABELS = {"fixed": "Фиксированный", "webster": "Вебстер", "actuated": "Actuated",
          "maxpressure": "Max-Pressure", "ppo_single": "PPO (1 нагрузка)",
          "ppo_mixed": "PPO (смесь)", "ppo_mixed_backlog": "PPO (смесь + очередь)"}
MODEL_PATHS = {"ppo_single": "ppo_single",
               "ppo_mixed": "models/ppo_mixed_default",
               "ppo_mixed_backlog": "models/ppo_mixed_backlog"}
STATIC_TL = re.compile(r"<tlLogic.*?</tlLogic>", flags=re.S)


def webster_net(load):
    ns, we = LOADS[load]
    g_ns, g_we, C = webster_plan(ns, we)
    print(f"Вебстер [{load}]: NS={g_ns} с, WE={g_we} с, цикл={C:.0f} с (+8 с потерь)")
    xml = open(NET_STATIC).read()
    new = STATIC_TL.sub(webster_tl_xml(g_ns, g_we), xml)
    assert new != xml, "tlLogic не найден"
    path = f"networks/single-intersection-webster-{load}.net.xml"
    open(path, "w").write(new)
    return path


def run_episode(method, load, seed, models, net_paths):
    if method == "webster":
        net = net_paths[load]
    elif method == "actuated":
        net = NET_ACT
    else:
        net = NET_STATIC
    env = SumoEnvironment(
        net_file=net, route_file=f"networks/routes_{load}.rou.xml",
        use_gui=False, num_seconds=3600, single_agent=True,
        fixed_ts=method in ("fixed", "webster", "actuated"),
        sumo_seed=seed, sumo_warnings=False)
    obs, _ = env.reset()
    seen = set()
    wait, wait_mean, stopped, speed, queue_total = [], [], [], [], []
    done = False
    while not done:
        if method.startswith("ppo"):
            action, _ = models[method].predict(obs, deterministic=True)
        elif method == "maxpressure":
            action = max_pressure_action(env)
        else:
            action = 0
        obs, _, term, trunc, info = env.step(action)
        done = term or trunc
        seen.update(env.sumo.vehicle.getIDList())
        wait.append(info["system_total_waiting_time"])
        wait_mean.append(info["system_mean_waiting_time"])
        stopped.append(info["system_total_stopped"])
        speed.append(info["system_mean_speed"])
        queue_total.append(info["system_total_stopped"]
                           + len(env.sumo.simulation.getPendingVehicles()))
    throughput = len(seen) - len(env.sumo.vehicle.getIDList())
    backlog = len(env.sumo.simulation.getPendingVehicles())
    env.close()
    row = {"method": method, "load": load, "seed": seed,
           "wait_time": np.mean(wait), "mean_wait_per_vehicle": np.mean(wait_mean),
           "stopped": np.mean(stopped), "mean_speed": np.mean(speed),
           "throughput": throughput, "backlog": backlog,
           "queue_total": np.mean(queue_total)}
    return row, queue_total


def bar_plot(summary, methods, metric, ylabel, fname):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    n = len(methods)
    width = 0.8 / n
    x = np.arange(len(LOADS))
    for i, m in enumerate(methods):
        means = [summary.loc[(l, m), (metric, "mean")] for l in LOADS]
        stds = [summary.loc[(l, m), (metric, "std")] for l in LOADS]
        ax.bar(x + (i - (n - 1) / 2) * width, means, width, yerr=stds,
               capsize=2, label=LABELS[m])
    ax.set_xticks(x)
    ax.set_xticklabels([LOAD_LABELS[l] for l in LOADS])
    ax.set_xlabel("Нагрузка")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"results/summary/{fname}", dpi=200)
    plt.close(fig)


def queue_plot(series, methods, load, fname):
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for m in methods:
        y = series[(m, load)]
        ax.plot(np.arange(len(y)) * 5, y, label=LABELS[m], linewidth=1.1)
    ax.set_xlabel("Время симуляции, с")
    ax.set_ylabel("Остановившиеся + ожидающие въезда, машин")
    ax.set_title(f"Нагрузка: {LOAD_LABELS[load].lower()}, seed 0")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"results/summary/{fname}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    prepare()
    write_routes()
    os.makedirs("results/summary", exist_ok=True)
    net_paths = {l: webster_net(l) for l in LOADS}

    models = {}
    for m, path in MODEL_PATHS.items():
        if os.path.exists(path + ".zip"):
            models[m] = PPO.load(path)
        else:
            print(f"[пропуск] модель {path}.zip не найдена, метод {m} не запускается")
    methods = [m for m in METHODS if not m.startswith("ppo") or m in models]

    rows, series = [], {}
    total = len(LOADS) * len(methods) * len(SEEDS)
    k = 0
    for load in LOADS:
        for method in methods:
            for seed in SEEDS:
                row, q = run_episode(method, load, seed, models, net_paths)
                rows.append(row)
                if seed == 0:
                    series[(method, load)] = q
                k += 1
                print(f"[{k}/{total}]", {key: (round(v, 2) if isinstance(v, float) else v)
                                        for key, v in row.items()})

    df = pd.DataFrame(rows)
    df.to_csv("results/summary/v3_raw.csv", index=False)
    metrics = ["wait_time", "mean_wait_per_vehicle", "stopped", "mean_speed",
               "throughput", "backlog", "queue_total"]
    summary = df.groupby(["load", "method"])[metrics].agg(["mean", "std"])
    summary.to_csv("results/summary/v3_summary.csv")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(summary.round(2))
    bar_plot(summary, methods, "queue_total",
             "Остановившиеся + ожидающие въезда, машин", "v3_queue_total.png")
    bar_plot(summary, methods, "mean_wait_per_vehicle",
             "Среднее ожидание на одну машину, с", "v3_wait_per_vehicle.png")
    bar_plot(summary, methods, "throughput",
             "Пропускная способность, машин за час", "v3_throughput.png")
    queue_plot(series, methods, "high", "v3_queue_high.png")
    queue_plot(series, methods, "symmetric", "v3_queue_symmetric.png")
