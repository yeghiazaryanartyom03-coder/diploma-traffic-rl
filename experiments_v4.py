"""Эксперименты v4.
* PPO: 3 варианта x 3 сида обучения (models/ppo_*_s{0,1,2}.zip);
* базовые методы: fixed, Вебстер, actuated (maxDur 30/50/80), Max-Pressure;
* 7 сценариев: 4 прежних + 3 «невиденных» (x0.8, x1.15, зеркальный);
* новые метрики: queue_final (очередь в конце часа), switches (переключения фаз).
Запуск:
    python experiments_v4.py --quick                  # быстрая проверка
    python experiments_v4.py --eval-seeds 3 --workers 1
"""
import argparse
import concurrent.futures as cf
import glob
import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sumo_rl import SumoEnvironment
from experiments import NET_STATIC
from experiments_v2 import max_pressure_action
from webster import webster_plan, webster_tl_xml
# (вероятность появления машины в секунду: север-юг, запад-восток)
LOADS = {
    "low": (0.10, 0.25),
    "medium": (0.20, 0.50),
    "high": (0.26, 0.65),
    "symmetric": (0.35, 0.35),
    "unseen_08": (0.16, 0.40),        # 0.8 x средней
    "unseen_115": (0.23, 0.575),      # 1.15 x средней
    "unseen_mirror": (0.50, 0.20),    # зеркальный: доминирует север-юг
}
LOAD_LABELS = {"low": "Низкая", "medium": "Средняя", "high": "Высокая",
               "symmetric": "Симметричная", "unseen_08": "×0,8 (новая)",
               "unseen_115": "×1,15 (новая)", "unseen_mirror": "Зеркальная (новая)"}
BASELINES = ["fixed", "webster", "actuated30", "actuated50", "actuated80",
             "maxpressure"]
PPO_PATTERNS = {"ppo_single": "models/ppo_single_default_s*.zip",
                "ppo_mixed": "models/ppo_mixed_default_s*.zip",
                "ppo_backlog": "models/ppo_mixed_backlog_s*.zip"}
LABELS = {"fixed": "Фиксированный", "webster": "Вебстер",
          "actuated30": "Actuated (30)", "actuated50": "Actuated (50)",
          "actuated80": "Actuated (80)", "maxpressure": "Max-Pressure",
          "ppo_single": "PPO (1 нагрузка)", "ppo_mixed": "PPO (смесь)",
          "ppo_backlog": "PPO (смесь + очередь)"}
MAIN = ["fixed", "webster", "actuated50", "maxpressure",
        "ppo_single", "ppo_mixed", "ppo_backlog"]
TL = re.compile(r"<tlLogic.*?</tlLogic>", flags=re.S)
MODEL_CACHE = {}
def write_routes():
    os.makedirs("networks", exist_ok=True)
    for name, (ns, we) in LOADS.items():
        with open(f"networks/routes_{name}.rou.xml", "w") as f:
            f.write(
                "<routes>\n"
                '    <route id="route_ns" edges="n_t t_s"/>\n'
                '    <route id="route_we" edges="w_t t_e"/>\n'
                f'    <flow id="flow_ns" route="route_ns" begin="0" end="100000" probability="{ns}" departSpeed="max" departPos="base" departLane="best"/>\n'
                f'    <flow id="flow_we" route="route_we" begin="0" end="100000" probability="{we}" departSpeed="max" departPos="base" departLane="best"/>\n'
                "</routes>\n")
def _write_net(path, tl_xml):
    xml = open(NET_STATIC).read()
    new = TL.sub(tl_xml, xml)
    assert new != xml, "tlLogic не найден"
    open(path, "w").write(new)
    return path
def make_nets():
    nets = {}
    for max_dur in (30, 50, 80):
        tl = ('<tlLogic id="t" type="actuated" programID="0" offset="0">\n'
              f'        <phase duration="30" minDur="5" maxDur="{max_dur}" state="GGrr"/>\n'
              '        <phase duration="2" state="yyrr"/>\n'
              f'        <phase duration="30" minDur="5" maxDur="{max_dur}" state="rrGG"/>\n'
              '        <phase duration="2" state="rryy"/>\n'
              '    </tlLogic>')
        nets[f"actuated{max_dur}"] = _write_net(
            f"networks/single-intersection-actuated-{max_dur}.net.xml", tl)
    for load, (ns, we) in LOADS.items():
        g_ns, g_we, C = webster_plan(ns, we)
        print(f"Вебстер [{load}]: NS={g_ns} с, WE={g_we} с, цикл={C:.0f} с")
        nets[("webster", load)] = _write_net(
            f"networks/single-intersection-webster-{load}.net.xml",
            webster_tl_xml(g_ns, g_we))
    return nets
def build_tasks(loads, eval_seeds, nets):
    tasks = []
    ppo = {}
    for variant, pattern in PPO_PATTERNS.items():
        paths = sorted(glob.glob(pattern))
        ppo[variant] = paths
        print(f"{variant}: найдено моделей {len(paths)}")
    for load in loads:
        for seed in eval_seeds:
            for m in BASELINES:
                if m == "webster":
                    net = nets[("webster", load)]
                elif m.startswith("actuated"):
                    net = nets[m]
                else:
                    net = NET_STATIC
                tasks.append(dict(variant=m, method=m, load=load, seed=seed,
                                  net=net, model=None, train_seed=-1))
            for variant, paths in ppo.items():
                for p in paths:
                    ts = int(re.search(r"_s(\d+).*\.zip$", p).group(1))
                    tasks.append(dict(variant=variant, method="ppo", load=load,
                                      seed=seed, net=NET_STATIC, model=p,
                                      train_seed=ts))
    return tasks
def get_model(path):
    if path not in MODEL_CACHE:
        import torch
        from stable_baselines3 import PPO
        torch.set_num_threads(1)
        MODEL_CACHE[path] = PPO.load(path, device="cpu")
    return MODEL_CACHE[path]
def get_green_group(env):
    """Return 0 for NS green, 1 for WE green, None during yellow/all-red.

    The network uses signal states GGrr/yyrr/rrGG/rryy. Reading the actual
    signal state works for both SUMO-controlled baselines and RL controllers.
    """
    state = env.sumo.trafficlight.getRedYellowGreenState("t")
    state = state.upper()
    ns_green = any(c == "G" for c in state[:2])
    we_green = any(c == "G" for c in state[2:4])
    if ns_green and not we_green:
        return 0
    if we_green and not ns_green:
        return 1
    return None

def run_task(t):
    m = t["method"]
    env = SumoEnvironment(
        net_file=t["net"], route_file=f"networks/routes_{t['load']}.rou.xml",
        use_gui=False, num_seconds=3600, single_agent=True,
        fixed_ts=m != "ppo" and m != "maxpressure",
        sumo_seed=t["seed"], sumo_warnings=False)
    model = get_model(t["model"]) if m == "ppo" else None
    obs, _ = env.reset()
    seen = set()
    wait_mean = []
    stopped = []
    speed = []
    queue_network = []
    queue_pending = []
    queue_total = []
    switches = 0
    last_group = get_green_group(env)
    done = False
    while not done:
        if m == "ppo":
            action, _ = model.predict(obs, deterministic=True)
        elif m == "maxpressure":
            action = max_pressure_action(env)
        else:
            action = 0
        obs, _, term, trunc, info = env.step(action)
        done = term or trunc
        seen.update(env.sumo.vehicle.getIDList())
        group = get_green_group(env)
        # Yellow/all-red states return None and do not count as a phase change.
        # Count only when the next green direction differs from the previous one.
        if group is not None:
            if last_group is not None and group != last_group:
                switches += 1
            last_group = group
        wait_mean.append(info["system_mean_waiting_time"])
        stopped.append(info["system_total_stopped"])
        speed.append(info["system_mean_speed"])
        pending = len(env.sumo.simulation.getPendingVehicles())
        network_queue = info["system_total_stopped"]
        queue_network.append(network_queue)
        queue_pending.append(pending)
        queue_total.append(network_queue + pending)
    throughput = len(seen) - len(env.sumo.vehicle.getIDList())
    backlog = len(env.sumo.simulation.getPendingVehicles())
    env.close()
    row = dict(
        variant=t["variant"],
        load=t["load"],
        seed=t["seed"],
        train_seed=t["train_seed"],
        mean_wait_active=np.mean(wait_mean),
        stopped=np.mean(stopped),
        mean_speed=np.mean(speed),
        throughput=throughput,
        backlog=backlog,
        queue_network=np.mean(queue_network),
        queue_pending=np.mean(queue_pending),
        queue_total=np.mean(queue_total),
        queue_final=np.mean(queue_total[-10:]),
        switches=switches
    )
    keep = t["seed"] == 0 and t["train_seed"] in (-1, 0)
    return row, (queue_total if keep else None)
def bar_plot(summary, loads, variants, metric, ylabel, fname):
    fig, ax = plt.subplots(figsize=(14, 5.5))
    n = len(variants)
    width = 0.8 / n
    x = np.arange(len(loads))
    for i, v in enumerate(variants):
        means = [summary.loc[(l, v), (metric, "mean")] for l in loads]
        stds = [summary.loc[(l, v), (metric, "std")] for l in loads]
        ax.bar(x + (i - (n - 1) / 2) * width, means, width, yerr=stds,
               capsize=2, label=LABELS[v])
    ax.set_xticks(x)
    ax.set_xticklabels([LOAD_LABELS[l] for l in loads], fontsize=8)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(f"results/summary/{fname}", dpi=200)
    plt.close(fig)
def queue_plot(series, variants, load, fname):
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for v in variants:
        if (v, load) in series:
            y = series[(v, load)]
            ax.plot(np.arange(len(y)) * 5, y, label=LABELS[v], linewidth=1.1)
    ax.set_xlabel("Время симуляции, с")
    ax.set_ylabel("Остановившиеся + ожидающие въезда, машин")
    ax.set_title(f"Нагрузка: {LOAD_LABELS[load].lower()}, seed 0")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"results/summary/{fname}", dpi=200)
    plt.close(fig)
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    os.makedirs("results/summary", exist_ok=True)
    write_routes()
    nets = make_nets()
    loads = ["medium", "unseen_mirror"] if args.quick else list(LOADS)
    eval_seeds = [0] if args.quick else list(range(args.eval_seeds))
    tasks = build_tasks(loads, eval_seeds, nets)
    print(f"Всего эпизодов: {len(tasks)}")
    rows, series = [], {}
    if args.workers > 1:
        with cf.ProcessPoolExecutor(max_workers=args.workers) as ex:
            results = ex.map(run_task, tasks)
            for k, (t, (row, q)) in enumerate(zip(tasks, results), 1):
                rows.append(row)
                if q is not None:
                    series[(t["variant"], t["load"])] = q
                print(f"[{k}/{len(tasks)}]", t["variant"], t["load"], t["seed"])
    else:
        for k, t in enumerate(tasks, 1):
            row, q = run_task(t)
            rows.append(row)
            if q is not None:
                series[(t["variant"], t["load"])] = q
            print(f"[{k}/{len(tasks)}]", t["variant"], t["load"], t["seed"],
                  round(row["queue_total"], 2))
    df = pd.DataFrame(rows)
    df.to_csv("results/summary/v4_raw.csv", index=False)
    metrics = [
        "mean_wait_active",
        "stopped",
        "mean_speed",
        "throughput",
        "backlog",
        "queue_network",
        "queue_pending",
        "queue_total",
        "queue_final",
        "switches"
    ]
    summary = df.groupby(["load", "variant"])[metrics].agg(["mean", "std"])
    summary.to_csv("results/summary/v4_summary.csv")
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(summary.round(2))
    present = [v for v in MAIN if v in set(df["variant"])]
    bar_plot(summary, loads, present, "queue_total",
             "Остановившиеся + ожидающие въезда, машин", "v4_queue_total.png")
    bar_plot(summary, loads, present, "queue_final",
             "Очередь в конце часа, машин", "v4_queue_final.png")
    bar_plot(summary, loads, present, "switches",
             "Число переключений фаз за час", "v4_switches.png")
    for ld in ("high", "unseen_mirror"):
        if ld in loads:
            queue_plot(series, present, ld, f"v4_queue_{ld}.png")
