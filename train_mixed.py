"""Обучение PPO на смеси нагрузок (low, medium, high, symmetric).

Запуск:
    python train_mixed.py --steps 2048 --name smoke          # быстрая проверка
    python train_mixed.py --reward default --name ppo_mixed_default
    python train_mixed.py --reward backlog --name ppo_mixed_backlog
"""
import argparse
import os
import random

import gymnasium as gym
from stable_baselines3 import PPO
from sumo_rl import SumoEnvironment

from experiments import NET_STATIC
from experiments_v2 import write_routes

TRAIN_LOADS = ["low", "medium", "high", "symmetric"]


def reward_backlog(ts):
    """Награда = -(остановившиеся + ожидающие въезда) / 10.

    В отличие от стандартной награды SUMO-RL учитывает машины, которые
    не смогли въехать в сеть, поэтому «прятать» очередь за пределами сети
    агенту невыгодно."""
    try:
        queued = ts.get_total_queued()
    except AttributeError:  # запасной вариант, если метода нет в вашей версии
        queued = sum(ts.sumo.lane.getLastStepHaltingNumber(l) for l in ts.lanes)
    pending = len(ts.sumo.simulation.getPendingVehicles())
    return -(queued + pending) / 10.0


class MixedLoadEnv(gym.Env):
    """На каждом эпизоде случайно выбирает один из сценариев нагрузки."""

    metadata = {"render_modes": []}

    def __init__(self, loads, reward_fn=None, num_seconds=3600, seed=0):
        super().__init__()
        self.loads = list(loads)
        self.rng = random.Random(seed)
        kwargs = dict(net_file=NET_STATIC, use_gui=False,
                      num_seconds=num_seconds, single_agent=True,
                      sumo_warnings=False)
        if reward_fn is not None:
            kwargs["reward_fn"] = reward_fn
        self.envs = {
            l: SumoEnvironment(route_file=f"networks/routes_{l}.rou.xml", **kwargs)
            for l in self.loads
        }
        first = self.envs[self.loads[0]]
        self.observation_space = first.observation_space
        self.action_space = first.action_space
        self.current = None
        self.load = None

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng.seed(seed)
        if self.current is not None:
            self.current.close()
        self.load = self.rng.choice(self.loads)
        self.current = self.envs[self.load]
        obs, info = self.current.reset()
        info = dict(info or {})
        info["load"] = self.load
        return obs, info

    def step(self, action):
        return self.current.step(action)

    def close(self):
        for e in self.envs.values():
            e.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reward", choices=["default", "backlog"], default="default")
    ap.add_argument("--steps", type=int, default=300_000)
    ap.add_argument("--name", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    name = args.name or f"ppo_mixed_{args.reward}"

    os.makedirs("networks", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    write_routes()

    reward_fn = reward_backlog if args.reward == "backlog" else None
    env = MixedLoadEnv(TRAIN_LOADS, reward_fn=reward_fn, seed=args.seed)
    model = PPO("MlpPolicy", env, verbose=1, seed=args.seed,
                tensorboard_log="tensorboard_logs/")
    model.learn(total_timesteps=args.steps, tb_log_name=name)
    model.save(f"models/{name}")
    env.close()
    print("Сохранено:", f"models/{name}.zip")
    