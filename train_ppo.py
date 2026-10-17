from sumo_rl import SumoEnvironment
from stable_baselines3 import PPO

base = "sumo-rl/sumo_rl/nets/single-intersection/"

def make_env(fixed=False):
    return SumoEnvironment(
        net_file=base + "single-intersection.net.xml",
        route_file=base + "single-intersection.rou.xml",
        use_gui=False,
        num_seconds=3600,
        single_agent=True,
        fixed_ts=fixed,
        out_csv_name="results/ppo_single" if not fixed else "results/fixed_single",
    )

env = make_env()
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="tensorboard_logs/")
model.learn(total_timesteps=100_000)
model.save("ppo_single")
env.close()
