import os
from sumo_rl import SumoEnvironment

base = "sumo-rl/sumo_rl/nets/single-intersection/"
env = SumoEnvironment(
    net_file=base + "single-intersection.net.xml",
    route_file=base + "single-intersection.rou.xml",
    use_gui=False,
    num_seconds=1000,
    single_agent=True,
)

obs, info = env.reset()
done = False
total = 0
while not done:
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    total += reward
    done = terminated or truncated
print("Готово, суммарная награда:", total)
env.close()
