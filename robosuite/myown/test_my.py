import numpy as np
from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *
from robosuite.environments.manipulation.lift_cloth import LiftCloth
from robosuite.environments.manipulation.lift import Lift
from robosuite.environments.base import register_env
import time

# create environment instance
print()
env = suite.make(
    env_name='LiftCloth', # try with other tasks like "Stack" and "Door"
    robots='Panda',  # try with other robots like "Sawyer" and "Jaco"
    controller_configs=load_controller_config(default_controller='OSC_POSE'),
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    reward_shaping=True,
)

# reset the environment
env.reset()
# env.sim.data.xfrc_applied[env.mid1_id, :3] = np.array([0,0,-2])
# env.sim.data.xfrc_applied[env.mid2_id, :3] = np.array([0,0,-2])

# # Force the corners move randomly
# env.sim.data.xfrc_applied[env.corner_id, :3] = np.random.uniform(-.3,.3,size=3)
# _, _, _, _ = env.step(np.zeros(env.robots[0].action_dim))
# env.sim.data.xfrc_applied[:, :3] = np.zeros((3,))
# print(env.robots[0].action_dim)
for i in range(1000):
    action = np.random.randn(env.robots[0].action_dim) # sample random action
    # action = np.zeros(env.robots[0].action_dim)
    # if not i%100 and not i == 0:
    #     env.reset()
    obs, reward, done, info = env.step(action)  # take action in the environment
    env.render()  # render on display