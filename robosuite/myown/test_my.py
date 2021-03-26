import numpy as np
from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *
from robosuite.environments.manipulation.lift_cloth import LiftCloth
from robosuite.environments.manipulation.lift import Lift
from robosuite.environments.manipulation.straighten_rope import StraightenRope
from robosuite.environments.base import register_env
import time

# create environment instance
register_env(StraightenRope)
env = suite.make(
    env_name='StraightenRope', 
    robots='Panda', 
    controller_configs=load_controller_config(default_controller='OSC_POSE'),
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    reward_shaping=True,
)

# reset the environment
env.reset()
for i in range(1000):
    action = np.random.randn(env.robots[0].action_dim) # sample random action
    # action = np.zeros(env.robots[0].action_dim)
    # if not i%100 and not i == 0:
    #     env.reset()
    obs, reward, done, info = env.step(action)  # take action in the environment
    env.render()  # render on display