from dm_control import mjcf
from dm_control import composer
import numpy as np
import sys
from dm_control import viewer

import robosuite as suite
env = suite.make(
        env_name='Lift',
        robots='Sawyer',
        has_renderer=False,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=100,
    )
env.reset()
env_xml = env.model.get_xml()

cloth = """
<mujoco model="parent">
    <worldbody>
        <composite type="grid" count="9 9 1" spacing="0.05" offset="0.7 0 1.0">
                <geom size=".02"/>
        </composite>
    </worldbody>
</mujoco>"""

world = mjcf.from_xml_string(env_xml)
cloth = mjcf.from_xml_string(cloth)

class MyEntity(composer.ModelWrapperEntity):
    def _build(self, mjcf_model):
        self._mjcf_model = mjcf_model
        self._mjcf_root = mjcf_model

cloth_entity = MyEntity(cloth)
world_entity = MyEntity(world)

world_entity.attach(cloth_entity)

task = composer.NullTask(world_entity)
task.control_timestep = 0.02
env = composer.Environment(task)

viewer.launch(env)
action_spec = env.action_spec()
null_action = np.zeros(action_spec.shape, action_spec.dtype)
num_steps = 1000
for _ in range(num_steps):
    env.step(null_action)