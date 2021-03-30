from collections import OrderedDict
import numpy as np
import re

from robosuite.utils.transform_utils import convert_quat
from robosuite.utils.mjcf_utils import CustomMaterial

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.grippers import GripperModel

from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject, BottleObject, CanObject
from robosuite.models.objects.xml_objects import ClothObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.observables import Observable, sensor
#cloth size:0.24*0.24

class LiftCloth(SingleArmEnv):
    """
    This class corresponds to the lifting task for a single robot arm.

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!

        env_configuration (str): Specifies how to position the robots within the environment (default is "default").
            For most single arm environments, this argument has no impact on the robot setup.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        table_full_size (3-tuple): x, y, and z dimensions of the table.

        table_friction (3-tuple): the three mujoco friction parameters for
            the table.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (cloth) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        placement_initializer (ObjectPositionSampler): if provided, will
            be used to place objects on every reset, else a UniformRandomSampler
            is used by default.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

    Raises:
        AssertionError: [Invalid number of robots specified]
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1., 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
    ):
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
        )

    def reward(self, action=None):
        """
        Reward function for the task.

        Sparse un-normalized reward:

            - a discrete reward of 2.25 is provided if the cloth is lifted

        Un-normalized summed components if using reward shaping:

            - Reaching: in [0, 1], to encourage the arm to reach the cloth
            - Grasping: in {0, 0.25}, non-zero if arm is grasping the cloth
            - Lifting: in {0, 1}, non-zero if arm has lifted the cloth

        The sparse reward only consists of the lifting component.

        Note that the final reward is normalized and scaled by
        reward_scale / 2.25 as well so that the max score is equal to reward_scale

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        # print("reward")
        reward = 0.
        # print(self.sim.data.contact[: self.sim.data.ncon])
        # print(dir(self.sim.data))
        # print(self.sim.data.xfrc_applied['B0_0'])
        # sparse completion reward
        cloth_element_pos = self._get_element_pos()
        if self._check_success(cloth_element_pos):
            print("the real success!!!!!!!!!!!!!!!!!!!!!!!!")
            reward = 2.25
        # use a shaping reward
        elif self.reward_shaping:
            # reaching reward
            # cloth_pos = self.sim.data.body_xpos[self.cloth_body_id]
            gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
            gripper_site_pos= np.repeat(gripper_site_pos.reshape(1,3), cloth_element_pos.shape[0], axis=0)
            dist = np.min(np.sqrt(np.sum((gripper_site_pos - cloth_element_pos)**2, axis=1)))
            reaching_reward = 0.4*(1 - np.tanh(10.0 * dist))
            # print("reaching reward: ", reaching_reward)
            reward += reaching_reward
            if not np.all(action==0):
                cloth_pos_change = np.mean(np.sqrt(np.sum((self.init_pos[:,:2] - cloth_element_pos[:,:2])**2, axis=1)))
                if cloth_pos_change > 0.3:
                    print("too much movement")
                    reward -= 0.1
            # grasping reward
            if self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cloth):
                grasp_reward = 0.25
                grasped_geoms = self.get_contacts(self.robots[0].gripper)
                grasped_geom_ids = [self.sim.model.geom_name2id(name) for name in grasped_geoms]
                grasped_geom_z_pos = [self.sim.data.geom_xpos[id][2] for id in grasped_geom_ids]
                table_height = self.model.mujoco_arena.table_offset[2]
                average_height = np.mean(grasped_geom_z_pos)
                grasped_height = average_height - table_height
                rising_reward = 0.5*np.tanh(grasped_height*10)
                if grasped_height > 0.05:
                    print("nearly successful!")
                    grasp_reward += 0.3
                    rising_reward = np.tanh(5.0 * grasped_height)
                grasp_reward += rising_reward
                reward += grasp_reward
                # print("grasp reward: ", grasp_reward)


        # Scale reward if requested
        if self.reward_scale is not None:
            reward *= self.reward_scale / 2.25

        return reward

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()
        # print("load")

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # initialize objects of interest
        tex_attrib = {
            "type": "cloth",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        self.cloth = ClothObject("cloth")

        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.cloth)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.cloth,
                x_range=[0.10, 0.14],
                y_range=[-0.03, 0.03],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots], 
            mujoco_objects=self.cloth,
        )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        # print("setup_references")
        super()._setup_references()
        # Additional object references from this env
        self.cloth_body_id = self.sim.model.body_name2id(self.cloth.root_body)

        # clear previous xfrc_force

    def _initialize_cloth_pos(self):
        """
        Initialize the pose of cloth object
        """
        
        self._get_element_ids()
        self.init_pos = self._get_element_pos()
        cloth_size = [[0, self.cloth._count[0]-1], [0, self.cloth._count[1]-1]]
        corner_names = [self.cloth.naming_prefix + f'B{i}_{j}' for i in cloth_size[0]
                                                          for j in cloth_size[1]]
        self.corner_id = [self.sim.model.body_name2id(corner_names[i]) for i in range(4)]
        mid1_id = self.sim.model.body_name2id(self.cloth.naming_prefix + 'B3_4')
        mid2_id = self.sim.model.body_name2id(self.cloth.naming_prefix + 'B4_4')

        # Hold the cloth in position
        self.sim.data.xfrc_applied[mid1_id, :3] = np.random.uniform(-2,0,size=3)
        self.sim.data.xfrc_applied[mid2_id, :3] = np.random.uniform(-2,0,size=3)

        # Force the corners move randomly
        self.sim.data.xfrc_applied[self.corner_id, :3] = np.random.uniform(-1.5,1.5,size=(len(self.corner_id),3))
        _, _, _, _ = self.step(np.zeros(self.robots[0].action_dim))
        self.sim.data.xfrc_applied[:, :3] = np.zeros((3,))
        self.init_pos = self._get_element_pos()

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        # print("observables")
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            # Get robot prefix and define observables modality
            pf = self.robots[0].robot_model.naming_prefix
            modality = "object"
            
            # cloth-related observables
            @sensor(modality=modality)
            def corner_pos(obs_cache):
                corner_positions = [np.array(self.sim.data.body_xpos[id]) for id in self.corner_id]
                return np.concatenate(corner_positions)

            @sensor(modality=modality)
            def cloth_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.cloth_body_id])

            # @sensor(modality=modality)
            # def cloth_quat(obs_cache):
            #     return convert_quat(np.array(self.sim.data.body_xquat[self.cloth_body_id]), to="xyzw")

            # @sensor(modality=modality)
            # def gripper_to_cloth_pos(obs_cache):
            #     return obs_cache[f"{pf}eef_pos"] - obs_cache["cloth_pos"] if \
            #         f"{pf}eef_pos" in obs_cache and "cloth_pos" in obs_cache else np.zeros(3)

            sensors = [cloth_pos, corner_pos]
            names = [s.__name__ for s in sensors]

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )
        return observables

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        # print("rest_internal")
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))
        self._initialize_cloth_pos()
        # print("reward", np.array(self.sim.data.body_xpos[self.cloth_body_id]))

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the cloth.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)
        # print("visual")
        # Color the gripper visualization site according to its distance to the cloth
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.cloth)

    def _get_element_ids(self):
        """
        Get the ids of cloth elements
        """
        pf = self.cloth.naming_prefix
        self.element_ids = [[self.sim.model.geom_name2id(f'{pf}G{i}_{j}')
                        for j in range(9)]
                        for i in range(9)]
    
    def _get_element_pos(self):
        """
        Get the global positions of cloth elements
        """
        element_pos = np.array([[self.sim.data.geom_xpos[self.element_ids[i][j]]
                                 for j in range(9)]
                                 for i in range(9)], dtype='float32') 
        return element_pos.reshape(-1,3)       

    def _check_success(self, object_pose):
        """
        Check if cloth has been lifted.

        Returns:
            bool: True if cloth has been lifted
        """
        # cloth_height = self.sim.data.body_xpos[self.cloth_body_id][2]
        cloth_height = np.mean(object_pose[:,2])
        table_height = self.model.mujoco_arena.table_offset[2]

        # cloth is higher than the table top above a margin
        return cloth_height > table_height + 0.1#0.8
    