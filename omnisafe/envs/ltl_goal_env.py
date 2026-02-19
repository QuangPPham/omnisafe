from __future__ import annotations

from typing import Any, ClassVar
import random

import numpy as np
import safety_gymnasium
import torch

from omnisafe.envs.core import CMDP, env_register
from omnisafe.typing import DEVICE_CPU, Box, OmnisafeSpace
from omnisafe.common.logger import Logger
from omnisafe.utils.dfa import DFA

mona_dfa_string_goal0 = """
digraph MONA_DFA {
rankdir = LR;
center = true;
size = "7.5,10.5";
edge [fontname = Courier];
node [height = .5, width = .5];
node [shape = doublecircle]; 2;
node [shape = circle]; 1;
init [shape = plaintext, label = ""];
init -> 1;
1 -> 1 [label="~goal"];
1 -> 2 [label="goal"];
2 -> 2 [label="goal"];
2 -> 1 [label="~goal"];
}
"""

@env_register
class LTLGoalEnv(CMDP):
    """Safety Gymnasium Environment.

    Args:
        env_id (str): Environment id.
        num_envs (int, optional): Number of environments. Defaults to 1.
        device (torch.device, optional): Device to store the data. Defaults to
            ``torch.device('cpu')``.

    Keyword Args:
        render_mode (str, optional): The render mode ranges from 'human' to 'rgb_array' and 'rgb_array_list'.
            Defaults to 'rgb_array'.
        camera_name (str, optional): The camera name.
        camera_id (int, optional): The camera id.
        width (int, optional): The width of the rendered image. Defaults to 256.
        height (int, optional): The height of the rendered image. Defaults to 256.

    Attributes:
        need_auto_reset_wrapper (bool): Whether to use auto reset wrapper.
        need_time_limit_wrapper (bool): Whether to use time limit wrapper.
    """

    need_auto_reset_wrapper: bool = False
    need_time_limit_wrapper: bool = False

    _support_envs: ClassVar[list[str]] = [
        'ltl-SafetyPointGoal0-v0',
        'ltl-SafetyPointGoal1-v0',
        'ltl-SafetyPointGoal2-v0',
        'ltl-SafetyCarGoal0-v0',
        'ltl-SafetyCarGoal1-v0',
        'ltl-SafetyCarGoal2-v0',
        'ltl-SafetyAntGoal0-v0',
        'ltl-SafetyAntGoal1-v0',
        'ltl-SafetyAntGoal2-v0',
        'ltl-SafetyDoggoGoal0-v0',
        'ltl-SafetyDoggoGoal1-v0',
        'ltl-SafetyDoggoGoal2-v0',
        'ltl-SafetyRacecarGoal0-v0',
        'ltl-SafetyRacecarGoal1-v0',
        'ltl-SafetyRacecarGoal2-v0',
    ]

    def __init__(self, 
                    env_id:str, num_envs:int=1, device:torch.device=DEVICE_CPU,
                    gamma:float=0.99, gammaB:float=0.99, **kwargs:Any,
        ) -> None:
        """Initialize an instance of :class:`SafetyGymnasiumEnv`."""
        super().__init__(env_id)
        self._num_envs = num_envs
        self._device = torch.device(device)
        self.dfa = DFA(mona_dfa_string_goal0)
        num_q_states = self.dfa.shape[1]
        env_id = env_id[4:]
        self.q_state = self.dfa.q0
        self._gamma = gamma
        self._gammaB = gammaB

        if num_envs > 1:
            raise NotImplementedError('Only support num_envs=1 now.')
        else:
            self.need_time_limit_wrapper = True
            self.need_auto_reset_wrapper = True
            self._env = safety_gymnasium.make(id=env_id, autoreset=False, **kwargs)
            assert isinstance(self._env.action_space, Box), 'Only support Box action space.'
            assert isinstance(self._env.observation_space, Box), 'Only support Box observation space.'
            self._action_space = self._env.action_space
            # observation = on-board sensors + lidar to goals + hazards
            observation_space = self._env.observation_space
            low = observation_space.low
            high = observation_space.high
            dim = observation_space.shape[0]
            # Add observation for dfa state
            if isinstance(low, np.ndarray):
                new_low = np.hstack((low, 0))
                new_high = np.hstack((high, num_q_states))
            else:
                new_low = np.array([low]*dim + [0])
                new_high = np.array([high]*dim + [num_q_states])
            #self._observation_space = Box(new_low, new_high, (dim+1,), dtype=np.float32)
            self._observation_space = observation_space
        self._metadata = self._env.metadata

    def step(
        self, 
        action: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
    ]:
        """Step the environment.

        .. note::
            OmniSafe uses auto reset wrapper to reset the environment when the episode is
            terminated. So the ``obs`` will be the first observation of the next episode. And the
            true ``final_observation`` in ``info`` will be stored in the ``final_observation`` key
            of ``info``.

        Args:
            action (torch.Tensor): Action to take.

        Returns:
            observation: The agent's observation of the current environment.
            reward: The amount of reward returned after previous action.
            cost: The amount of cost returned after previous action.
            terminated: Whether the episode has ended.
            truncated: Whether the episode has been truncated due to a time limit.
            info: Some information logged by the environment.
        """
        # Step
        obs, reward, cost, terminated, truncated, info = self._env.step(action.detach().cpu().numpy())

        # See if agent is within the goal region, if so set goal_reached=True
        # goal_achieved = self._env.task.goal_achieved
        if 'goal_met' in info:
            label = {
                'goal': True
            }
        else:
            label = {
                'goal': False
            }

        # Step dfa
        self.q_state = self.dfa.step(self.q_state, label)[0]

        # Set reward and discount factor based on dfa state
        if self.q_state in self.dfa.acc:
            #reward = 1. - self._gammaB
            gamma = self._gammaB
        else:
            #reward = 0.
            gamma = self._gamma

        # Append dfa state to observation
        #obs = np.hstack((obs, self.q_state))

        # Turn to tensors
        obs, reward, cost, terminated, truncated, gamma = (
            torch.as_tensor(x, dtype=torch.float32, device=self._device)
            for x in (obs, reward, cost, terminated, truncated, gamma))
        
        # Add discount factor to info dictionary
        info['gamma'] = gamma

        if 'final_observation' in info:
            info['final_observation'] = np.array(
                [array if array is not None else np.zeros(obs.shape[-1]) for array in info['final_observation']]
                )
            info['final_observation'] = torch.as_tensor(info['final_observation'], dtype=torch.float32, device=self._device)

        return obs, reward, cost, terminated, truncated, info

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[
        torch.Tensor, 
        dict[str, Any]
        ]:
        """Reset the environment.

        Args:
            seed (int, optional): The random seed. Defaults to None.
            options (dict[str, Any], optional): The options for the environment. Defaults to None.

        Returns:
            observation: Agent's observation of the current environment.
            info: Some information logged by the environment.
        """
        obs, info = self._env.reset(seed=seed, options=options)
        return torch.as_tensor(obs, dtype=torch.float32, device=self._device), info

    @property
    def max_episode_steps(self) -> int:
        """The max steps per episode."""
        return self._env.spec.max_episode_steps

    def set_seed(self, seed: int) -> None:
        """Set the seed for the environment.

        Args:
            seed (int): Seed to set.
        """
        self.reset(seed=seed)

    def render(self) -> Any:
        """Compute the render frames as specified by :attr:`render_mode` during the initialization of the environment.

        Returns:
            The render frames: we recommend to use `np.ndarray`
                which could construct video by moviepy.
        """
        return self._env.render()

    def close(self) -> None:
        """Close the environment."""
        self._env.close()
