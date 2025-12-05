import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

import numpy as np
import pygame
import gymnasium as gym
from gymnasium.spaces import Box, Dict
from flappy_bird_env import FlappyBirdEnv


def normalize(value, min_val, max_val):
    clipped_value = np.clip(value, min_val, max_val)
    normalized = (clipped_value - min_val) / (max_val - min_val) * 2 - 1
    return np.clip(normalized, -1.0, 1.0)


class FlappyBirdEnvWithContinuousObs(FlappyBirdEnv):

    def __init__(self, render_mode=None):
        super().__init__(render_mode=render_mode)

        self.observation_space = Dict({
            "bird_height": Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
            "x_to_pipe": Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
            "y_to_gap_center": Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        })
        
        self._has_printed_debug = False

        # self._screen = None 

    @property
    def observation(self):
        normed_bird_height = normalize(
            self._bird.y,
            0, 685  # hardcoded. or 730 - self._bird.image.get_height(),
        )
        normed_x_to_pipe = normalize(
            self._pipes[0].x + self._pipes[0].pipe_bottom.get_width() - self._bird.x,
            0, self._base.width - self._bird.x,
        )
        normed_y_to_gap_center = normalize(
            self._bird.y - (self._pipes[0].height + self._pipes[0].gap / 2),
            -550, 580 - self._bird.image.get_height(),
        )
        return {
            "bird_height": np.array([normed_bird_height], dtype=np.float32),
            "x_to_pipe": np.array([normed_x_to_pipe], dtype=np.float32),
            "y_to_gap_center": np.array([normed_y_to_gap_center], dtype=np.float32),
        }
    
    @property
    def reward(self):
        if any([not pipe.passed and pipe.x < self._bird.x for pipe in self._pipes]):
            return 1.0 
        elif not self.terminated:
            return 0.01 
        else:
            return -1.0 
        
    # def render(self):
    #     if self.render_mode == "rgb_array":
    #         if self._screen is None:
    #             pygame.init()
    #             # (288, 512) Flappy Bird
    #             self._screen = pygame.display.set_mode((288, 512))
            
    #         try:
    #             super().render()
    #         except Exception:
    #             pass
            
    #         return np.transpose(pygame.surfarray.array3d(self._screen), axes=(1, 0, 2))
    #     else:
    #         return super().render()