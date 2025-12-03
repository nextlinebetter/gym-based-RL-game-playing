import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

import numpy as np
import pygame
import gymnasium as gym
from gymnasium.spaces import Box, Dict
from flappy_bird_env import FlappyBirdEnv

class FlappyBirdEnvWithContinuousObs(FlappyBirdEnv):

    def __init__(self, render_mode=None):
        super().__init__(render_mode=render_mode)

        self.observation_space = Dict({
            "bird_y": Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
            "bird_v": Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
            "pipe_x": Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
            "pipe_y_diff": Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        })
        
        self._has_printed_debug = False

        self._screen = None 

    def _get_bird_velocity(self):
        possible_names = ['vel', 'vel_y', 'velocity', 'v', 'speed', 'playerVelY', 'vals']
        for name in possible_names:
            if hasattr(self._bird, name):
                return getattr(self._bird, name)
        
        if not self._has_printed_debug:
            self._has_printed_debug = True
        return 0.0

    @property
    def observation(self):
        bird_y = self._bird.y
        bird_v = self._get_bird_velocity()
        
        if len(self._pipes) > 0:
            pipe = self._pipes[0]
            pipe_x = pipe.x
            pipe_gap_center = pipe.height + pipe.gap / 2 
        else:
            pipe_x = 288.0
            pipe_gap_center = 256.0

        raw_bird_y = (bird_y / 512.0) * 2 - 1      
        raw_bird_v = bird_v / 10.0                 
        raw_pipe_x = (pipe_x / 288.0) * 2 - 1      
        raw_y_diff = (bird_y - pipe_gap_center) / 512.0 

        norm_bird_y = np.clip(raw_bird_y, -1.0, 1.0)
        norm_bird_v = np.clip(raw_bird_v, -1.0, 1.0)
        norm_pipe_x = np.clip(raw_pipe_x, -1.0, 1.0)
        norm_y_diff = np.clip(raw_y_diff, -1.0, 1.0)

        return {
            "bird_y": np.array([norm_bird_y], dtype=np.float32),
            "bird_v": np.array([norm_bird_v], dtype=np.float32),
            "pipe_x": np.array([norm_pipe_x], dtype=np.float32),
            "pipe_y_diff": np.array([norm_y_diff], dtype=np.float32) 
        }
    
    @property
    def reward(self):
        if any([not pipe.passed and pipe.x < self._bird.x for pipe in self._pipes]):
            return 1.0 
        elif not self.terminated:
            return 0.1 
        else:
            return -1.0 
        
    def render(self):
        if self.render_mode == "rgb_array":
            if self._screen is None:
                pygame.init()
                # (288, 512) Flappy Bird
                self._screen = pygame.display.set_mode((288, 512))
            
            try:
                super().render()
            except Exception:
                pass
            
            return np.transpose(pygame.surfarray.array3d(self._screen), axes=(1, 0, 2))
        else:
            return super().render()