import os
import sys
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import traci
from config import SUMOCFG_FILE, TRAFFIC_LIGHT_ID, INCOMING_LANES

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

class TrafficEnv(gym.Env):
    def __init__(self, use_gui=False):
        super(TrafficEnv, self).__init__()
        self.use_gui = use_gui
        self.sumocfg_file = SUMOCFG_FILE
        self.traffic_light_id = TRAFFIC_LIGHT_ID
        self.incoming_lanes = INCOMING_LANES
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(len(self.incoming_lanes),), dtype=np.float32)

    def _start_simulation(self):
        sumo_binary = 'sumo-gui' if self.use_gui else 'sumo'
        traci.start([sumo_binary, "-c", self.sumocfg_file])

    def reset(self, seed=None, options=None):
        if traci.isLoaded():
            traci.close()
        self._start_simulation()
        for _ in range(5):
            traci.simulationStep()
        observation = self._get_state()
        info = {}
        return observation, info

    def _get_state(self):
        state = [traci.lane.getLastStepHaltingNumber(lane) for lane in self.incoming_lanes]
        return np.array(state, dtype=np.float32)

    def step(self, action):
        current_phase = traci.trafficlight.getPhase(self.traffic_light_id)
        target_phase = action * 2
        if current_phase != target_phase:
            traci.trafficlight.setPhase(self.traffic_light_id, target_phase)
        for _ in range(10):
            traci.simulationStep()
        next_state = self._get_state()
        reward = -sum(np.square(next_state))
        done = traci.simulation.getTime() >= 3600
        truncated = False
        info = {}
        return next_state, reward, done, truncated, info

    def close(self):
        if traci.isLoaded():
            traci.close()