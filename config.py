# --- Config SUMO environment ---
USE_GUI = False 
SUMOCFG_FILE = 'E:\LET ME COOK\REL301m\dqn_traffic_project\sumo_files\cross.sumocfg'
TRAFFIC_LIGHT_ID = 'J1'
INCOMING_LANES = [
    'E1_0', 'E1_1', 'E1_2',
    'N1_0', 'N1_1', 'N1_2',
    'W1_0', 'W1_1', 'W1_2',
    'S1_0', 'S1_1', 'S1_2'
]

# --- Config agent DQN ---
STATE_SIZE = len(INCOMING_LANES) 
ACTION_SIZE = 2 

MEMORY_SIZE = 7000
GAMMA = 0.95        
LEARNING_RATE = 0.0005

# --- Config strategy EPSILON-GREEDY ---
EPSILON_INIT = 1.0   
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.999945

# --- Config training ---
EPISODES = 150
BATCH_SIZE = 32

# --- Config save model ---
MODEL_SAVE_PATH = 'E:\LET ME COOK\REL301m\dqn_traffic_project\models\dqn_traffic_model_3lane.h5'

#1.0: hoàn toàn khám phá.