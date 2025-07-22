# --- Config SUMO environment ---
USE_GUI = False 
SUMOCFG_FILE = r'E:\LET ME COOK\REL301m\dqn_traffic_project\sumo_files\cross.sumocfg'

# --- Config traffic light ---
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

MEMORY_SIZE = 10000
GAMMA = 0.95        
LEARNING_RATE = 0.0001

# --- Config strategy EPSILON-GREEDY ---
EPSILON_INIT = 1.0 
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.99996

# --- Config early stop ---
EARLY_STOP_PATIENCE = 50

# --- Config training ---
EPISODES = 300
BATCH_SIZE = 512

# --- Config save model ---
MODEL_SAVE_PATH = r'E:\LET ME COOK\REL301m\dqn_traffic_project\models\dqn_traffic_model_3lane.h5'

#1.0: hoàn toàn khám phá.