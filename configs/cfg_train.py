# DATASET
dataset = 'Carla'
data_root = 'C:\\Users\\marvi\\Datasets\\Lane\\CarlaLane'
num_lanes = 4
num_cls = 2

# NETWORK
use_aux = True
griding_num = 200
backbone = '18'
classification = True

# TRAIN
epoch = 50
batch_size = 32
optimizer = 'SGD' # ['SGD','Adam']
learning_rate = 0.1
weight_decay = 1e-4
momentum = 0.9

# SCHEDULER
scheduler = 'multi' # ['multi', 'cos']
steps = [25,38]
gamma  = 0.1
warmup = 'linear'
warmup_iters = 695

# LOSS
sim_loss_w = 0.0
shp_loss_w = 0.0

# FINETUNE or RESUME MODEL PATH
finetune = None
resume = None
save_path = "C:\\Users\\marvi\\carla_lane_detection\\weights"

