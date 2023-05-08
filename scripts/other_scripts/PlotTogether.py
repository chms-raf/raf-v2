import json
import matplotlib.pyplot as plt

experiment_folder = '/home/labuser/ros_ws/src/raf/arm_camera_dataset2/models/model_food/Huimings_Model'

# def load_json_arr(json_path):
#     lines = []
#     with open(json_path, 'r') as f:
#         for line in f:
#             lines.append(json.loads(line))
#     return lines

# experiment_metrics = load_json_arr(experiment_folder + '/metrics.json')

# plt.plot(
#     [x['iteration'] for x in experiment_metrics if 'total_loss' in x], 
#     [x['total_loss'] for x in experiment_metrics if 'total_loss' in x])
# plt.plot(
#     [x['iteration'] for x in experiment_metrics if 'validation_loss' in x], 
#     [x['validation_loss'] for x in experiment_metrics if 'validation_loss' in x])
# plt.legend(['total_loss', 'validation_loss'], loc='upper left')
# plt.show()

def load_json_arr(json_path):
    lines = []
    with open(json_path, 'r') as f:
        for line in f:
            lines.append(json.loads(line))
    return lines

experiment_metrics = load_json_arr(experiment_folder + '/metrics.json')

fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Loss')

ax1.plot(
    [x['iteration'] for x in experiment_metrics if 'total_loss' in x], 
    [x['total_loss'] for x in experiment_metrics if 'total_loss' in x], color="black", label="Total Loss")
ax1.plot(
    [x['iteration'] for x in experiment_metrics if 'validation_loss' in x], 
    [x['validation_loss'] for x in experiment_metrics if 'validation_loss' in x], color="dimgray", label="Val Loss")
    
ax1.tick_params(axis='y')
plt.legend(loc='upper left')

ax2 = ax1.twinx()

color = 'tab:orange'
ax2.set_ylabel('AP')
ax2.plot(
    [x['iteration'] for x in experiment_metrics if 'validation_loss' in x], 
    [x['bbox/AP'] for x in experiment_metrics if 'bbox/AP' in x], color=color, label="bbox AP")
ax2.plot(
    [x['iteration'] for x in experiment_metrics if 'validation_loss' in x], 
    [x['segm/AP'] for x in experiment_metrics if 'segm/AP' in x], color="blue", label="segm AP")
ax2.tick_params(axis='y')

plt.legend(loc='upper right')
plt.show()