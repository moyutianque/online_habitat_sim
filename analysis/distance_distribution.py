import json
from turtle import color
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import os.path as osp
import seaborn as sns  # for nicer graphics

split = 'val_unseen'
chunk_resolution = 4 # in meters
result_dir = f"scripts/{split}_mln3d_routes.json"
out_root = './analysis'
with open(result_dir, "r") as f:
    data= {str(v['episode_id']):v for v in json.load(f)}


distance_counter = defaultdict(int) # tot
distance_correct = defaultdict(int) # correct
for ep_id, v in data.items():
    distance = v['action_seq'].count(1) * 1.5
    chunk = int(distance/chunk_resolution)
    distance_counter[chunk] += 1
    if v['is_correct']:
        distance_correct[chunk] += 1
    else:
        distance_correct[chunk] += 0

distance_counter = sorted(distance_counter.items(), key=lambda item: item[0])
distance_counter = np.array(distance_counter).transpose()
distance_correct = sorted(distance_correct.items(), key=lambda item: item[0])
distance_correct = np.array(distance_correct).transpose()
x, y = distance_correct[0], distance_correct[1]/distance_counter[1]

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.bar(x,y,color='b')
ax1.set_xlim([0,30])
ax1.set_xlabel(f"distance chunk (resolution={chunk_resolution}m)")
ax1.set_ylabel('Accuracy', color='b')
ax1.set_ylim([0,1])
ax2.plot(x, distance_counter[1], 'r--')
ax2.set_ylabel('Total number of data', color='r')

plt.title(f"{split.upper()} split Accuracy w.r.t Distance")
plt.savefig(osp.join(out_root, f"distance_acc-{split}.png"))

