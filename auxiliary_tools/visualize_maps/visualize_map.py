"""
Script for generate map visualization for maps which has yaml file in
"auxiliary_tools/visualize_maps/maps" folder and also exist at
"lib/python3.8/site-packages/duckietown_world/data/gd1/maps" folder.

Output folder: "auxiliary_tools/visualize_maps/maps_visualized"

command to run: python auxiliary_tools/visualize_maps/visualize_map.py
"""
from os import listdir
from os.path import isfile, join

import gym_duckietown
from duckietown_world.world_duckietown import load_map
from duckietown_world.svg_drawing.dt_draw_maps.draw_maps import draw_map

#get all filenames
mypath = "auxiliary_tools/visualize_maps/maps"

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

new_list = []
for each in onlyfiles:
    new_list.append(each[:-5])
    

for map in new_list:
    duckietown_map = load_map(map)
    draw_map('auxiliary_tools/visualize_maps/maps_visualized/' + map, duckietown_map)

print("Maps are visualized into the (maps_visualized) folder.")