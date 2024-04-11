#shapely 2.0 版本建议使用下面注释的1行
import os
os.environ['USE_PYGEOS'] = '0'
import geopandas as gp
import pandas as pd
import shapely
#使用geopandas读取路网文件，路网文件需要先进行前处理，比如进行相交线打断操作推荐使用arcgis
#需要转换成投影坐标计算长度
roads = gp.read_file(r"C:\Users\13955\Desktop\1.shp\Export_Output_2.shp")
roads = roads.to_crs(epsg=32756)
roads = roads[roads.geometry.type == 'LineString']
roads['length'] = roads.length
roads = roads.to_crs(epsg=4326)
#以下的转换代码主要为获取线的两个端点，并重新编码，匹配后可获得图结构的顶点和边情况
# Compute the start- and end-position based on linestring
roads['Start_pos'] = roads.geometry.apply(lambda x: x.coords[0])
roads['End_pos'] = roads.geometry.apply(lambda x: x.coords[-1])
print(roads.head(5))

# Create Series of unique nodes and their associated position
s_points = pd.concat([roads.Start_pos,roads.End_pos], ignore_index=True)
s_points = s_points.drop_duplicates().reset_index(drop=True)
print(s_points)

# Add index of start and end node of linestring to geopandas DataFrame
df_points = pd.DataFrame(s_points, columns=['Start_pos'])
df_points['FNODE_'] = df_points.index
roads = pd.merge(roads, df_points, on='Start_pos', how='inner')
df_points = pd.DataFrame(s_points, columns=['End_pos'])
df_points['TNODE_'] = df_points.index
roads = pd.merge(roads, df_points, on='End_pos', how='inner')
print(roads)

# Bring nodes and their position in form needed for osmnx (give arbitrary osmid (index) despite not osm file)
df_points.columns = ['pos', 'osmid']
df_points[['x', 'y']] = df_points['pos'].apply(pd.Series)
df_node_xy = df_points.drop('pos', axis=1)
print(df_node_xy)

import igraph as ig
G = ig.Graph()
G.add_vertices(len(s_points))
G.add_edges(roads[['FNODE_','TNODE_']].values)

G.vs["id"] = df_node_xy['osmid'].values
G.vs["x"] = df_node_xy['x'].values
G.vs["y"] = df_node_xy['y'].values
G.es["length"] = roads['length'].values

import numpy as np
import random
from collections import defaultdict
# 定义超参数
learning_rate = 0.1
discount_factor = 0.95
num_epochs = 1000
# 定义状态空间（简化处理，假设每个节点都有一个唯一的状态标识）
states = list(range(len(s_points)))
# 定义动作空间（选择下一个节点作为动作）
actions = lambda state: [a for a in states if a != state]  # 不选择当前节点作为下一个节点
# 初始化Q表
q_table = defaultdict(lambda: np.zeros(len(states)))
# 定义奖励函数（这里简单设置为路径长度的倒数）
def reward_function(start, end):
    # 检查起点和终点之间是否存在路径
    valid_path = roads[(roads['FNODE_'] == start) & (roads['TNODE_'] == end)]
    if valid_path.empty:
        print(f"No valid path found between nodes {start} and {end}")
        return 0  # 或者其他处理方式
    else:
        path_length = valid_path.iloc[0]['length']
        return 1 / (path_length + 1e-6)  # 避免除以零

# Q-Learning训练过程
for epoch in range(num_epochs):
    # 随机选择起点和终点
    start_state = random.choice(states)
    end_state = random.choice(states)
    while start_state == end_state:
        end_state = random.choice(states)  # 确保起点和终点不同
    # 初始化当前状态和动作
    current_state = start_state
    done = False
    path = [current_state]
    # 模拟一个完整的路径规划过程
    while not done:
        # 选择动作（使用ε-greedy策略）
        if random.random() < 0.1:  # 探索新动作
            next_action = random.choice(actions(current_state))
        else:  # 利用已有知识选择最优动作
            next_action = max(actions(current_state), key=lambda a: q_table[current_state][a])
            # 转移到下一个状态
        next_state = next_action
        path.append(next_state)
        # 如果到达终点，则设置done为True
        if next_state == end_state:
            done = True

            # 更新Q值
        # 更新Q值
        reward = reward_function(current_state, next_state)
        q_predict = q_table[current_state][next_action]
        if done:
            q_target = reward
        else:
            q_target = reward + discount_factor * np.max(q_table[next_state])  # 使用 np.max() 获取最大值
        q_table[current_state][next_action] += learning_rate * (q_target - q_predict)

        # 更新当前状态
        current_state = next_state

    print(f"Epoch {epoch + 1}/{num_epochs}, Path: {path}, Reward: {reward}")


# 使用训练好的模型进行路径规划
# 使用训练好的模型进行路径规划
def path_planning(start, end):
    current_state = start
    path = [current_state]
    while current_state != end:
        # 选择最优动作（利用Q表）
        # 根据Q表选择最优动作（下一个节点）
        next_action = max(actions(current_state), key=lambda a: q_table[current_state][a])

        # 转移到下一个状态
        current_state = next_action
        path.append(current_state)
    return path
# 示例：使用训练好的模型进行路径规划
start_node = 0  # 假设起点是节点0
end_node = len(s_points) - 1  # 假设终点是最后一个节点
planned_path = path_planning(start_node, end_node)
if planned_path:
    print("Planned Path:", planned_path)
else:
    print("Path planning failed.")
