import geopandas as gp
import pandas as pd
import igraph as ig
import numpy as np
import random
from collections import defaultdict

# 读取路网数据并进行预处理
roads = gp.read_file(r"C:\Users\13955\Desktop\1.shp\Export_Output_2.shp")
roads = roads.to_crs(epsg=32756)
roads = roads[roads.geometry.type == 'LineString']
roads['length'] = roads.length
roads = roads.to_crs(epsg=4326)

# 提取起始点和终点坐标
roads['Start_pos'] = roads['geometry'].apply(lambda x: x.coords[0])
roads['End_pos'] = roads['geometry'].apply(lambda x: x.coords[-1])

# 提取节点和边信息
s_points = pd.concat([roads['Start_pos'], roads['End_pos']], ignore_index=True).drop_duplicates().reset_index(drop=True)
df_points = pd.DataFrame(s_points, columns=['Start_pos'])
df_points['FNODE_'] = df_points.index
roads = pd.merge(roads, df_points, on='Start_pos', how='inner')
df_points = pd.DataFrame(s_points, columns=['End_pos'])
df_points['TNODE_'] = df_points.index
roads = pd.merge(roads, df_points, on='End_pos', how='inner')

# 构建图结构
G = ig.Graph()
G.add_vertices(len(s_points))
G.add_edges(roads[['FNODE_', 'TNODE_']].values)
df_points.columns = ['pos', 'osmid']
df_points[['x', 'y']] = df_points['pos'].apply(pd.Series)
df_node_xy = df_points.drop('pos', axis=1)
G.vs["id"] = df_node_xy['osmid'].values
G.vs["x"] = df_node_xy['x'].values
G.vs["y"] = df_node_xy['y'].values
G.es["length"] = roads['length'].values

# 定义Q-Learning训练函数
def train_q_learning(roads, s_points, num_epochs=1000, max_path_search=100, learning_rate=0.1, discount_factor=0.95):
    states = list(range(len(s_points)))
    actions = lambda state: [a for a in states if a != state]
    q_table = defaultdict(lambda: np.zeros(len(states)))

    def reward_function(start, end):
        valid_path = roads[(roads['FNODE_'] == start) & (roads['TNODE_'] == end)]
        if valid_path.empty:
            return -1  # 修改奖励函数，如果路径不存在，返回负奖励
        else:
            path_length = valid_path.iloc[0]['length']
            return 1 / (path_length + 1e-6)

    for epoch in range(num_epochs):
        start_state = random.choice(states)
        end_state = random.choice(states)
        while start_state == end_state:
            end_state = random.choice(states)
        current_state = start_state
        done = False
        path = [current_state]
        path_search_count = 0

        while not done and path_search_count < max_path_search:
            if random.random() < 0.1:
                next_action = random.choice(actions(current_state))
            else:
                next_action = max(actions(current_state), key=lambda a: q_table[current_state][a])

            next_state = next_action
            path.append(next_state)

            if next_state == end_state:
                done = True

            reward = reward_function(current_state, next_state)
            q_predict = q_table[current_state][next_action]
            if done:
                q_target = reward
            else:
                q_target = reward + discount_factor * np.max(q_table[next_state])
            q_table[current_state][next_action] += learning_rate * (q_target - q_predict)

            current_state = next_state
            path_search_count += 1

        print(f"Epoch {epoch + 1}/{num_epochs}, Path: {path}, Reward: {reward}")

    return q_table

# 训练 Q-Learning 模型
q_table = train_q_learning(roads, s_points, num_epochs=1000, learning_rate=0.1, discount_factor=0.95)

# 使用训练好的模型进行路径规划
# 使用训练好的模型进行路径规划
# 使用训练好的模型进行路径规划
def path_planning(start, end, q_table, states, actions, epsilon=0.1):
    current_state = start
    path = [current_state]
    while current_state != end:
        if random.random() < epsilon:
            next_action = random.choice(actions(current_state))
        else:
            next_action = max(actions(current_state), key=lambda a: q_table[current_state][a])
        current_state = next_action
        path.append(current_state)
    return path

# 示例：路径规划
start_node = 0
end_node = len(s_points) - 1
planned_path = path_planning(start_node, end_node, q_table, list(range(len(s_points))), actions, epsilon=0.1)
if planned_path:
    print("Planned Path:", planned_path)
else:
    print("Path planning failed.")


