import cv2
import numpy as np
import matplotlib.pyplot as plt
from queue import PriorityQueue

# 原始地图处理（保持你的代码不变）
map_img = cv2.imread('./map/vivocity.png')
map_img = cv2.cvtColor(map_img, cv2.COLOR_BGR2RGB)
light_free = (210, 200, 190)
dark_free = (250, 240, 230)
free_space = cv2.inRange(map_img, light_free, dark_free)

# A*算法实现
class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.g = float('inf')
        self.h = 0
        self.parent = None

    def __lt__(self, other):
        return (self.g + self.h) < (other.g + other.h)

def heuristic(a, b):
    # 使用曼哈顿距离作为启发函数
    return abs(a.x - b.x) + abs(a.y - b.y)

def a_star(start, goal, grid):
    rows, cols = grid.shape
    open_set = PriorityQueue()
    
    start_node = Node(*start)
    goal_node = Node(*goal)
    
    start_node.g = 0
    start_node.h = heuristic(start_node, goal_node)
    open_set.put((start_node.g + start_node.h, start_node))
    
    visited = np.zeros((rows, cols), dtype=bool)
    
    while not open_set.empty():
        _, current = open_set.get()
        
        if current.x == goal_node.x and current.y == goal_node.y:
            path = []
            while current:
                path.append((current.x, current.y))
                current = current.parent
            return path[::-1]
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                
                x = current.x + dx
                y = current.y + dy
                
                if 0 <= x < rows and 0 <= y < cols:
                    if grid[x, y] == 255 and not visited[x, y]:  # 255表示可通行区域
                        neighbor = Node(x, y)
                        neighbor.g = current.g + 1
                        neighbor.h = heuristic(neighbor, goal_node)
                        neighbor.parent = current
                        
                        if not visited[x, y]:
                            visited[x, y] = True
                            open_set.put((neighbor.g + neighbor.h, neighbor))
    
    return None  # 没有找到路径

# 转换为可通行网格（0=障碍，255=可通行）
grid = free_space

# 设置起点和终点（需要根据你的地图实际位置调整）
start = (50, 100)  # (y, x) 坐标
goal = (200, 300)  # (y, x) 坐标

# 运行A*算法
path = a_star(start, goal, grid)

# 可视化结果
plt.figure(figsize=(12, 6))

# 原始地图
plt.subplot(1, 3, 1)
plt.imshow(map_img)
plt.title("Original Map")
plt.scatter([start[1], goal[1]], [start[0], goal[0]], c=['green', 'red'], s=50)

# 自由空间地图
plt.subplot(1, 3, 2)
plt.imshow(free_space, cmap='gray')
plt.title("Free Space")
plt.scatter([start[1], goal[1]], [start[0], goal[0]], c=['green', 'red'], s=50)

# 带路径的结果
result_img = map_img.copy()
if path:
    for (y, x) in path:
        cv2.circle(result_img, (x, y), 2, (255, 0, 0), -1)
    # 绘制起点终点
    cv2.circle(result_img, (start[1], start[0]), 5, (0, 255, 0), -1)
    cv2.circle(result_img, (goal[1], goal[0]), 5, (255, 0, 0), -1)
else:
    print("No path found!")

plt.subplot(1, 3, 3)
plt.imshow(result_img)
plt.title("Path Finding Result")

plt.tight_layout()
plt.show()
