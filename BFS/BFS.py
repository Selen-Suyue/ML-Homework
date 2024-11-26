import collections
import random
import networkx as nx
import matplotlib.pyplot as plt

def generate_random_directed_graph(max_nodes, edge_probability):
  """生成随机有向图"""
  num_nodes = random.randint(2, max_nodes) 
  graph = {}
  nodes = [chr(ord('A') + i) for i in range(num_nodes)]

  for i in range(num_nodes):
    graph[nodes[i]] = []
    for j in range(num_nodes):
      if i != j and random.random() < edge_probability:
        graph[nodes[i]].append(nodes[j])

  return graph

def bfs(graph, start_node):
  """执行宽度优先搜索"""
  visited = set()
  queue = collections.deque([start_node])
  visited_order = []

  while queue:
    node = queue.popleft()
    if node not in visited:
      visited.add(node)
      visited_order.append(node)
      neighbors = graph.get(node, [])
      for neighbor in neighbors:
        if neighbor not in visited:
          queue.append(neighbor)

  return visited_order

def write_to_file(filename, example, result):
  """将样例和结果写入txt文件"""
  with open(filename, 'a', encoding='utf-8') as f: 
    f.write(f"样例: {example}\n")
    f.write(f"结果: {result}\n\n")

def visualize_graph(graph, visited_order):
  """可视化有向图"""
  G = nx.DiGraph(graph)
  pos = nx.spring_layout(G) 
  
  node_colors = ['lightblue' if node in visited_order else 'lightgray' for node in G.nodes()]
  edge_colors = ['blue' if u in visited_order and v in visited_order else 'gray' for u, v in G.edges()]

  nx.draw(G, pos, with_labels=True, node_color=node_colors, edge_color=edge_colors, arrows=True)  
  plt.show()


max_nodes = 8
edge_probability = 0.3

graph = generate_random_directed_graph(max_nodes, edge_probability)
start_node = random.choice(list(graph.keys()))
result = bfs(graph, start_node)

example = {'graph': graph, 'start_node': start_node}
write_to_file("bfs_results.txt", example, result)

print(f"生成的图: {graph}")
print(f"起始节点: {start_node}")
print(f"访问节点的顺序: {result}")

visualize_graph(graph, result) 