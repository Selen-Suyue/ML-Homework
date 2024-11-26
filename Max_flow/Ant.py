import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
import argparse
from termcolor import cprint

class AntColony:
    def __init__(self, graph, num_ants, alpha, beta, evaporation_rate, iterations):
        self.graph = graph
        self.num_ants = num_ants
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.iterations = iterations
        self.pheromone_matrix = np.ones(graph.shape)

    def run(self):
        sum_flow = 0
        best_path = None
        for _ in range(self.iterations):
            self.pheromone_matrix = np.ones(self.graph.shape)
            max_flow = 0
            for _ in range(self.num_ants):
                path, flow = self.find_path()
                if path == "No path":
                    continue
                if flow > max_flow:
                    max_flow = flow
                    best_path = path
                self.update_pheromone(path, flow)
            if max_flow == 0:
                break
            self.reduce_flow(best_path, max_flow)
            cprint(f"max_path_flow of current iteration is: {max_flow}", "light_magenta")
            sum_flow += max_flow
        return sum_flow

    def reduce_flow(self, path, flow):
        for i in range(len(path) - 1):
            self.graph[path[i], path[i + 1]] = max(0, self.graph[path[i], path[i + 1]] - flow)

    def find_path(self):
        start_node = 0
        end_node = len(self.graph) - 1
        path = []
        current_node = start_node
        flow = float('inf')
        while current_node != end_node:
            path.append(current_node)
            next_node = self.choose_next_node(current_node)
            if next_node == "No node":
                for i in range(len(path)):
                    self.pheromone_matrix[current_node, path[i]] = 0
                    self.pheromone_matrix[path[i], current_node] = 0
                return "No path", "No flow"
            flow = min(flow, self.graph[current_node, next_node])
            current_node = next_node
        path.append(end_node)
        return path, flow

    def choose_next_node(self, current_node):
        probabilities = []
        for next_node in range(len(self.graph[current_node])):
            if self.graph[current_node, next_node] > 0:
                pheromone = self.pheromone_matrix[current_node, next_node]
                heuristic = self.graph[current_node, next_node]
                probabilities.append(pheromone**self.alpha * heuristic**self.beta)
            else:
                probabilities.append(0)
        total_probability = sum(probabilities)
        if total_probability == 0:
            return "No node"
        probabilities = [p / total_probability for p in probabilities]
        return random.choices(range(len(self.graph[current_node])), weights=probabilities)[0]

    def update_pheromone(self, path, flow):
        for i in range(len(path) - 1):
            self.pheromone_matrix[path[i], path[i + 1]] += flow
        self.pheromone_matrix *= (1 - self.evaporation_rate)


def ford_fulkerson(graph):
    n = len(graph)
    max_flow = 0
    while True:
        path = bfs(graph, 0, n - 1)
        if not path:
            break
        flow = min(graph[u][v] for u, v in zip(path[:-1], path[1:]))
        for u, v in zip(path[:-1], path[1:]):
            graph[u][v] -= flow
            graph[v][u] += flow
        max_flow += flow
    return max_flow

def bfs(graph, source, sink):
    queue = [source]
    visited = [False] * len(graph)
    visited[source] = True
    parent = [-1] * len(graph)
    while queue:
        u = queue.pop(0)
        for v in range(len(graph[u])):
            if graph[u][v] > 0 and not visited[v]:
                visited[v] = True
                parent[v] = u
                queue.append(v)
                if v == sink:
                    path = []
                    while v != -1:
                        path.append(v)
                        v = parent[v]
                    return path[::-1]
    return None


def run_experiment(num_nodes, num_trials, output_file,vis):
    with open(output_file, 'a') as f:
        f.write("\n\n\n")
        f.write(f"Results for {num_nodes} nodes\n")
        f.write("=" * 50 + "\n")
        f.write(f"{'Trial':<10}{'Ford-Fulkerson Time (s)':<25}{'Ford-Fulkerson Flow':<25}{'Ant Colony Time (s)':<25}{'Ant Colony Flow':<25}\n")
        f.write("=" * 50 + "\n")

        for trial in range(1, num_trials + 1):
            graph = np.random.randint(1, 10, size=(num_nodes, num_nodes))
            graph = np.triu(graph, k=1)

            # Ford-Fulkerson
            start_time = time.time()
            true_max_flow = ford_fulkerson(graph.copy())
            ford_fulkerson_time = time.time() - start_time

            # ACO
            alpha = 2
            beta = 3
            evaporation_rate = 0.3
            num_ants = num_nodes ** 4
            iterations = num_nodes ** 2
            ant_colony = AntColony(graph.copy(), num_ants, alpha, beta, evaporation_rate, iterations)
            start_time = time.time()
            max_flow_ant = ant_colony.run()
            ant_colony_time = time.time() - start_time

            cprint(f"True Maximum Flow:{true_max_flow}", "red")
            cprint(f"Ant Colony Maximum Flow:{max_flow_ant}", "green")
            f.write(f"{trial:<10}{ford_fulkerson_time:<25.4f}{true_max_flow:<25}{ant_colony_time:<25.4f}{max_flow_ant:<25}\n")

            # Visualization for every 3 trials
            if vis:
                G = nx.DiGraph()
                for i in range(num_nodes):
                    G.add_node(i)
                for i in range(num_nodes):
                    for j in range(num_nodes):
                        if graph[i, j] > 0:
                            G.add_edge(i, j, weight=graph[i, j])

                pos = nx.spring_layout(G)
                nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightblue', font_size=10)
                edge_labels = nx.get_edge_attributes(G, 'weight')
                nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
                plt.show()


if __name__ == "__main__":
    # Graph and Exp Configs
    parser = argparse.ArgumentParser(description="Run flow algorithms and record results.")
    parser.add_argument("--num_nodes", type=int, default=6, help="Number of nodes in the graph (default is 6).")
    parser.add_argument("--num_trials", type=int, default=10, help="Number of trials to run (default is 10).")
    parser.add_argument("--output_file", type=str, default="results.txt", help="Output file to store the results (default is results.txt).")
    parser.add_argument("--visualize", type=bool, default=False, help="if visualize the graph")

    args = parser.parse_args()

    run_experiment(args.num_nodes, args.num_trials, args.output_file, args.visualize)

    cprint(f"Experiment completed. Results saved in {args.output_file}", "yellow")
