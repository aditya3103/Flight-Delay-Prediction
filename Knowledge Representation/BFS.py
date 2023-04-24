import requests
from weather import get_weather
import EdgeCalculation
import FuzzyLogic
from collections import deque



def bfs_search(start, goal, airports):
    visited = set()
    costs = EdgeCalculation.build_search_tree()
    ignored_cost = costs[start, goal]
    costs[start, goal]=0
    costs[goal, start]=0
    queue = deque([(start, [start], 0)])
    while queue:
        (node, path, actual_time) = queue.popleft()
        visited.add(node)
        if node == goal:   
            print("Sequence of nodes traveled:", "->".join(path))
            print("Difference between actual_time and average_time:", actual_time - ignored_cost)
            return actual_time - ignored_cost
        for neighbor in airports:
            if neighbor not in visited and neighbor != node and costs[node, neighbor]!=0:
                cost = costs[node, neighbor]
                mult = FuzzyLogic.fuzzy(get_weather(node))
                queue.append((neighbor, path + [neighbor], actual_time + mult*cost))









