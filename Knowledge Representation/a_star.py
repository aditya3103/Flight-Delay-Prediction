import heapq
from weather import get_weather
import EdgeCalculation
import FuzzyLogic
from collections import deque
import heuristic


def a_star_search(start, goal, cities, airports):
    visited = set()
    costs = EdgeCalculation.build_search_tree()
    ignored_cost = costs[start, goal]
    costs[start, goal] = 0
    costs[goal, start] = 0
    pq = [(0, start, [start], 0)]
    while pq:
        (f_score, node, path, delay) = heapq.heappop(pq)
        visited.add(node)
        if node == goal:
            print("Sequence of nodes traveled:", "->".join(path))
            print("Delay:", delay)
            return delay-ignored_cost
        for neighbor in cities:
            if neighbor not in visited and neighbor != node and costs[node, neighbor] != 0:
                g_score = costs[node, neighbor]
                h_score = heuristic.haversine_distance(airports[neighbor]['latitude'], airports[neighbor]['longitude'], airports[goal]['latitude'], airports[goal]['longitude'])
                multiplier = FuzzyLogic.fuzzy(get_weather(node))
                f_score = g_score * multiplier + h_score
                new_delay = g_score * multiplier
                heapq.heappush(pq, (f_score, neighbor, path + [neighbor], (delay+new_delay)))
    return None
