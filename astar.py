import heapq


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(start, goal, width, height, obstacles):
    if start == goal:
        return []

    open_set = []
    heapq.heappush(open_set, (manhattan(start, goal), 0, start))

    came_from = {}
    g_score = {start: 0}
    visited = set()

    while open_set:
        _, g, current = heapq.heappop(open_set)

        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            path = []
            while current != start:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path

        x, y = current
        for nx, ny in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]:
            nxt = (nx, ny)
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            if nxt in obstacles:
                continue

            new_g = g + 1
            if new_g < g_score.get(nxt, float("inf")):
                came_from[nxt] = current
                g_score[nxt] = new_g
                f = new_g + manhattan(nxt, goal)
                heapq.heappush(open_set, (f, new_g, nxt))

    return None
