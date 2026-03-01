import heapq
import math
import pygame

# Constants needed for logic boundary checks
ROWS = 40
NODE_WALL = (70, 80, 100)

def heuristic(p1, p2, h_type):
    x1, y1 = p1
    x2, y2 = p2
    if h_type == "Manhattan": return abs(x1 - x2) + abs(y1 - y2)
    return math.hypot(x1 - x2, y1 - y2)

# ==========================================
# ALGORITHM 1: A* SEARCH
# ==========================================
def run_astar(grid, start, end, h_type, stats, draw_callback=None):
    open_set, came, count = [], {}, 0
    heapq.heappush(open_set, (0, count, start))
    
    g = {n: float("inf") for row in grid for n in row}
    g[start] = 0
    f = {n: float("inf") for row in grid for n in row}
    f[start] = heuristic(start.get_pos(), end.get_pos(), h_type)
    
    visited_nodes = 0
    while open_set:
        if draw_callback:
            for e in pygame.event.get():
                if e.type == pygame.QUIT: pygame.quit(); quit()
        
        cur = heapq.heappop(open_set)[2]

        if cur == end:
            path = []
            while cur in came:
                cur = came[cur]
                if cur != start: cur.make_path()
                path.append(cur)
            return True, path, visited_nodes, g[end] 

        for r_off, c_off in [(1,0), (-1,0), (0,1), (0,-1)]:
            r, c = cur.row + r_off, cur.col + c_off
            if 0 <= r < ROWS and 0 <= c < ROWS:
                nb = grid[r][c]
                if nb.color != NODE_WALL:
                    temp_g = g[cur] + nb.weight
                    
                    if temp_g < g[nb]:
                        came[nb] = cur
                        g[nb] = temp_g
                        h_val = heuristic(nb.get_pos(), end.get_pos(), h_type)
                        f[nb] = temp_g + h_val  
                        count += 1
                        heapq.heappush(open_set, (f[nb], count, nb))
                        if nb != end and nb != start: nb.make_open()
        
        if cur != start: 
            cur.make_closed()
            visited_nodes += 1
        
        if draw_callback:
            draw_callback()
            
    return False, [], visited_nodes, "N/A"


# ==========================================
# ALGORITHM 2: GREEDY BEST-FIRST SEARCH
# ==========================================
def run_gbfs(grid, start, end, h_type, stats, draw_callback=None):
    open_set, came, count = [], {}, 0
    closed_set = set() 
    heapq.heappush(open_set, (0, count, start))
    
    g = {n: float("inf") for row in grid for n in row}
    g[start] = 0 
    
    visited_nodes = 0
    while open_set:
        if draw_callback:
            for e in pygame.event.get():
                if e.type == pygame.QUIT: pygame.quit(); quit()
        
        cur = heapq.heappop(open_set)[2]
        
        if cur in closed_set:
            continue
        closed_set.add(cur)

        if cur == end:
            path = []
            while cur in came:
                cur = came[cur]
                if cur != start: cur.make_path()
                path.append(cur)
            return True, path, visited_nodes, g[end] 

        for r_off, c_off in [(1,0), (-1,0), (0,1), (0,-1)]:
            r, c = cur.row + r_off, cur.col + c_off
            if 0 <= r < ROWS and 0 <= c < ROWS:
                nb = grid[r][c]
                if nb.color != NODE_WALL and nb not in closed_set:
                    
                    temp_g = g[cur] + nb.weight
                    if temp_g < g[nb]:
                        came[nb] = cur
                        g[nb] = temp_g
                        
                    h_val = heuristic(nb.get_pos(), end.get_pos(), h_type)
                    count += 1
                    
                    heapq.heappush(open_set, (h_val, count, nb))
                    if nb != end and nb != start: nb.make_open()
        
        if cur != start: 
            cur.make_closed()
            visited_nodes += 1
        
        if draw_callback:
            draw_callback()
            
    return False, [], visited_nodes, "N/A"