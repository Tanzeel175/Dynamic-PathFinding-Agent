import pygame
import random
import time
import math
import heapq
import sys

# --- CONFIGURATION & WHITE THEME ---
WIDTH, HEIGHT = 980, 700
GRID_WIDTH = 680
ROWS = 40
CELL_SIZE = GRID_WIDTH // ROWS

BG_MAIN = (255, 255, 255)
GRID_LINE = (220, 220, 230)
NODE_EMPTY = (255, 255, 255)
NODE_WALL_COLOR = (45, 50, 65)
NODE_START = (0, 200, 150)
NODE_GOAL = (255, 50, 100)
NODE_WEIGHT = (200, 100, 255)
NODE_OPEN = (190, 230, 255)
NODE_CLOSED = (225, 235, 245)
NODE_PATH = (0, 150, 255)
AGENT_HEAD = (0, 0, 0)

UI_PANEL = (245, 245, 250)
UI_ACCENT = (0, 180, 200)
TEXT_MAIN = (30, 35, 45)
TEXT_SUB = (100, 110, 120)

pygame.init()
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("PATH-FINDING AGENT - FINAL")

FONT_SM = pygame.font.SysFont("Segoe UI", 12, bold=True) 
FONT_MED = pygame.font.SysFont("Segoe UI", 14, bold=True)
FONT_LG = pygame.font.SysFont("Segoe UI", 26, bold=True)

# --- CLASSES ---
class Node:
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.x = col * CELL_SIZE
        self.y = row * CELL_SIZE
        self.color = NODE_EMPTY
        self.weight = 1

    def get_pos(self): return (self.row, self.col)
    
    def reset(self): 
        self.color = NODE_EMPTY
        self.weight = 1
        
    def make_start(self): self.color = NODE_START
    def make_goal(self): self.color = NODE_GOAL
    def make_barrier(self): self.color = NODE_WALL_COLOR
    def make_weight(self): 
        self.color = NODE_WEIGHT
        self.weight = 5 
    def make_open(self): self.color = NODE_OPEN
    def make_closed(self): self.color = NODE_CLOSED
    def make_path(self): self.color = NODE_PATH
    def make_agent(self): self.color = AGENT_HEAD
    
    def draw(self, win): 
        rect = (self.x + 1, self.y + 1, CELL_SIZE - 2, CELL_SIZE - 2)
        pygame.draw.rect(win, self.color, rect, border_radius=4)
        if self.weight > 1:
            pygame.draw.circle(win, (255, 255, 255, 180), (self.x + CELL_SIZE//2, self.y + CELL_SIZE//2), CELL_SIZE//4)

    def __lt__(self, other):
        return False

class ModernButton:
    def __init__(self, x, y, w, h, text, val):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.val = val
        
    def draw(self, win, active=False):
        bg_color = UI_ACCENT if active else UI_PANEL
        txt_color = (255, 255, 255) if active else TEXT_SUB
        pygame.draw.rect(win, bg_color, self.rect, border_radius=12)
        if not active:
            pygame.draw.rect(win, (210, 215, 225), self.rect, 2, border_radius=12)
        txt_surf = FONT_MED.render(self.text, True, txt_color)
        win.blit(txt_surf, (self.rect.centerx - txt_surf.get_width()//2, self.rect.centery - txt_surf.get_height()//2))

# --- ALGORITHMS ---
def heuristic(p1, p2, h_type):
    x1, y1 = p1
    x2, y2 = p2
    if h_type == "Manhattan": return abs(x1 - x2) + abs(y1 - y2)
    return math.hypot(x1 - x2, y1 - y2)

def run_astar(grid, start, end, h_type, draw_callback):
    count = 0
    open_set = []
    heapq.heappush(open_set, (0, count, start))
    came = {}
    g = {n: float("inf") for row in grid for n in row}
    g[start] = 0
    open_set_hash = {start}
    visited_nodes = 0
    
    while open_set:
        if draw_callback:
            for e in pygame.event.get():
                if e.type == pygame.QUIT: pygame.quit(); sys.exit()
        
        cur = heapq.heappop(open_set)[2]
        open_set_hash.remove(cur)

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
                if nb.color != NODE_WALL_COLOR:
                    temp_g = g[cur] + nb.weight
                    if temp_g < g[nb]:
                        came[nb] = cur
                        g[nb] = temp_g
                        f_score = temp_g + heuristic(nb.get_pos(), end.get_pos(), h_type)
                        if nb not in open_set_hash:
                            count += 1
                            heapq.heappush(open_set, (f_score, count, nb))
                            open_set_hash.add(nb)
                            if nb != end and nb != start: nb.make_open()
        
        if cur != start: 
            cur.make_closed()
            visited_nodes += 1
        
        if draw_callback: draw_callback()
            
    return False, [], visited_nodes, "N/A"

def run_gbfs(grid, start, end, h_type, draw_callback):
    count = 0
    open_set = []
    heapq.heappush(open_set, (0, count, start))
    came = {}
    closed_set = set() 
    open_set_hash = {start}
    g = {n: float("inf") for row in grid for n in row}
    g[start] = 0 
    visited_nodes = 0
    
    while open_set:
        if draw_callback:
            for e in pygame.event.get():
                if e.type == pygame.QUIT: pygame.quit(); sys.exit()
        
        cur = heapq.heappop(open_set)[2]
        open_set_hash.remove(cur)
        
        if cur in closed_set: continue
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
                if nb.color != NODE_WALL_COLOR and nb not in closed_set:
                    temp_g = g[cur] + nb.weight
                    if temp_g < g[nb]:
                        came[nb] = cur
                        g[nb] = temp_g
                        
                    if nb not in open_set_hash:
                        h_val = heuristic(nb.get_pos(), end.get_pos(), h_type)
                        count += 1
                        heapq.heappush(open_set, (h_val, count, nb))
                        open_set_hash.add(nb)
                        if nb != end and nb != start: nb.make_open()
        
        if cur != start: 
            cur.make_closed()
            visited_nodes += 1
        
        if draw_callback: draw_callback()
            
    return False, [], visited_nodes, "N/A"

# --- RENDER ENGINE ---
def draw_grid(win, grid):
    win.fill(BG_MAIN)
    for row in grid:
        for node in row: node.draw(win)
    for i in range(ROWS + 1):
        pygame.draw.line(win, GRID_LINE, (0, i * CELL_SIZE), (GRID_WIDTH, i * CELL_SIZE))
        pygame.draw.line(win, GRID_LINE, (i * CELL_SIZE, 0), (i * CELL_SIZE, GRID_WIDTH))

def draw_sidebar(win, ui, stats):
    px = GRID_WIDTH
    pygame.draw.rect(win, UI_PANEL, (px, 0, WIDTH - px, HEIGHT))
    pygame.draw.line(win, GRID_LINE, (px, 0), (px, HEIGHT), 2)
    win.blit(FONT_LG.render("PATH-FINDING", True, UI_ACCENT), (px + 25, 25))
    win.blit(FONT_LG.render("AGENT", True, TEXT_MAIN), (px + 25, 55))

    for key, btn in ui.items():
        if isinstance(btn, ModernButton):
            active = False
            if key == 'btn_astar': active = ui['algo'] == 'A*'
            elif key == 'btn_gbfs': active = ui['algo'] == 'GBFS'
            elif key == 'btn_man': active = ui['heuristic'] == 'Manhattan'
            elif key == 'btn_euc': active = ui['heuristic'] == 'Euclidean'
            elif key == 'btn_static': active = ui['mode'] == 'Static'
            elif key == 'btn_dynamic': active = ui['mode'] == 'Dynamic'
            elif key == 'btn_draw_wall': active = ui['draw_mode'] == 'Wall'
            elif key == 'btn_draw_weight': active = ui['draw_mode'] == 'Weight'
            elif key == 'btn_set_start': active = ui['draw_mode'] == 'Start'
            elif key == 'btn_set_goal': active = ui['draw_mode'] == 'Goal'
            btn.draw(win, active)

    sections = [("ALGORITHM", 110), ("HEURISTIC", 185), ("SIMULATION", 260), ("PLACEMENT", 335), ("STATS", 580)]
    for text, y in sections:
        glow = FONT_SM.render(text, True, UI_ACCENT)
        win.blit(glow, (px + 25, y))
        pygame.draw.line(win, GRID_LINE, (px + 25, y + 18), (WIDTH - 25, y + 18))

    y_st = 610
    for s in [f"Visited: {stats['visited']}", f"Path Cost: {stats['cost']}", f"Time: {stats['time']}ms"]:
        win.blit(FONT_MED.render(s, True, TEXT_MAIN), (px + 30, y_st))
        y_st += 25

def clear_search_visuals(grid):
    for row in grid:
        for n in row:
            if n.color in [NODE_OPEN, NODE_CLOSED, NODE_PATH]: n.reset()

def generate_maze(grid, start, end):
    for row in grid:
        for n in row:
            if n != start and n != end:
                n.reset()
                if random.random() < 0.3: 
                    n.make_barrier()

# --- MAIN CONTROLLER ---
def main():
    grid = [[Node(i, j) for j in range(ROWS)] for i in range(ROWS)]
    start = grid[5][5]
    end = grid[ROWS-6][ROWS-6]
    start.make_start()
    end.make_goal()
    
    px_btn = GRID_WIDTH + 25
    ui = {
        'algo':'A*', 'heuristic':'Manhattan', 'mode':'Static', 'draw_mode':'Wall',
        'btn_astar': ModernButton(px_btn, 135, 120, 30, 'A-STAR', 'A*'),
        'btn_gbfs': ModernButton(px_btn + 130, 135, 120, 30, 'GREEDY', 'GBFS'),
        'btn_man': ModernButton(px_btn, 210, 120, 30, 'MANHATTAN', 'Manhattan'),
        'btn_euc': ModernButton(px_btn + 130, 210, 120, 30, 'EUCLIDEAN', 'Euclidean'),
        'btn_static': ModernButton(px_btn, 285, 120, 30, 'STATIC', 'Static'),
        'btn_dynamic': ModernButton(px_btn + 130, 285, 120, 30, 'DYNAMIC', 'Dynamic'),
        'btn_draw_wall': ModernButton(px_btn, 360, 120, 30, 'WALLS', 'Wall'),
        'btn_draw_weight': ModernButton(px_btn + 130, 360, 120, 30, 'WEIGHTS', 'Weight'),
        'btn_set_start': ModernButton(px_btn, 400, 120, 30, 'START POS', 'Start'),
        'btn_set_goal': ModernButton(px_btn + 130, 400, 120, 30, 'GOAL POS', 'Goal'),
        'btn_start': ModernButton(px_btn, 455, 250, 40, 'START AGENT', 'Run'),
        'btn_gen_maze': ModernButton(px_btn, 505, 250, 30, 'GENERATE MAZE', 'Maze'),
        'btn_reset': ModernButton(px_btn, 545, 250, 30, 'RESET SYSTEM', 'Reset')
    }
    
    stats = {'visited': 0, 'cost': 0, 'time': 0}
    agent_moving = False
    agent_path = []
    last_obs_time = time.time()
    last_move_time = time.time()
    run = True

    def update_screen_callback():
        draw_grid(WIN, grid)
        draw_sidebar(WIN, ui, stats)
        pygame.display.update()

    while run:
        if agent_moving and agent_path:
            current_time = time.time()
            if current_time - last_move_time > 0.1:
                next_node = agent_path.pop()
                start.reset()
                start = next_node
                start.make_agent()
                last_move_time = current_time
                
                if start == end:
                    start.make_goal()
                    agent_moving = False

            if ui['mode'] == 'Dynamic' and current_time - last_obs_time > 0.01 and agent_moving:
                r, c = random.randint(0, ROWS-1), random.randint(0, ROWS-1)
                target = grid[r][c]
                
                if target.color in [NODE_EMPTY, NODE_OPEN, NODE_CLOSED, NODE_PATH] and target != start and target != end:
                    if random.random() > 0.3: target.make_barrier()
                    else: target.make_weight()
                    
                    if target in agent_path:
                        clear_search_visuals(grid)
                        t0 = time.perf_counter()
                        if ui['algo'] == "A*":
                            success, new_path, visited, cost = run_astar(grid, start, end, ui['heuristic'], None)
                        else:
                            success, new_path, visited, cost = run_gbfs(grid, start, end, ui['heuristic'], None)
                            
                        stats.update({'visited': visited, 'cost': cost, 'time': round((time.perf_counter()-t0)*1000, 2)})
                        if success: agent_path = new_path
                        else: agent_moving = False 
                
                last_obs_time = current_time

        update_screen_callback()

        for e in pygame.event.get():
            if e.type == pygame.QUIT: run = False
            if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                mx, my = e.pos
                if mx >= GRID_WIDTH:
                    for key, btn in ui.items():
                        if isinstance(btn, ModernButton) and btn.rect.collidepoint(e.pos):
                            if key in ['btn_astar','btn_gbfs']: ui['algo']=btn.val
                            elif key in ['btn_man','btn_euc']: ui['heuristic']=btn.val
                            elif key in ['btn_static','btn_dynamic']: ui['mode']=btn.val
                            elif key in ['btn_draw_wall','btn_draw_weight','btn_set_start','btn_set_goal']: ui['draw_mode']=btn.val
                            elif key == 'btn_gen_maze':
                                agent_moving = False
                                generate_maze(grid, start, end)
                            elif key == 'btn_start':
                                agent_moving = False
                                clear_search_visuals(grid)
                                start.make_start() 
                                t0 = time.perf_counter()
                                if ui['algo'] == "A*":
                                    success, path_list, visited, cost = run_astar(grid, start, end, ui['heuristic'], update_screen_callback)
                                else:
                                    success, path_list, visited, cost = run_gbfs(grid, start, end, ui['heuristic'], update_screen_callback)
                                stats.update({'visited': visited, 'cost': cost, 'time': round((time.perf_counter()-t0)*1000, 2)})
                                if success:
                                    agent_path = path_list
                                    agent_moving = True
                                    last_move_time = time.time()
                                    last_obs_time = time.time()
                            elif key == 'btn_reset':
                                agent_moving = False
                                stats = {'visited': 0, 'cost': 0, 'time': 0}
                                for row in grid:
                                    for n in row: 
                                        if n != start and n != end: n.reset()
                                start.make_start()
                else:
                    if not agent_moving: 
                        c, r = mx // CELL_SIZE, my // CELL_SIZE
                        if 0 <= r < ROWS and 0 <= c < ROWS:
                            node = grid[r][c]
                            if ui['draw_mode'] == 'Wall' and node not in [start, end]: node.make_barrier()
                            elif ui['draw_mode'] == 'Weight' and node not in [start, end]: node.make_weight()
                            elif ui['draw_mode'] == 'Start' and node != end: 
                                start.reset()
                                start = node
                                start.make_start()
                            elif ui['draw_mode'] == 'Goal' and node != start: 
                                end.reset()
                                end = node
                                end.make_goal()
            
            elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 3:
                 if not agent_moving:
                    mx, my = e.pos
                    if mx < GRID_WIDTH:
                        c, r = mx // CELL_SIZE, my // CELL_SIZE
                        if 0 <= r < ROWS and 0 <= c < ROWS:
                            node = grid[r][c]
                            if node != start and node != end: node.reset()

    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()
