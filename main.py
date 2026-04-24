import sys
import math
import pygame
import numpy as np

from map_config import GRID, ROWS, COLS, CELL_SIZE
from astar import astar

WIN_W = COLS * CELL_SIZE
WIN_H = ROWS * CELL_SIZE + 50

C_FREE      = (245, 245, 242)
C_WALL      = (40,  40,  45)
C_DYN_OBS   = (180, 50,  50)
C_GRID_LINE = (210, 210, 205)
C_PATH      = (100, 160, 240)
C_ROBOT     = (30,  110, 220)
C_ROBOT_RIM = (255, 255, 255)
C_GOAL      = (50,  195, 80)
C_SENSOR    = (80,  160, 240)
C_DETECTED  = (240, 120, 40)
C_REPLAN    = (240, 200, 40)
C_STATUS_BG = (30,  30,  35)
C_STATUS_TX = (220, 220, 220)
C_NO_PATH   = (220, 60,  50)
C_REACHED   = (50,  195, 80)
C_TRAIL     = (255, 160, 60)

SPEEDS = [
    ("Slow",   0.04),
    ("Normal", 0.09),
    ("Fast",   0.20),
]
SPEED_IDX = 1

sensor_range = 3
SENSOR_MIN   = 1
SENSOR_MAX   = 8

TRAIL_MAX_ALPHA   = 200
TRAIL_FADE        = 3
TRAIL_RECORD_DIST = 0.4


def get_sensor_readings(live_grid, robot_r, robot_c, s_range):
    detected = []
    cr, cc = int(round(robot_r)), int(round(robot_c))
    for angle_deg in range(0, 360, 22):
        angle = math.radians(angle_deg)
        for step in range(1, s_range + 1):
            nr = cr + int(round(step * math.sin(angle)))
            nc = cc + int(round(step * math.cos(angle)))
            if 0 <= nr < ROWS and 0 <= nc < COLS:
                if live_grid[nr, nc] >= 1:
                    detected.append((nr, nc))
                    break
    return list(set(detected))


def snap_to_free(live_grid, r, c):
    if live_grid[r, c] == 0:
        return (r, c)
    for dist in range(1, 5):
        for dr in range(-dist, dist + 1):
            for dc in range(-dist, dist + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < ROWS and 0 <= nc < COLS:
                    if live_grid[nr, nc] == 0:
                        return (nr, nc)
    return (r, c)


class Robot:
    def __init__(self, row, col):
        self.r = float(row)
        self.c = float(col)
        self.angle = 0.0

    @property
    def speed(self):
        return SPEEDS[SPEED_IDX][1]

    def move_toward(self, target_r, target_c):
        dr = target_r - self.r
        dc = target_c - self.c
        dist = (dr ** 2 + dc ** 2) ** 0.5
        if dist < self.speed:
            self.r, self.c = float(target_r), float(target_c)
            return True
        self.angle = math.atan2(dc, dr)
        self.r += self.speed * dr / dist
        self.c += self.speed * dc / dist
        return False

    def draw(self, surface):
        cx = int(self.c * CELL_SIZE + CELL_SIZE / 2)
        cy = int(self.r * CELL_SIZE + CELL_SIZE / 2)
        radius = CELL_SIZE // 2 - 3
        pygame.draw.circle(surface, C_ROBOT,     (cx, cy), radius)
        pygame.draw.circle(surface, C_ROBOT_RIM, (cx, cy), radius, 2)
        ex = cx + int((radius - 4) * math.cos(self.angle))
        ey = cy + int((radius - 4) * math.sin(self.angle))
        pygame.draw.circle(surface, C_ROBOT_RIM, (ex, ey), 3)

    def draw_sensor(self, surface):
        cx = int(self.c * CELL_SIZE + CELL_SIZE / 2)
        cy = int(self.r * CELL_SIZE + CELL_SIZE / 2)
        radius_px = sensor_range * CELL_SIZE
        sensor_surf = pygame.Surface((radius_px * 2, radius_px * 2), pygame.SRCALPHA)
        pygame.draw.circle(sensor_surf, (80, 160, 240, 30),
                           (radius_px, radius_px), radius_px)
        pygame.draw.circle(sensor_surf, (80, 160, 240, 90),
                           (radius_px, radius_px), radius_px, 2)
        surface.blit(sensor_surf, (cx - radius_px, cy - radius_px))


def update_trail(trail, robot_r, robot_c, last_crumb):
    """Fade existing crumbs; add a new one if robot moved far enough."""
    trail[:] = [(r, c, a - TRAIL_FADE) for (r, c, a) in trail if a - TRAIL_FADE > 0]
    lr, lc = last_crumb
    if ((robot_r - lr) ** 2 + (robot_c - lc) ** 2) ** 0.5 >= TRAIL_RECORD_DIST:
        trail.append((robot_r, robot_c, TRAIL_MAX_ALPHA))
        return (robot_r, robot_c)
    return last_crumb


def draw_trail(surface, trail):
    for (r, c, alpha) in trail:
        if alpha <= 0:
            continue
        cx = int(c * CELL_SIZE + CELL_SIZE / 2)
        cy = int(r * CELL_SIZE + CELL_SIZE / 2)
        dot = pygame.Surface((10, 10), pygame.SRCALPHA)
        pygame.draw.circle(dot, (*C_TRAIL, int(alpha)), (5, 5), 4)
        surface.blit(dot, (cx - 5, cy - 5))


def draw_obstacle_preview(surface, hover_cell, live_grid, phase):
    """Tint the hovered cell red (place) or green (remove) during navigation."""
    if hover_cell is None or phase not in ("running", "blocked"):
        return
    cr, cc = hover_cell
    if GRID[cr, cc] == 1:
        return
    existing = live_grid[cr, cc]
    fill   = (80,  200, 80,  80) if existing == 2 else (220, 60, 60,  80)
    border = (80,  200, 80, 200) if existing == 2 else (220, 60, 60, 200)
    s = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
    s.fill(fill)
    pygame.draw.rect(s, border, s.get_rect(), 2)
    surface.blit(s, (cc * CELL_SIZE, cr * CELL_SIZE))


def draw_grid(surface, live_grid, detected_cells):
    for r in range(ROWS):
        for c in range(COLS):
            rect = pygame.Rect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            if live_grid[r, c] == 2:
                color = C_DYN_OBS
            elif live_grid[r, c] >= 1:
                color = C_WALL
            else:
                color = C_FREE
            pygame.draw.rect(surface, color, rect)
            if (r, c) in detected_cells:
                pygame.draw.rect(surface, C_DETECTED, rect, 3)
            pygame.draw.rect(surface, C_GRID_LINE, rect, 1)


def draw_path(surface, path):
    for i, (r, c) in enumerate(path):
        alpha = max(60, 255 - i * 8)
        s = pygame.Surface((CELL_SIZE - 14, CELL_SIZE - 14), pygame.SRCALPHA)
        s.fill((100, 160, 240, alpha))
        surface.blit(s, (c * CELL_SIZE + 7, r * CELL_SIZE + 7))


def draw_goal(surface, goal):
    r, c = goal
    rect = pygame.Rect(c * CELL_SIZE + 4, r * CELL_SIZE + 4,
                       CELL_SIZE - 8, CELL_SIZE - 8)
    pygame.draw.rect(surface, C_GOAL, rect, border_radius=5)
    mx = c * CELL_SIZE + CELL_SIZE // 2
    my = r * CELL_SIZE + CELL_SIZE // 2
    pygame.draw.line(surface, (255, 255, 255), (mx - 6, my), (mx + 6, my), 2)
    pygame.draw.line(surface, (255, 255, 255), (mx, my - 6), (mx, my + 6), 2)


def draw_no_path_alert(surface, font, tick):
    """Pulsing centered alert box shown when no path exists."""
    # Pulse opacity between 180 and 255
    pulse = int(180 + 75 * abs(math.sin(tick * 0.05)))

    box_w, box_h = 380, 90
    bx = (WIN_W - box_w) // 2
    by = (WIN_H - 50 - box_h) // 2

    # Shadow
    shadow = pygame.Surface((box_w + 6, box_h + 6), pygame.SRCALPHA)
    shadow.fill((0, 0, 0, 80))
    surface.blit(shadow, (bx - 3, by + 3))

    # Box background
    box_surf = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
    box_surf.fill((40, 10, 10, min(pulse, 230)))
    surface.blit(box_surf, (bx, by))

    # Red border — two rects for thickness
    pygame.draw.rect(surface, (200, 50, 50), (bx, by, box_w, box_h), 3, border_radius=6)
    pygame.draw.rect(surface, (255, 80, 80), (bx + 2, by + 2, box_w - 4, box_h - 4), 1, border_radius=5)

    # Icon + heading
    big   = pygame.font.SysFont("Segoe UI", 20, bold=True)
    small = pygame.font.SysFont("Segoe UI", 14)

    heading = big.render("⚠  NO PATH FOUND", True, (255, 80, 80))
    sub     = small.render("Remove an obstacle to let the robot continue", True, (210, 160, 160))

    surface.blit(heading, (bx + (box_w - heading.get_width()) // 2, by + 18))
    surface.blit(sub,     (bx + (box_w - sub.get_width())     // 2, by + 54))


def draw_status(surface, font, text, color, extra, paused):
    bar = pygame.Rect(0, WIN_H - 50, WIN_W, 50)
    pygame.draw.rect(surface, C_STATUS_BG, bar)

    display_text  = f"PAUSED  —  {text}" if paused else text
    display_color = (240, 200, 40) if paused else color

    lbl = font.render(display_text, True, display_color)
    surface.blit(lbl, (12, WIN_H - 38))

    if extra and not paused:
        small = pygame.font.SysFont("Segoe UI", 13)
        sub = small.render(extra, True, (130, 130, 130))
        surface.blit(sub, (12, WIN_H - 18))
    elif paused:
        small = pygame.font.SysFont("Segoe UI", 13)
        sub = small.render("F = step one frame", True, (160, 160, 80))
        surface.blit(sub, (12, WIN_H - 18))

    speed_label = SPEEDS[SPEED_IDX][0]
    speed_colors = {"Slow": (100, 180, 255), "Normal": (180, 220, 100), "Fast": (255, 120, 80)}
    sc = speed_colors[speed_label]

    hint = font.render("R=reset  S=spd  [/]=sensor  Space=pause  F=step", True, (90, 90, 90))
    surface.blit(hint, (WIN_W - 400, WIN_H - 38))

    small2 = pygame.font.SysFont("Segoe UI", 13)
    info = small2.render(f"Speed: {speed_label}   Sensor: {sensor_range}", True, sc)
    surface.blit(info, (WIN_W - 400, WIN_H - 18))


def do_replan(robot, live_grid, goal):
    sr, sc = snap_to_free(live_grid, int(round(robot.r)), int(round(robot.c)))
    robot.r, robot.c = float(sr), float(sc)
    return astar(live_grid, (sr, sc), goal)


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("AMR Navigation — Obstacle Detection & Replanning")
    clock = pygame.time.Clock()
    font  = pygame.font.SysFont("Segoe UI", 17)

    global SPEED_IDX, sensor_range

    live_grid       = GRID.copy()
    robot           = None
    goal            = None
    path            = []
    wp_idx          = 0
    phase           = "place_start"
    status          = "Click a free cell to place the ROBOT"
    status_c        = C_STATUS_TX
    status_extra    = ""
    detected        = []
    replan_flash    = 0
    replan_count    = 0
    replan_cooldown = 0

    trail       = []
    last_crumb  = (0.0, 0.0)

    paused      = False
    do_step     = False
    tick        = 0

    hover_cell  = None

    while True:
        # Track mouse for obstacle preview
        mx_raw, my_raw = pygame.mouse.get_pos()
        if my_raw < WIN_H - 50:
            hc = mx_raw // CELL_SIZE
            hr = my_raw // CELL_SIZE
            hover_cell = (hr, hc) if (0 <= hr < ROWS and 0 <= hc < COLS) else None
        else:
            hover_cell = None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:

                if event.key == pygame.K_r:
                    live_grid       = GRID.copy()
                    robot           = None
                    goal            = None
                    path            = []
                    wp_idx          = 0
                    phase           = "place_start"
                    detected        = []
                    replan_count    = 0
                    replan_flash    = 0
                    replan_cooldown = 0
                    trail           = []
                    last_crumb      = (0.0, 0.0)
                    paused          = False
                    do_step         = False
                    tick            = 0
                    status          = "Click a free cell to place the ROBOT"
                    status_c        = C_STATUS_TX
                    status_extra    = ""

                elif event.key == pygame.K_s:
                    SPEED_IDX = (SPEED_IDX + 1) % len(SPEEDS)
                    status    = f"Speed set to: {SPEEDS[SPEED_IDX][0]}"
                    status_c  = C_STATUS_TX

                elif event.key == pygame.K_LEFTBRACKET:
                    sensor_range = max(SENSOR_MIN, sensor_range - 1)
                    status   = f"Sensor range: {sensor_range}"
                    status_c = C_STATUS_TX

                elif event.key == pygame.K_RIGHTBRACKET:
                    sensor_range = min(SENSOR_MAX, sensor_range + 1)
                    status   = f"Sensor range: {sensor_range}"
                    status_c = C_STATUS_TX

                elif event.key == pygame.K_SPACE:
                    if phase in ("running", "blocked", "done"):
                        paused = not paused

                elif event.key == pygame.K_f:
                    if paused and phase == "running":
                        do_step = True

            # Left click — place robot / goal / obstacle
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                if my >= WIN_H - 50:
                    continue
                cc = mx // CELL_SIZE
                cr = my // CELL_SIZE
                if not (0 <= cr < ROWS and 0 <= cc < COLS):
                    continue

                if phase == "place_start":
                    if live_grid[cr, cc] != 0:
                        status = "That cell is blocked — pick a free cell"
                        continue
                    robot      = Robot(cr, cc)
                    last_crumb = (float(cr), float(cc))
                    phase      = "place_goal"
                    status     = "Now click to place the GOAL"
                    status_extra = ""

                elif phase == "place_goal":
                    if live_grid[cr, cc] != 0:
                        status = "That cell is blocked — pick a free cell"
                        continue
                    goal   = (cr, cc)
                    path   = astar(live_grid, (int(robot.r), int(robot.c)), goal)
                    wp_idx = 0
                    if path:
                        phase        = "running"
                        status       = f"Path found — {len(path)} steps"
                        status_extra = "Click grid to place / remove obstacles"
                        replan_cooldown = 0
                    else:
                        goal     = None
                        status   = "No path found — pick another goal"
                        status_c = C_NO_PATH

                elif phase in ("running", "blocked"):
                    if GRID[cr, cc] == 1:
                        continue
                    if goal and (cr, cc) == goal:
                        continue
                    if robot and (int(round(robot.r)), int(round(robot.c))) == (cr, cc):
                        continue

                    live_grid[cr, cc] = 0 if live_grid[cr, cc] == 2 else 2

                    if goal and robot:
                        rr_r = int(round(robot.r))
                        rc_r = int(round(robot.c))
                        dist_obs = max(abs(cr - rr_r), abs(cc - rc_r))
                        if dist_obs <= sensor_range:
                            replan_cooldown = 0
                            new_path = do_replan(robot, live_grid, goal)
                            if new_path:
                                path         = new_path
                                wp_idx       = 0
                                replan_count += 1
                                replan_flash  = 30
                                phase        = "running"
                                status       = f"Obstacle placed! Replanned (#{replan_count})"
                                status_c     = C_REPLAN
                                replan_cooldown = 45
                            else:
                                phase    = "blocked"
                                status   = "No path exists! Remove an obstacle"
                                status_c = C_NO_PATH

        # ── Simulation tick ───────────────────────────────
        tick_allowed = (not paused) or do_step
        do_step = False

        if tick_allowed:
            if robot:
                detected = get_sensor_readings(live_grid, robot.r, robot.c, sensor_range)

            if phase == "running" and path and robot and replan_cooldown == 0:
                detected_set = set(detected)
                path_ahead   = set(path[wp_idx: wp_idx + sensor_range * 2])
                if detected_set & path_ahead:
                    new_path = do_replan(robot, live_grid, goal)
                    if new_path:
                        path         = new_path
                        wp_idx       = 0
                        replan_count += 1
                        replan_flash  = 30
                        status        = f"Path blocked! Replanned (#{replan_count})"
                        status_c      = C_REPLAN
                        replan_cooldown = 45
                    else:
                        phase    = "blocked"
                        status   = "No path exists! Remove an obstacle"
                        status_c = C_NO_PATH

            if replan_cooldown > 0:
                replan_cooldown -= 1

            if phase == "running" and robot and path and wp_idx < len(path):
                tr, tc = path[wp_idx]
                if robot.move_toward(tr, tc):
                    wp_idx += 1
                    if wp_idx >= len(path):
                        phase        = "done"
                        status       = f"Goal reached! Replanned {replan_count}x — Press R to restart"
                        status_c     = C_REACHED
                        status_extra = ""
                    elif replan_flash == 0:
                        status   = f"Navigating... step {wp_idx}/{len(path)}"
                        status_c = C_STATUS_TX

            if replan_flash > 0:
                replan_flash -= 1
                if replan_flash == 0:
                    status_c = C_STATUS_TX

            if robot and phase in ("running", "done"):
                last_crumb = update_trail(trail, robot.r, robot.c, last_crumb)

        tick += 1

        # ── Draw ──────────────────────────────────────────
        screen.fill(C_FREE)
        draw_grid(screen, live_grid, set(detected))

        if path and phase in ("running", "blocked"):
            draw_path(screen, path[wp_idx:])

        if goal:
            draw_goal(screen, goal)

        draw_trail(screen, trail)
        draw_obstacle_preview(screen, hover_cell, live_grid, phase)

        if robot:
            robot.draw_sensor(screen)
            robot.draw(screen)

        if phase == "blocked":
            draw_no_path_alert(surface=screen, font=font, tick=tick)

        draw_status(screen, font, status, status_c, status_extra, paused)

        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()
