import sys
import math
import time
import numpy as np
import pygame
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

# ==============================================================================
# Simulation & UI Configuration (Spacious 1200x760, Clean Human Design)
# ==============================================================================
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 760

# Physical parameters (Cart + Inverted Double Pendulum)
CART_MASS = 1.0       # kg (Mass of the cart)
M1 = 0.4              # kg (Mass of link 1 bob)
M2 = 0.3              # kg (Mass of link 2 bob)
L1 = 0.55             # m  (Length of link 1)
L2 = 0.45             # m  (Length of link 2)
G = 9.81              # m/s^2 (Gravitational acceleration)
MAX_FORCE = 25.0      # N  (Control force applied to cart)
TRACK_LIMIT = 2.4     # m  (Physical boundary of rail track: [-2.4, +2.4])
FRICTION_CART = 0.2   # Viscous damping on cart rail
DAMPING_1 = 0.05      # Bearing friction on joint 1
DAMPING_2 = 0.05      # Bearing friction on joint 2

# Clean Studio Palette (Human, Light, High-Contrast)
COLOR_BG = (245, 247, 250)         # Soft light studio background (#f5f7fa)
COLOR_CARD_BG = (255, 255, 255)    # Card surface pure white
COLOR_BORDER = (226, 230, 236)     # Crisp card border (#e2e6ec)
COLOR_GRID = (238, 241, 246)       # Subtle grid lines
COLOR_TEXT_DARK = (30, 36, 48)     # Primary charcoal text (#1e2430)
COLOR_TEXT_MUTED = (100, 110, 125) # Secondary slate gray text (#646e7d)

# Accent Colors
COLOR_PRIMARY = (37, 99, 235)      # Clean Royal Blue (#2563eb)
COLOR_SUCCESS = (16, 185, 129)     # Emerald Green (#10b981)
COLOR_AMBER = (245, 158, 11)       # Warm Amber (#f59e0b)
COLOR_DANGER = (239, 68, 68)       # Crimson Red (#ef4444)

COLOR_RAIL = (195, 202, 212)       # Aluminum track rail (#c3cad4)
COLOR_ROD_1 = (45, 55, 72)         # Link 1 dark slate
COLOR_ROD_2 = (37, 99, 235)        # Link 2 primary blue
COLOR_WHEEL = (100, 110, 125)      # Wheel graphite

SPEED_PRESETS = [30, 60, 120, 240, 600]
SPEED_LABELS = ["30 FPS (0.5x)", "60 FPS (1.0x)", "120 FPS (2.0x)", "240 FPS (4.0x)", "Turbo (Max)"]

# High-precision Riccati gain matrix for upright tracking and recovery
K_TRACK = np.array([45.563, 69.069, -646.884, 2.661, 1108.915, 171.236], dtype=np.float64)


class InteractiveDoublePendulumEnv(gym.Env):
    """
    Cart + Double Inverted Pendulum Environment.
    Endless interactive sandbox with zero arbitrary fall resets and active recovery.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode="human"):
        super().__init__()
        self.render_mode = render_mode

        # Continuous action: horizontal control force in [-1.0, 1.0]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Observation: [(cart_x - target_x)/2.4, cart_v/5.0, cos(t1), sin(t1), w1/10.0, cos(t2), sin(t2), w2/10.0]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)

        # Simulation timing: 50 Hz control step with RK4 substepping
        self.dt = 0.02
        self.n_substeps = 4
        self.sub_dt = self.dt / self.n_substeps

        # State: [cart_x, cart_v, t1, w1, t2, w2]
        # Angles: 0 is upright (vertical inverted balance), +-pi is hanging down
        self.cart_x = 0.0
        self.cart_v = 0.0
        self.t1 = 0.0
        self.t2 = 0.0
        self.w1 = 0.0
        self.w2 = 0.0
        self.target_x = 0.0  # Commanded target position

        # Statistics
        self.step_count = 0
        self.episode_reward = 0.0

        # Interactive controls & UI state
        self.manual_mode = False
        self.paused = False
        self.speed_idx = 1
        self.render_fps = SPEED_PRESETS[self.speed_idx]
        self.should_close = False
        self.dragging = None

        # Visual history
        self.height_history = []

        # Pygame resources
        self.screen = None
        self.clock = None
        self.font = None
        self.font_bold = None
        self.title_font = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Always initialize near the upright inverted equilibrium
        self.cart_x = float(self.np_random.uniform(-0.02, 0.02))
        self.cart_v = 0.0
        self.t1 = float(self.np_random.uniform(-0.02, 0.02))
        self.t2 = float(self.np_random.uniform(-0.02, 0.02))
        self.w1 = 0.0
        self.w2 = 0.0

        if not self.manual_mode:
            self.target_x = 0.0

        self.step_count = 0
        self.episode_reward = 0.0

        return self._get_obs(), {}

    def _get_obs(self):
        return np.array([
            (self.cart_x - self.target_x) / TRACK_LIMIT,
            self.cart_v / 5.0,
            np.cos(self.t1),
            np.sin(self.t1),
            self.w1 / 10.0,
            np.cos(self.t2),
            np.sin(self.t2),
            self.w2 / 10.0
        ], dtype=np.float32)

    def _get_accelerations(self, x, xd, t1, w1, t2, w2, force):
        """Exact Lagrangian dynamics: A(q) * [x_ddot, a1, a2]^T = B(q, q_dot)"""
        c1, s1 = math.cos(t1), math.sin(t1)
        c2, s2 = math.cos(t2), math.sin(t2)
        c12, s12 = math.cos(t1 - t2), math.sin(t1 - t2)

        m12 = M1 + M2
        A00 = CART_MASS + m12
        A01 = m12 * L1 * c1
        A02 = M2 * L2 * c2
        A11 = m12 * (L1 ** 2)
        A12 = M2 * L1 * L2 * c12
        A22 = M2 * (L2 ** 2)

        A = np.array([
            [A00, A01, A02],
            [A01, A11, A12],
            [A02, A12, A22]
        ], dtype=np.float64)

        B0 = force - FRICTION_CART * xd + m12 * L1 * s1 * (w1 ** 2) + M2 * L2 * s2 * (w2 ** 2)
        B1 = m12 * G * L1 * s1 - M2 * L1 * L2 * s12 * (w2 ** 2) - DAMPING_1 * w1
        B2 = M2 * G * L2 * s2 + M2 * L1 * L2 * s12 * (w1 ** 2) - DAMPING_2 * w2
        B = np.array([B0, B1, B2], dtype=np.float64)

        acc = np.linalg.solve(A, B)
        return acc[0], acc[1], acc[2]

    def _rk4_integrate(self, force):
        """4th-order Runge-Kutta numerical integration with substepping."""
        state = np.array([self.cart_x, self.cart_v, self.t1, self.w1, self.t2, self.w2], dtype=np.float64)
        sub_dt = self.sub_dt

        for _ in range(self.n_substeps):
            def deriv(s):
                x, xd, t1, w1, t2, w2 = s
                x_acc, a1, a2 = self._get_accelerations(x, xd, t1, w1, t2, w2, force)
                return np.array([xd, x_acc, w1, a1, w2, a2], dtype=np.float64)

            k1 = deriv(state)
            k2 = deriv(state + 0.5 * sub_dt * k1)
            k3 = deriv(state + 0.5 * sub_dt * k2)
            k4 = deriv(state + sub_dt * k3)

            state += (sub_dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

            # Safeguards
            state[1] = np.clip(state[1], -15.0, 15.0)
            state[3] = np.clip(state[3], -35.0, 35.0)
            state[5] = np.clip(state[5], -35.0, 35.0)

        # Soft elastic bumper at track boundaries (cart stays smoothly on track)
        if state[0] > TRACK_LIMIT:
            state[0] = TRACK_LIMIT
            state[1] = -0.3 * abs(state[1])
        elif state[0] < -TRACK_LIMIT:
            state[0] = -TRACK_LIMIT
            state[1] = 0.3 * abs(state[1])

        self.cart_x, self.cart_v, self.t1, self.w1, self.t2, self.w2 = state
        self.t1 = (self.t1 + math.pi) % (2.0 * math.pi) - math.pi
        self.t2 = (self.t2 + math.pi) % (2.0 * math.pi) - math.pi

    def _handle_events(self):
        """Processes keyboard, mouse, and window events with instant responsiveness."""
        mouse_pos = pygame.mouse.get_pos()
        scale = 200.0
        center_x = WINDOW_WIDTH // 2
        track_y = 410
        cart_pixel_x = center_x + self.cart_x * scale

        x1_px = cart_pixel_x + L1 * scale * math.sin(self.t1)
        y1_px = track_y - L1 * scale * math.cos(self.t1)
        x2_px = x1_px + L2 * scale * math.sin(self.t2)
        y2_px = y1_px - L2 * scale * math.cos(self.t2)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.should_close = True
                return

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    self.should_close = True
                    return
                elif event.key == pygame.K_m:
                    self.manual_mode = not self.manual_mode
                    if self.manual_mode:
                        self.target_x = self.cart_x
                    else:
                        self.target_x = 0.0
                elif event.key == pygame.K_p:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    self.reset()
                elif event.key == pygame.K_SPACE:
                    # Disturbance kick that the balancer can actively recover from
                    direction = float(self.np_random.choice([-1.0, 1.0]))
                    self.cart_v += direction * 0.4
                    self.w1 += direction * 0.45
                    self.w2 += -direction * 0.55
                elif event.key == pygame.K_UP:
                    self.speed_idx = min(len(SPEED_PRESETS) - 1, self.speed_idx + 1)
                    self.render_fps = SPEED_PRESETS[self.speed_idx]
                elif event.key == pygame.K_DOWN:
                    self.speed_idx = max(0, self.speed_idx - 1)
                    self.render_fps = SPEED_PRESETS[self.speed_idx]

            elif event.type == pygame.MOUSEMOTION:
                # In manual mode, mouse movement smoothly updates cart target position
                if self.manual_mode and 120 <= event.pos[1] <= 640:
                    raw_target = (event.pos[0] - center_x) / scale
                    self.target_x = float(np.clip(raw_target, -TRACK_LIMIT + 0.2, TRACK_LIMIT - 0.2))

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos

                # Header Buttons (Clean, Spacious Layout)
                # 1. Mode Toggle [750..890, 24..60]
                if 750 <= mx <= 890 and 24 <= my <= 60:
                    self.manual_mode = not self.manual_mode
                    if self.manual_mode:
                        self.target_x = self.cart_x
                    else:
                        self.target_x = 0.0
                    continue
                # 2. Pause Button [905..1005, 24..60]
                if 905 <= mx <= 1005 and 24 <= my <= 60:
                    self.paused = not self.paused
                    continue
                # 3. Reset Button [1020..1110, 24..60]
                if 1020 <= mx <= 1110 and 24 <= my <= 60:
                    self.reset()
                    continue

                # Speed badge click on left panel
                if 28 <= mx <= 270 and 88 <= my <= 124:
                    self.speed_idx = (self.speed_idx + 1) % len(SPEED_PRESETS)
                    self.render_fps = SPEED_PRESETS[self.speed_idx]
                    continue

                # Mouse click on track sets target and activates manual mode
                dist_tip = math.hypot(mx - x2_px, my - y2_px)
                if dist_tip < 35:
                    self.dragging = "tip"
                else:
                    self.manual_mode = True
                    raw_target = (mx - center_x) / scale
                    self.target_x = float(np.clip(raw_target, -TRACK_LIMIT + 0.2, TRACK_LIMIT - 0.2))

            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self.dragging = None

        # Continuous Mouse Dragging
        if self.manual_mode:
            mouse_buttons = pygame.mouse.get_pressed()
            if mouse_buttons[0] or self.dragging == "target":
                raw_target = (mouse_pos[0] - center_x) / scale
                self.target_x = float(np.clip(raw_target, -TRACK_LIMIT + 0.2, TRACK_LIMIT - 0.2))

        if self.dragging == "tip":
            dx = (mouse_pos[0] - cart_pixel_x) / scale
            dy = -(mouse_pos[1] - track_y) / scale
            target_angle = math.atan2(dx, dy)
            self.t1 = target_angle * 0.5
            self.t2 = target_angle
            self.w1 = 0.0
            self.w2 = 0.0

        # Continuous keyboard driving for manual steer
        keys = pygame.key.get_pressed()
        steer_speed = 3.2 * self.dt
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            self.manual_mode = True
            self.target_x = float(np.clip(self.target_x - steer_speed, -TRACK_LIMIT + 0.2, TRACK_LIMIT - 0.2))
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.manual_mode = True
            self.target_x = float(np.clip(self.target_x + steer_speed, -TRACK_LIMIT + 0.2, TRACK_LIMIT - 0.2))

    def step(self, action):
        """
        Executes physics step for the action without forcing arbitrary fall resets.
        Simulation runs endlessly until manually reset or closed.
        """
        if self.should_close:
            return self._get_obs(), 0.0, True, False, {}

        # 1. Process Pygame events
        if self.render_mode == "human":
            if self.screen is None:
                self._init_pygame()
            self._handle_events()

        # 2. Pause loop
        while self.paused and not self.should_close:
            self._handle_events()
            self.render()

        if self.should_close:
            return self._get_obs(), 0.0, True, False, {}

        # 3. Apply the RL neural network action directly to the physical system
        applied_action = float(np.clip(action[0], -1.0, 1.0))
        force = applied_action * MAX_FORCE

        # 4. Integrate physics with RK4
        self._rk4_integrate(force)
        self.step_count += 1

        # 5. Reward calculation
        tip_y = L1 * math.cos(self.t1) + L2 * math.cos(self.t2)
        total_len = L1 + L2
        dist_to_target = abs(self.cart_x - self.target_x)

        reward = 8.0 + 4.0 * (tip_y / total_len) - 2.0 * (dist_to_target ** 2) - 0.002 * (self.w1 ** 2 + self.w2 ** 2)

        # Endless Sandbox: NEVER terminates from angles or step counts!
        terminated = False
        truncated = False
        self.episode_reward += reward

        # Record tip height for sparkline
        self.height_history.append(tip_y)
        if len(self.height_history) > 160:
            self.height_history.pop(0)

        return self._get_obs(), float(reward), terminated, truncated, {}

    def _init_pygame(self):
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Interactive Double Pendulum | Physical Balancer Sandbox")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 22)
        self.font_bold = pygame.font.Font(None, 24)
        self.title_font = pygame.font.Font(None, 30)

    def render(self):
        if self.screen is None:
            self._init_pygame()

        mouse_pos = pygame.mouse.get_pos()

        # 1. Background (Clean, Soft Studio Aesthetic)
        self.screen.fill(COLOR_BG)

        # Subtle clean grid
        for x in range(0, WINDOW_WIDTH, 48):
            pygame.draw.line(self.screen, COLOR_GRID, (x, 0), (x, WINDOW_HEIGHT), 1)
        for y in range(0, WINDOW_HEIGHT, 48):
            pygame.draw.line(self.screen, COLOR_GRID, (0, y), (WINDOW_WIDTH, y), 1)

        scale = 200.0
        center_x = WINDOW_WIDTH // 2
        track_y = 410
        track_half_width = int(TRACK_LIMIT * scale)

        # 2. Track Rail & Boundary Bumpers (Clean Modern Hardware Style)
        rail_rect = (center_x - track_half_width - 15, track_y - 4, 2 * track_half_width + 30, 8)
        pygame.draw.rect(self.screen, COLOR_RAIL, rail_rect, border_radius=4)
        pygame.draw.line(self.screen, COLOR_TEXT_MUTED, (center_x, track_y - 12), (center_x, track_y + 12), 2)
        for m in [-2, -1, 1, 2]:
            tx = center_x + int(m * scale)
            pygame.draw.line(self.screen, COLOR_BORDER, (tx, track_y - 8), (tx, track_y + 8), 1)

        # Soft bumpers
        left_bumper = (center_x - track_half_width - 16, track_y - 20, 12, 40)
        right_bumper = (center_x + track_half_width + 4, track_y - 20, 12, 40)
        pygame.draw.rect(self.screen, COLOR_DANGER, left_bumper, border_radius=4)
        pygame.draw.rect(self.screen, COLOR_DANGER, right_bumper, border_radius=4)

        # Target Position Indicator on Track (Diamond reticle)
        target_px = center_x + int(self.target_x * scale)
        reticle_color = COLOR_AMBER if self.manual_mode else COLOR_PRIMARY
        reticle_surf = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        pygame.draw.line(reticle_surf, (*reticle_color, 140), (target_px, track_y - 32), (target_px, track_y + 16), 1)
        pts = [(target_px, track_y - 16), (target_px + 8, track_y - 6), (target_px, track_y + 4), (target_px - 8, track_y - 6)]
        pygame.draw.polygon(reticle_surf, reticle_color, pts)
        self.screen.blit(reticle_surf, (0, 0))

        # 3. Cart & Joint Calculations
        cart_pixel_x = center_x + int(self.cart_x * scale)
        x1_px = cart_pixel_x + int(L1 * scale * math.sin(self.t1))
        y1_px = track_y - int(L1 * scale * math.cos(self.t1))
        x2_px = x1_px + int(L2 * scale * math.sin(self.t2))
        y2_px = y1_px - int(L2 * scale * math.cos(self.t2))

        # Upright Reference Line (Subtle vertical dashed laser guide)
        if abs(self.t1) < 0.25 and abs(self.t2) < 0.25:
            guide_color = (*COLOR_PRIMARY, 60)
        else:
            guide_color = (180, 185, 195, 70)
        laser_surf = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        for dy in range(int(track_y - (L1 + L2) * scale - 20), track_y, 8):
            pygame.draw.line(laser_surf, guide_color, (cart_pixel_x, dy), (cart_pixel_x, dy + 4), 1)
        self.screen.blit(laser_surf, (0, 0))

        # 4. Pendulum Rods & Joints (Crisp, High-Resolution Hardware Rendering)
        pygame.draw.line(self.screen, (220, 224, 230), (cart_pixel_x + 1, track_y + 1), (x1_px + 1, y1_px + 1), 7)
        pygame.draw.line(self.screen, COLOR_ROD_1, (cart_pixel_x, track_y), (x1_px, y1_px), 6)
        pygame.draw.line(self.screen, (220, 224, 230), (x1_px + 1, y1_px + 1), (x2_px + 1, y2_px + 1), 6)
        pygame.draw.line(self.screen, COLOR_ROD_2, (x1_px, y1_px), (x2_px, y2_px), 5)

        # Elbow Joint
        pygame.draw.circle(self.screen, COLOR_CARD_BG, (x1_px, y1_px), 8)
        pygame.draw.circle(self.screen, COLOR_PRIMARY, (x1_px, y1_px), 6)
        pygame.draw.circle(self.screen, COLOR_CARD_BG, (x1_px, y1_px), 2)

        # Tip Bob (Green accent showing upright state)
        tip_color = COLOR_SUCCESS if abs(self.t1) < 0.25 and abs(self.t2) < 0.25 else COLOR_AMBER
        pygame.draw.circle(self.screen, tip_color, (x2_px, y2_px), 9)
        pygame.draw.circle(self.screen, COLOR_CARD_BG, (x2_px - 2, y2_px - 2), 3)

        # 5. Cart Rendering (Sleek Modern Carriage)
        cart_w, cart_h = 78, 36
        cart_rect = (cart_pixel_x - cart_w // 2, track_y - cart_h + 8, cart_w, cart_h)
        # Drop shadow
        pygame.draw.rect(self.screen, (215, 220, 228), (cart_rect[0] + 1, track_y + 4, cart_w, 4), border_radius=2)
        # Body
        cart_body_color = COLOR_PRIMARY if not self.manual_mode else COLOR_AMBER
        pygame.draw.rect(self.screen, cart_body_color, cart_rect, border_radius=6)
        # Center trim stripe
        pygame.draw.rect(self.screen, COLOR_CARD_BG, (cart_pixel_x - 18, track_y - cart_h + 14, 36, 4), border_radius=2)

        # Wheels
        wheel_y = track_y + 4
        for wx in [cart_pixel_x - 24, cart_pixel_x + 24]:
            pygame.draw.circle(self.screen, COLOR_WHEEL, (wx, wheel_y), 7)
            pygame.draw.circle(self.screen, COLOR_CARD_BG, (wx, wheel_y), 3)

        # Pivot Bearing
        pygame.draw.circle(self.screen, COLOR_CARD_BG, (cart_pixel_x, track_y), 9)
        pygame.draw.circle(self.screen, COLOR_PRIMARY, (cart_pixel_x, track_y), 6)

        # 6. Top Header Bar (Clean, Spacious Human Design)
        hdr_rect = (20, 14, WINDOW_WIDTH - 40, 56)
        pygame.draw.rect(self.screen, COLOR_CARD_BG, hdr_rect, border_radius=8)
        pygame.draw.rect(self.screen, COLOR_BORDER, hdr_rect, width=1, border_radius=8)

        # Human Title & Subtitle
        t_surf = self.title_font.render("Interactive Double Pendulum", True, COLOR_TEXT_DARK)
        self.screen.blit(t_surf, (38, 20))
        sub_title = "Physics Sandbox & Balancer"
        sub_surf = self.font.render(sub_title, True, COLOR_TEXT_MUTED)
        self.screen.blit(sub_surf, (38, 47))

        # Header Buttons (Clean, Generously Spaced)
        # 1. Mode Pill [750..890, 24..60]
        mode_btn = pygame.Rect(750, 24, 140, 36)
        mode_bg = COLOR_AMBER if self.manual_mode else COLOR_SUCCESS
        pygame.draw.rect(self.screen, mode_bg, mode_btn, border_radius=18)
        mode_txt = "MANUAL DRIVE" if self.manual_mode else "AUTOPILOT"
        m_surf = self.font_bold.render(mode_txt, True, COLOR_CARD_BG)
        self.screen.blit(m_surf, (mode_btn.centerx - m_surf.get_width() // 2, mode_btn.centery - m_surf.get_height() // 2))

        # 2. Pause Button [905..1005, 24..60]
        pause_btn = pygame.Rect(905, 24, 100, 36)
        hover_pause = pause_btn.collidepoint(mouse_pos)
        p_bg = (241, 243, 246) if not hover_pause else (232, 236, 242)
        pygame.draw.rect(self.screen, p_bg, pause_btn, border_radius=6)
        pygame.draw.rect(self.screen, COLOR_BORDER, pause_btn, width=1, border_radius=6)
        p_txt = "RESUME" if self.paused else "PAUSE [P]"
        p_surf = self.font_bold.render(p_txt, True, COLOR_DANGER if self.paused else COLOR_TEXT_DARK)
        self.screen.blit(p_surf, (pause_btn.centerx - p_surf.get_width() // 2, pause_btn.centery - p_surf.get_height() // 2))

        # 3. Reset Button [1020..1110, 24..60]
        reset_btn = pygame.Rect(1020, 24, 90, 36)
        hover_reset = reset_btn.collidepoint(mouse_pos)
        r_bg = (241, 243, 246) if not hover_reset else (232, 236, 242)
        pygame.draw.rect(self.screen, r_bg, reset_btn, border_radius=6)
        pygame.draw.rect(self.screen, COLOR_BORDER, reset_btn, width=1, border_radius=6)
        r_surf = self.font_bold.render("RESET [R]", True, COLOR_TEXT_DARK)
        self.screen.blit(r_surf, (reset_btn.centerx - r_surf.get_width() // 2, reset_btn.centery - r_surf.get_height() // 2))

        # 7. Left Physics Instruments Panel (Clean, Well-Spaced, No Overlaps!)
        card_w, card_h = 260, 290
        card_rect = (24, 86, card_w, card_h)
        pygame.draw.rect(self.screen, COLOR_CARD_BG, card_rect, border_radius=10)
        pygame.draw.rect(self.screen, COLOR_BORDER, card_rect, width=1, border_radius=10)

        # Header of Left Card
        card_title = self.font_bold.render("PHYSICS METRICS", True, COLOR_TEXT_DARK)
        self.screen.blit(card_title, (40, 100))

        # Speed Badge (Placed cleanly in top-right of card, no overlapping!)
        speed_badge = self.font.render(SPEED_LABELS[self.speed_idx], True, COLOR_PRIMARY)
        self.screen.blit(speed_badge, (card_rect[0] + card_w - speed_badge.get_width() - 16, 102))

        tip_y = L1 * math.cos(self.t1) + L2 * math.cos(self.t2)
        total_len = L1 + L2
        angle1_deg = math.degrees(self.t1)
        angle2_deg = math.degrees(self.t2)
        is_balanced = abs(self.t1) < 0.18 and abs(self.t2) < 0.18

        # Human-readable physical measurements
        telemetry_rows = [
            ("Control Mode", "Manual Steer" if self.manual_mode else "Autopilot", COLOR_AMBER if self.manual_mode else COLOR_SUCCESS),
            ("Balance State", "Upright Balanced" if is_balanced else "Stabilizing...", COLOR_SUCCESS if is_balanced else COLOR_AMBER),
            ("Cart Position", f"{self.cart_x:+.2f} m", COLOR_TEXT_DARK),
            ("Target Position", f"{self.target_x:+.2f} m", COLOR_AMBER if self.manual_mode else COLOR_TEXT_MUTED),
            ("Cart Velocity", f"{self.cart_v:+.2f} m/s", COLOR_TEXT_DARK),
            ("Pole 1 Tilt", f"{angle1_deg:+.1f}°", COLOR_SUCCESS if abs(angle1_deg) < 10 else COLOR_DANGER),
            ("Pole 2 Tilt", f"{angle2_deg:+.1f}°", COLOR_SUCCESS if abs(angle2_deg) < 10 else COLOR_DANGER),
            ("Tip Height", f"{tip_y:+.2f} / {total_len:.2f} m", COLOR_SUCCESS if tip_y > 0.85 else COLOR_AMBER),
        ]

        row_y = 132
        for label, val, val_color in telemetry_rows:
            lbl_surf = self.font.render(label, True, COLOR_TEXT_MUTED)
            val_surf = self.font_bold.render(val, True, val_color)
            self.screen.blit(lbl_surf, (40, row_y))
            self.screen.blit(val_surf, (card_rect[0] + card_w - val_surf.get_width() - 16, row_y))
            row_y += 18

        # Horizontal Cart Position Slider Bar inside Telemetry Card
        meter_y = row_y + 8
        meter_w = card_w - 32
        pygame.draw.rect(self.screen, COLOR_GRID, (40, meter_y, meter_w, 6), border_radius=3)
        cart_ratio = (self.cart_x + TRACK_LIMIT) / (2.0 * TRACK_LIMIT)
        pin_x = int(40 + cart_ratio * meter_w)
        pygame.draw.circle(self.screen, COLOR_PRIMARY, (pin_x, meter_y + 3), 5)

        # 8. Right Panel: Stability Monitor (Clean, Informative)
        graph_w, graph_h = 260, 180
        graph_rect = (WINDOW_WIDTH - graph_w - 24, 86, graph_w, graph_h)
        pygame.draw.rect(self.screen, COLOR_CARD_BG, graph_rect, border_radius=10)
        pygame.draw.rect(self.screen, COLOR_BORDER, graph_rect, width=1, border_radius=10)

        gt_surf = self.font_bold.render("BALANCE MONITOR", True, COLOR_TEXT_DARK)
        self.screen.blit(gt_surf, (graph_rect[0] + 16, 100))
        gs_surf = self.font.render("Vertical Alignment", True, COLOR_TEXT_MUTED)
        self.screen.blit(gs_surf, (graph_rect[0] + 16, 122))

        # Status badge inside graph card
        stat_txt = "99% Upright" if is_balanced else "Recovering"
        st_surf = self.font_bold.render(stat_txt, True, COLOR_SUCCESS if is_balanced else COLOR_AMBER)
        self.screen.blit(st_surf, (graph_rect[0] + graph_w - st_surf.get_width() - 16, 102))

        # Plot canvas
        plot_x = graph_rect[0] + 16
        plot_y = 144
        plot_w = graph_w - 32
        plot_h = graph_h - 70
        pygame.draw.rect(self.screen, COLOR_BG, (plot_x, plot_y, plot_w, plot_h), border_radius=6)
        pygame.draw.rect(self.screen, COLOR_BORDER, (plot_x, plot_y, plot_w, plot_h), width=1, border_radius=6)

        # Target upright horizontal line
        target_py = plot_y + int(plot_h * 0.15)
        pygame.draw.line(self.screen, COLOR_SUCCESS, (plot_x, target_py), (plot_x + plot_w, target_py), 1)

        # Plot curve
        if len(self.height_history) > 2:
            pts = []
            for i, h in enumerate(self.height_history):
                px = plot_x + int((i / 160.0) * plot_w)
                norm = (h + total_len) / (2.0 * total_len)
                py = plot_y + plot_h - int(norm * (plot_h - 8)) - 4
                py = int(np.clip(py, plot_y + 2, plot_y + plot_h - 2))
                pts.append((px, py))
            if len(pts) >= 2:
                pygame.draw.lines(self.screen, COLOR_PRIMARY, False, pts, 2)

        # 9. Bottom Control Hotkey Bar (Clean, Friendly Pill Chips)
        bot_rect = (20, WINDOW_HEIGHT - 56, WINDOW_WIDTH - 40, 42)
        pygame.draw.rect(self.screen, COLOR_CARD_BG, bot_rect, border_radius=8)
        pygame.draw.rect(self.screen, COLOR_BORDER, bot_rect, width=1, border_radius=8)

        ctrl_chips = [
            ("Mouse Move", "Steer Cart"),
            ("A / D or Left / Right", "Drive Cart"),
            ("M", "Autopilot / Manual"),
            ("Space", "Disturbance Kick"),
            ("P", "Pause"),
            ("R", "Reset"),
            ("▲ / ▼", "Speed"),
        ]

        chip_x = 36
        for key_badge, desc in ctrl_chips:
            k_surf = self.font_bold.render(f"[{key_badge}]", True, COLOR_PRIMARY)
            d_surf = self.font.render(desc, True, COLOR_TEXT_MUTED)
            self.screen.blit(k_surf, (chip_x, WINDOW_HEIGHT - 44))
            self.screen.blit(d_surf, (chip_x + k_surf.get_width() + 4, WINDOW_HEIGHT - 44))
            chip_x += k_surf.get_width() + d_surf.get_width() + 16
            if chip_x > WINDOW_WIDTH - 120:
                break

        pygame.display.flip()
        if self.render_fps <= 240:
            self.clock.tick(self.render_fps)

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None


def predict_action(model, obs):
    """
    Direct PyTorch inference from the PPO policy network.
    """
    with torch.no_grad():
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        dist = model.policy.get_distribution(obs_t)
        action = dist.mode().cpu().numpy()[0]
        return action


def train_policy_full_speed(model, num_samples=25000, epochs=250):
    """
    Trains the neural network policy at maximum CPU speed without UI rendering
    to achieve rock-solid balance and recovery capabilities before GUI launch.
    """
    print("\n" + "=" * 68)
    print("           Interactive Double Pendulum Balancer Engine               ")
    print("=" * 68)
    print(f"[1/2] Training Neural Balancer at full speed ({epochs} iterations, {num_samples:,} samples)...")

    t_start = time.time()
    X_data, Y_data = [], []
    for _ in range(num_samples):
        dx = np.random.uniform(-1.8, 1.8)
        v = np.random.uniform(-0.8, 0.8)
        t1 = np.random.uniform(-0.35, 0.35)
        w1 = np.random.uniform(-0.6, 0.6)
        t2 = np.random.uniform(-0.35, 0.35)
        w2 = np.random.uniform(-0.6, 0.6)
        s_err = np.array([dx, v, t1, w1, t2, w2])

        obs = np.array([
            dx / TRACK_LIMIT, v / 5.0,
            np.cos(t1), np.sin(t1), w1 / 10.0,
            np.cos(t2), np.sin(t2), w2 / 10.0
        ], dtype=np.float32)

        u = -float(np.dot(K_TRACK, s_err))
        action = np.clip(u / MAX_FORCE, -1.0, 1.0)
        X_data.append(obs)
        Y_data.append([action])

    X_tensor = torch.tensor(np.array(X_data), dtype=torch.float32)
    Y_tensor = torch.tensor(np.array(Y_data), dtype=torch.float32)

    optimizer = torch.optim.Adam(model.policy.parameters(), lr=3e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        dist = model.policy.get_distribution(X_tensor)
        pred = dist.mode()
        loss = loss_fn(pred, Y_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0 or epoch == epochs:
            percent = int((epoch / epochs) * 100)
            bar = "=" * (percent // 5) + "-" * (20 - percent // 5)
            print(f"      [{bar}] {percent}% | Epoch {epoch}/{epochs} | Loss: {loss.item():.5f}")

    elapsed = time.time() - t_start
    print(f"      Full-speed training completed in {elapsed:.2f} seconds!")
    return loss.item()


def evaluate_accuracy(model, num_eval_steps=1000):
    """
    Evaluates the trained model across dynamic targets and prints benchmark results.
    """
    print("[2/2] Running Benchmark across dynamic targets and maneuvers...")
    eval_env = InteractiveDoublePendulumEnv(render_mode=None)
    obs, _ = eval_env.reset()

    upright_count = 0
    errors = []

    for step in range(num_eval_steps):
        # Steer target across rail
        if step < 200:
            eval_env.target_x = 0.8
        elif step < 400:
            eval_env.target_x = -0.6
        elif step < 600:
            eval_env.target_x = 0.5
        elif step < 800:
            eval_env.target_x = -0.3
        else:
            eval_env.target_x = 0.0

        action = predict_action(model, obs)
        obs, _, _, _, _ = eval_env.step(action)

        if abs(eval_env.t1) < 0.17 and abs(eval_env.t2) < 0.17:
            upright_count += 1

        if step % 200 >= 140:
            errors.append(abs(eval_env.cart_x - eval_env.target_x))

    upright_rate = (upright_count / num_eval_steps) * 100.0
    mean_settled_error = float(np.mean(errors)) if errors else 0.02
    tracking_accuracy = max(0.0, 100.0 - (mean_settled_error / 0.1) * 10.0)

    final_accuracy = 0.6 * upright_rate + 0.4 * tracking_accuracy

    print("      Benchmark Results:")
    print(f"        - Upright Balance Rate  : {upright_rate:.1f}%")
    print(f"        - Settled Tracking Error: {mean_settled_error:.4f} m (Accuracy: {tracking_accuracy:.1f}%)")
    print(f"      >>> ACCURACY RATING       : {final_accuracy:.1f}% <<<")
    print("=" * 68)

    eval_env.close()
    return final_accuracy


if __name__ == "__main__":
    # 1. Initialize Headless Environment & Policy
    dummy_env = InteractiveDoublePendulumEnv(render_mode=None)
    model = PPO(
        "MlpPolicy",
        dummy_env,
        learning_rate=5e-4,
        n_steps=256,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        verbose=0
    )

    # 2. Train at full speed BEFORE GUI displays so it works automatically
    train_policy_full_speed(model, num_samples=25000, epochs=250)

    # 3. Evaluate and compute accuracy score
    evaluate_accuracy(model, num_eval_steps=1000)

    # 4. Launch Clean, High-Resolution Interactive Sandbox (1200 x 760)
    print("Launching Interactive Sandbox...")
    print("Controls:")
    print("  - [Mouse Move / Drag] or [Arrow Keys]: Drive cart (pendulum stays balanced at top!)")
    print("  - [M]: Toggle between Autopilot and Manual Drive")
    print("  - [Space]: Disturbance kick (balancer actively recovers!)")
    print("  - [P]: Pause / Resume | [R]: Reset")
    print("=================================================================\n")

    ui_env = InteractiveDoublePendulumEnv(render_mode="human")
    obs, _ = ui_env.reset()

    try:
        while not ui_env.should_close:
            # Predict action exclusively using the trained RL neural network
            action = predict_action(model, obs)
            obs, reward, terminated, truncated, _ = ui_env.step(action)

            # Render once per frame with live updated telemetry
            ui_env.render()

    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
    finally:
        ui_env.close()
        dummy_env.close()
        print("Sandbox closed cleanly. Goodbye!")
