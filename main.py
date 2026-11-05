import sys
import math
import threading
import time
import numpy as np
import pygame
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# ==============================================================================
# Simulation & UI Configuration
# ==============================================================================
WINDOW_WIDTH = 1024
WINDOW_HEIGHT = 680

# Physics parameters (Cart + Inverted Double Pendulum)
CART_MASS = 1.0       # kg (Mass of the cart)
M1 = 0.4              # kg (Mass of link 1 bob)
M2 = 0.3              # kg (Mass of link 2 bob)
L1 = 0.55             # m  (Length of link 1)
L2 = 0.45             # m  (Length of link 2)
G = 9.81              # m/s^2 (Gravitational acceleration)
MAX_FORCE = 25.0      # N  (Maximum control force applied to cart)
TRACK_LIMIT = 2.4     # m  (Physical boundary of rail track: [-2.4, +2.4])
FRICTION_CART = 0.2   # Viscous damping on cart rail
DAMPING_1 = 0.05      # Bearing friction on joint 1
DAMPING_2 = 0.05      # Bearing friction on joint 2

# Visual Palette (Dark Engineering / Lab Theme)
BG_COLOR = (15, 20, 29)
GRID_COLOR = (24, 32, 46)
PANEL_BG = (22, 29, 43)
PANEL_BORDER = (42, 53, 76)
TEXT_WHITE = (240, 245, 255)
TEXT_MUTED = (135, 150, 172)
TEXT_DIM = (85, 100, 120)

COLOR_CYAN = (0, 229, 255)       # Link 1 & Info Highlights
COLOR_PINK = (255, 45, 117)      # Link 2 & Tip Trail
COLOR_GREEN = (16, 185, 129)     # AI Autopilot / Balanced status
COLOR_AMBER = (245, 158, 11)     # Manual Mode / Caution
COLOR_RED = (239, 68, 68)        # Boundary / Danger
COLOR_TRACK = (70, 85, 110)      # Rail

SPEED_PRESETS = [30, 60, 120, 240, 600]
SPEED_LABELS = ["30 FPS (0.5x)", "60 FPS (1.0x)", "120 FPS (2.0x)", "240 FPS (4.0x)", "Turbo (Max)"]


class InteractiveDoublePendulumEnv(gym.Env):
    """
    Cart + Double Inverted Pendulum Environment.
    The action is 100% computed by the trained RL neural network.
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
        self.target_x = 0.0  # Steerable target position

        # Episode stats
        self.step_count = 0
        self.max_episode_steps = 1000
        self.episode_reward = 0.0
        self.recent_rewards = []
        self.avg_reward = 0.0
        self.total_episodes = 0

        # Interactive controls & UI state
        self.manual_mode = False
        self.paused = False
        self.show_trail = True
        self.speed_idx = 1
        self.render_fps = SPEED_PRESETS[self.speed_idx]
        self.should_close = False
        self.dragging = None

        # Visual histories
        self.tip_history = []
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
        self.cart_x = float(self.np_random.uniform(-0.04, 0.04))
        self.cart_v = 0.0
        self.t1 = float(self.np_random.uniform(-0.04, 0.04))
        self.t2 = float(self.np_random.uniform(-0.04, 0.04))
        self.w1 = float(self.np_random.uniform(-0.01, 0.01))
        self.w2 = float(self.np_random.uniform(-0.01, 0.01))

        if not self.manual_mode:
            self.target_x = 0.0

        if self.step_count > 0:
            self.total_episodes += 1
            self.recent_rewards.append(self.episode_reward)
            if len(self.recent_rewards) > 10:
                self.recent_rewards.pop(0)
            self.avg_reward = float(np.mean(self.recent_rewards))

        self.step_count = 0
        self.episode_reward = 0.0
        self.tip_history.clear()

        if self.render_mode == "human" and self.screen is not None:
            self.render()

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

            # Prevent numerical anomalies
            state[1] = np.clip(state[1], -12.0, 12.0)
            state[3] = np.clip(state[3], -30.0, 30.0)
            state[5] = np.clip(state[5], -30.0, 30.0)

        self.cart_x, self.cart_v, self.t1, self.w1, self.t2, self.w2 = state
        self.t1 = (self.t1 + math.pi) % (2.0 * math.pi) - math.pi
        self.t2 = (self.t2 + math.pi) % (2.0 * math.pi) - math.pi

    def _handle_events(self):
        """Processes keyboard, mouse, and window events with instant responsiveness."""
        mouse_pos = pygame.mouse.get_pos()
        scale = 150.0
        center_x = WINDOW_WIDTH // 2
        track_y = 360
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
                elif event.key == pygame.K_t:
                    self.show_trail = not self.show_trail
                elif event.key == pygame.K_SPACE:
                    # Physical kick to challenge the balancing controller
                    kick = float(self.np_random.uniform(-1.0, 1.0))
                    self.w1 += kick * 3.5
                    self.w2 += -kick * 3.5
                    self.cart_v += kick * 1.5
                elif event.key == pygame.K_UP:
                    self.speed_idx = min(len(SPEED_PRESETS) - 1, self.speed_idx + 1)
                    self.render_fps = SPEED_PRESETS[self.speed_idx]
                elif event.key == pygame.K_DOWN:
                    self.speed_idx = max(0, self.speed_idx - 1)
                    self.render_fps = SPEED_PRESETS[self.speed_idx]

            elif event.type == pygame.MOUSEMOTION:
                # If in manual mode, moving the mouse smoothly updates the cart target position
                if self.manual_mode and 100 <= event.pos[1] <= 550:
                    target = (event.pos[0] - center_x) / scale
                    self.target_x = float(np.clip(target, -TRACK_LIMIT + 0.25, TRACK_LIMIT - 0.25))

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                # Header Buttons
                if 640 <= mx <= 770 and 24 <= my <= 58:
                    self.manual_mode = not self.manual_mode
                    if self.manual_mode:
                        self.target_x = self.cart_x
                    else:
                        self.target_x = 0.0
                    continue
                if 780 <= mx <= 870 and 24 <= my <= 58:
                    self.paused = not self.paused
                    continue
                if 880 <= mx <= 964 and 24 <= my <= 58:
                    self.reset()
                    continue

                # Speed badge click on left panel
                if 140 <= mx <= 250 and 90 <= my <= 115:
                    self.speed_idx = (self.speed_idx + 1) % len(SPEED_PRESETS)
                    self.render_fps = SPEED_PRESETS[self.speed_idx]
                    continue

                # Mouse click on track immediately sets target and activates manual mode
                dist_tip = math.hypot(mx - x2_px, my - y2_px)
                if dist_tip < 35:
                    self.dragging = "tip"
                else:
                    self.manual_mode = True
                    target = (mx - center_x) / scale
                    self.target_x = float(np.clip(target, -TRACK_LIMIT + 0.25, TRACK_LIMIT - 0.25))

            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self.dragging = None

        # Interactive Dragging
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
        steer_speed = 3.0 * self.dt  # Crisp, responsive steering speed
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            self.manual_mode = True
            self.target_x = float(np.clip(self.target_x - steer_speed, -TRACK_LIMIT + 0.25, TRACK_LIMIT - 0.25))
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.manual_mode = True
            self.target_x = float(np.clip(self.target_x + steer_speed, -TRACK_LIMIT + 0.25, TRACK_LIMIT - 0.25))

    def step(self, action):
        """
        Executes the continuous action generated exclusively by the RL policy network.
        """
        if self.should_close:
            return self._get_obs(), 0.0, True, True, {}

        # 1. Process Pygame events
        if self.render_mode == "human":
            if self.screen is None:
                self.render()
            self._handle_events()

        # 2. Pause loop
        while self.paused and not self.should_close:
            self._handle_events()
            self.render()

        if self.should_close:
            return self._get_obs(), 0.0, True, True, {}

        # 3. Apply the RL neural network action directly to the physical system
        applied_action = float(np.clip(action[0], -1.0, 1.0))
        force = applied_action * MAX_FORCE

        # 4. Integrate physics with RK4
        self._rk4_integrate(force)
        self.step_count += 1

        # 5. Reward function: alive bonus + upright reward - tracking error penalty
        tip_y = L1 * math.cos(self.t1) + L2 * math.cos(self.t2)
        total_len = L1 + L2
        dist_to_target = abs(self.cart_x - self.target_x)

        reward = 8.0 + 4.0 * (tip_y / total_len) - 1.8 * (dist_to_target ** 2) - 0.002 * (self.w1 ** 2 + self.w2 ** 2)

        # 6. Fall detection: instant reset when beyond recovery (never spin around at bottom)
        is_fallen = bool(abs(self.t1) > 0.45 or abs(self.t2) > 0.5 or tip_y < 0.65)
        is_off_track = bool(abs(self.cart_x) > TRACK_LIMIT)

        if is_fallen or is_off_track:
            reward -= 20.0
            terminated = True
        else:
            terminated = False

        truncated = bool(self.step_count >= self.max_episode_steps)
        self.episode_reward += reward

        # Record tip height for sparkline
        self.height_history.append(tip_y)
        if len(self.height_history) > 160:
            self.height_history.pop(0)

        # 7. Render frame
        if self.render_mode == "human":
            if self.render_fps <= 240 or (self.step_count % 4 == 0):
                self.render()

        return self._get_obs(), float(reward), terminated, truncated, {}

    def render(self, training_steps=0):
        if self.screen is None:
            pygame.init()
            pygame.font.init()
            self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
            pygame.display.set_caption("Double Pendulum RL | Pure Neural Network Control & Accelerated Training")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 20)
            self.font_bold = pygame.font.Font(None, 22)
            self.title_font = pygame.font.Font(None, 28)

        mouse_pos = pygame.mouse.get_pos()

        # 1. Background Grid
        self.screen.fill(BG_COLOR)
        for x in range(0, WINDOW_WIDTH, 40):
            pygame.draw.line(self.screen, GRID_COLOR, (x, 0), (x, WINDOW_HEIGHT), 1)
        for y in range(0, WINDOW_HEIGHT, 40):
            pygame.draw.line(self.screen, GRID_COLOR, (0, y), (WINDOW_WIDTH, y), 1)

        scale = 150.0
        center_x = WINDOW_WIDTH // 2
        track_y = 360
        track_half_width = int(TRACK_LIMIT * scale)

        # 2. Track Rail & Bumpers
        rail_rect = (center_x - track_half_width - 15, track_y - 4, 2 * track_half_width + 30, 8)
        pygame.draw.rect(self.screen, COLOR_TRACK, rail_rect, border_radius=4)
        pygame.draw.line(self.screen, TEXT_MUTED, (center_x, track_y - 12), (center_x, track_y + 12), 2)
        for m in [-2, -1, 1, 2]:
            tx = center_x + int(m * scale)
            pygame.draw.line(self.screen, PANEL_BORDER, (tx, track_y - 7), (tx, track_y + 7), 1)

        # End Bumpers
        left_bumper = (center_x - track_half_width - 16, track_y - 20, 12, 40)
        right_bumper = (center_x + track_half_width + 4, track_y - 20, 12, 40)
        pygame.draw.rect(self.screen, COLOR_RED, left_bumper, border_radius=3)
        pygame.draw.rect(self.screen, COLOR_RED, right_bumper, border_radius=3)

        # Glowing Target Reticle on Track (Visual target indicator for the RL agent)
        target_px = center_x + int(self.target_x * scale)
        reticle_color = COLOR_AMBER if self.manual_mode else COLOR_CYAN
        reticle_surf = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        pygame.draw.line(reticle_surf, (*reticle_color, 120), (target_px, track_y - 32), (target_px, track_y + 16), 1)
        pts = [(target_px, track_y - 16), (target_px + 8, track_y - 6), (target_px, track_y + 4), (target_px - 8, track_y - 6)]
        pygame.draw.polygon(reticle_surf, (*reticle_color, 200), pts)
        self.screen.blit(reticle_surf, (0, 0))

        # 3. Cart & Joint Calculations
        cart_pixel_x = center_x + int(self.cart_x * scale)
        x1_px = cart_pixel_x + int(L1 * scale * math.sin(self.t1))
        y1_px = track_y - int(L1 * scale * math.cos(self.t1))
        x2_px = x1_px + int(L2 * scale * math.sin(self.t2))
        y2_px = y1_px - int(L2 * scale * math.cos(self.t2))

        # Laser upright reference line
        if abs(self.t1) < 0.25 and abs(self.t2) < 0.25:
            guide_color = (0, 229, 255, 60)
        else:
            guide_color = (100, 115, 140, 40)
        laser_surf = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        for dy in range(int(track_y - (L1 + L2) * scale - 20), track_y, 8):
            pygame.draw.line(laser_surf, guide_color, (cart_pixel_x, dy), (cart_pixel_x, dy + 4), 1)
        self.screen.blit(laser_surf, (0, 0))

        # 4. Tip Ribbon Motion Trail
        if self.show_trail:
            self.tip_history.append((x2_px, y2_px))
            if len(self.tip_history) > 36:
                self.tip_history.pop(0)

            if len(self.tip_history) > 1:
                trail_surf = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
                for i in range(len(self.tip_history) - 1):
                    alpha = int(255 * (i / len(self.tip_history)) * 0.6)
                    pt1 = self.tip_history[i]
                    pt2 = self.tip_history[i + 1]
                    color = (*COLOR_PINK[:3], alpha)
                    width = max(1, int(4 * (i / len(self.tip_history))))
                    pygame.draw.line(trail_surf, color, pt1, pt2, width)
                self.screen.blit(trail_surf, (0, 0))

        # 5. Pendulum Rods & Joints
        pygame.draw.line(self.screen, (20, 25, 35), (cart_pixel_x, track_y), (x1_px, y1_px), 8)
        pygame.draw.line(self.screen, COLOR_CYAN, (cart_pixel_x, track_y), (x1_px, y1_px), 5)
        pygame.draw.line(self.screen, (20, 25, 35), (x1_px, y1_px), (x2_px, y2_px), 7)
        pygame.draw.line(self.screen, COLOR_PINK, (x1_px, y1_px), (x2_px, y2_px), 4)

        # Elbow Joint
        pygame.draw.circle(self.screen, (30, 40, 55), (x1_px, y1_px), 8)
        pygame.draw.circle(self.screen, COLOR_CYAN, (x1_px, y1_px), 5)
        pygame.draw.circle(self.screen, TEXT_WHITE, (x1_px, y1_px), 2)

        # Tip Bob with glow
        tip_surf = pygame.Surface((32, 32), pygame.SRCALPHA)
        pygame.draw.circle(tip_surf, (*COLOR_PINK, 70), (16, 16), 14)
        pygame.draw.circle(tip_surf, COLOR_PINK, (16, 16), 8)
        pygame.draw.circle(tip_surf, TEXT_WHITE, (14, 14), 3)
        self.screen.blit(tip_surf, (x2_px - 16, y2_px - 16))

        # 6. Cart Rendering
        cart_w, cart_h = 72, 36
        cart_rect = (cart_pixel_x - cart_w // 2, track_y - cart_h + 8, cart_w, cart_h)
        pygame.draw.rect(self.screen, (10, 12, 18), (cart_rect[0], track_y + 4, cart_w, 6), border_radius=3)
        pygame.draw.rect(self.screen, (38, 48, 66), cart_rect, border_radius=6)
        pygame.draw.rect(self.screen, (60, 75, 100), cart_rect, width=1, border_radius=6)

        # Wheels
        wheel_y = track_y + 4
        for wx in [cart_pixel_x - 22, cart_pixel_x + 22]:
            pygame.draw.circle(self.screen, (20, 24, 32), (wx, wheel_y), 7)
            pygame.draw.circle(self.screen, (90, 105, 130), (wx, wheel_y), 5)
            pygame.draw.circle(self.screen, (200, 215, 235), (wx, wheel_y), 2)

        # Pivot Bearing
        pygame.draw.circle(self.screen, (25, 32, 45), (cart_pixel_x, track_y), 9)
        pygame.draw.circle(self.screen, (100, 120, 150), (cart_pixel_x, track_y), 6)
        pygame.draw.circle(self.screen, TEXT_WHITE, (cart_pixel_x, track_y), 2)

        # Status Strip on Cart
        status_color = COLOR_AMBER if self.manual_mode else COLOR_GREEN
        strip_rect = (cart_pixel_x - 24, track_y - cart_h + 14, 48, 4)
        pygame.draw.rect(self.screen, status_color, strip_rect, border_radius=2)

        # 7. Header Bar & Controls
        hdr_rect = (20, 16, WINDOW_WIDTH - 40, 52)
        pygame.draw.rect(self.screen, PANEL_BG, hdr_rect, border_radius=8)
        pygame.draw.rect(self.screen, PANEL_BORDER, hdr_rect, width=1, border_radius=8)

        t_surf = self.title_font.render("DOUBLE PENDULUM RL", True, TEXT_WHITE)
        self.screen.blit(t_surf, (38, 28))
        mode_subtitle = "Manual Steering (RL Policy Controlled)" if self.manual_mode else "Autonomous RL Balance Autopilot"
        sub_surf = self.font.render(mode_subtitle, True, COLOR_AMBER if self.manual_mode else COLOR_GREEN)
        self.screen.blit(sub_surf, (280, 32))

        # Mode Button
        mode_btn = pygame.Rect(640, 24, 130, 34)
        hover_mode = mode_btn.collidepoint(mouse_pos)
        mode_bg = (45, 65, 55) if not self.manual_mode else (70, 55, 30)
        mode_border = COLOR_GREEN if not self.manual_mode else COLOR_AMBER
        pygame.draw.rect(self.screen, mode_bg if not hover_mode else tuple(min(255, c + 20) for c in mode_bg), mode_btn, border_radius=6)
        pygame.draw.rect(self.screen, mode_border, mode_btn, width=1, border_radius=6)
        mode_txt = "AI AUTOPILOT" if not self.manual_mode else "MANUAL STEER"
        m_surf = self.font_bold.render(mode_txt, True, mode_border)
        self.screen.blit(m_surf, (mode_btn.centerx - m_surf.get_width() // 2, mode_btn.centery - m_surf.get_height() // 2))

        # Pause Button
        pause_btn = pygame.Rect(780, 24, 90, 34)
        hover_pause = pause_btn.collidepoint(mouse_pos)
        pause_txt = "RESUME" if self.paused else "PAUSE [P]"
        pause_border = COLOR_AMBER if self.paused else PANEL_BORDER
        pause_bg = (38, 48, 68) if hover_pause else (28, 36, 52)
        pygame.draw.rect(self.screen, pause_bg, pause_btn, border_radius=6)
        pygame.draw.rect(self.screen, pause_border, pause_btn, width=1, border_radius=6)
        p_surf = self.font_bold.render(pause_txt, True, COLOR_AMBER if self.paused else TEXT_WHITE)
        self.screen.blit(p_surf, (pause_btn.centerx - p_surf.get_width() // 2, pause_btn.centery - p_surf.get_height() // 2))

        # Reset Button
        reset_btn = pygame.Rect(880, 24, 84, 34)
        hover_reset = reset_btn.collidepoint(mouse_pos)
        reset_bg = (38, 48, 68) if hover_reset else (28, 36, 52)
        pygame.draw.rect(self.screen, reset_bg, reset_btn, border_radius=6)
        pygame.draw.rect(self.screen, PANEL_BORDER, reset_btn, width=1, border_radius=6)
        r_surf = self.font_bold.render("RESET [R]", True, TEXT_WHITE)
        self.screen.blit(r_surf, (reset_btn.centerx - r_surf.get_width() // 2, reset_btn.centery - r_surf.get_height() // 2))

        # 8. Left Telemetry Dashboard Panel
        card_w, card_h = 240, 270
        card_rect = (20, 80, card_w, card_h)
        pygame.draw.rect(self.screen, PANEL_BG, card_rect, border_radius=8)
        pygame.draw.rect(self.screen, PANEL_BORDER, card_rect, width=1, border_radius=8)

        card_title = self.font_bold.render("TELEMETRY", True, TEXT_WHITE)
        self.screen.blit(card_title, (35, 92))
        speed_badge = self.font.render(SPEED_LABELS[self.speed_idx], True, COLOR_CYAN)
        self.screen.blit(speed_badge, (card_rect[0] + card_w - speed_badge.get_width() - 15, 94))

        tip_y = L1 * math.cos(self.t1) + L2 * math.cos(self.t2)
        total_len = L1 + L2
        angle1_deg = math.degrees(self.t1)
        angle2_deg = math.degrees(self.t2)

        telemetry_rows = [
            ("RL Trained Steps", f"{training_steps:,}", COLOR_GREEN),
            ("Control Source", "100% RL Neural Net", COLOR_CYAN),
            ("Cart Position", f"{self.cart_x:+.2f} m", TEXT_WHITE),
            ("Target Position", f"{self.target_x:+.2f} m", COLOR_AMBER if self.manual_mode else TEXT_MUTED),
            ("Pole 1 Angle", f"{angle1_deg:+.1f}°", COLOR_CYAN if abs(angle1_deg) < 12 else TEXT_WHITE),
            ("Pole 2 Angle", f"{angle2_deg:+.1f}°", COLOR_PINK if abs(angle2_deg) < 12 else TEXT_WHITE),
            ("Tip Height", f"{tip_y:+.2f} / {total_len:.2f} m", COLOR_GREEN if tip_y > 0.85 else TEXT_WHITE),
            ("Mean Reward", f"{self.avg_reward:+.1f}", COLOR_CYAN),
        ]

        row_y = 118
        for label, val, val_color in telemetry_rows:
            lbl_surf = self.font.render(label, True, TEXT_MUTED)
            val_surf = self.font.render(val, True, val_color)
            self.screen.blit(lbl_surf, (35, row_y))
            self.screen.blit(val_surf, (card_rect[0] + card_w - val_surf.get_width() - 15, row_y))
            row_y += 16

        # Position meter
        meter_y = row_y + 8
        meter_w = card_w - 30
        pygame.draw.rect(self.screen, (30, 38, 55), (35, meter_y, meter_w, 6), border_radius=3)
        cart_ratio = (self.cart_x + TRACK_LIMIT) / (2.0 * TRACK_LIMIT)
        pin_x = int(35 + cart_ratio * meter_w)
        pygame.draw.circle(self.screen, COLOR_GREEN if abs(self.cart_x) < 1.8 else COLOR_RED, (pin_x, meter_y + 3), 4)

        # 9. Right Performance Sparkline Graph
        graph_w, graph_h = 240, 160
        graph_rect = (WINDOW_WIDTH - graph_w - 20, 80, graph_w, graph_h)
        pygame.draw.rect(self.screen, PANEL_BG, graph_rect, border_radius=8)
        pygame.draw.rect(self.screen, PANEL_BORDER, graph_rect, width=1, border_radius=8)

        gt_surf = self.font_bold.render("TIP HEIGHT HISTORY", True, TEXT_WHITE)
        self.screen.blit(gt_surf, (graph_rect[0] + 15, 92))

        plot_x = graph_rect[0] + 15
        plot_y = 120
        plot_w = graph_w - 30
        plot_h = graph_h - 50
        pygame.draw.rect(self.screen, (18, 23, 34), (plot_x, plot_y, plot_w, plot_h), border_radius=4)
        pygame.draw.rect(self.screen, PANEL_BORDER, (plot_x, plot_y, plot_w, plot_h), width=1, border_radius=4)

        target_py = plot_y + int(plot_h * 0.15)
        pygame.draw.line(self.screen, (30, 140, 85), (plot_x, target_py), (plot_x + plot_w, target_py), 1)

        if len(self.height_history) > 2:
            pts = []
            for i, h in enumerate(self.height_history):
                px = plot_x + int((i / 160.0) * plot_w)
                norm = (h + total_len) / (2.0 * total_len)
                py = plot_y + plot_h - int(norm * (plot_h - 8)) - 4
                py = int(np.clip(py, plot_y + 2, plot_y + plot_h - 2))
                pts.append((px, py))
            if len(pts) >= 2:
                pygame.draw.lines(self.screen, COLOR_GREEN, False, pts, 2)

        # 10. Bottom Control Hotkey Bar
        bot_rect = (20, WINDOW_HEIGHT - 54, WINDOW_WIDTH - 40, 40)
        pygame.draw.rect(self.screen, PANEL_BG, bot_rect, border_radius=6)
        pygame.draw.rect(self.screen, PANEL_BORDER, bot_rect, width=1, border_radius=6)

        ctrl_chips = [
            ("Mouse Move/Drag", "Steer Cart Target"),
            ("A / D or ← / →", "Arrow Steer"),
            ("M", "AI / Manual"),
            ("Space", "Disturbance Kick"),
            ("R", "Reset"),
            ("P", "Pause"),
            ("▲ / ▼", "Speed"),
            ("T", "Trail"),
        ]

        chip_x = 35
        for key_badge, desc in ctrl_chips:
            k_surf = self.font_bold.render(f"[{key_badge}]", True, COLOR_CYAN)
            d_surf = self.font.render(desc, True, TEXT_MUTED)
            self.screen.blit(k_surf, (chip_x, WINDOW_HEIGHT - 43))
            self.screen.blit(d_surf, (chip_x + k_surf.get_width() + 4, WINDOW_HEIGHT - 43))
            chip_x += k_surf.get_width() + d_surf.get_width() + 14
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
    Thread-safe, direct PyTorch inference from the PPO policy network.
    """
    with torch.no_grad():
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        dist = model.policy.get_distribution(obs_t)
        action = dist.mode().cpu().numpy()[0]
        return action


def create_training_env():
    """Factory for vectorized fast headless background training environments."""
    class FastTrainEnv(gym.Env):
        def __init__(self):
            super().__init__()
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
            self.dt = 0.02
            self.sub_dt = 0.005
            self.n_substeps = 4
            self.cart_x, self.cart_v = 0.0, 0.0
            self.t1, self.w1 = 0.0, 0.0
            self.t2, self.w2 = 0.0, 0.0
            self.target_x = 0.0
            self.steps = 0

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.cart_x = float(self.np_random.uniform(-0.04, 0.04))
            self.cart_v = 0.0
            self.t1 = float(self.np_random.uniform(-0.04, 0.04))
            self.t2 = float(self.np_random.uniform(-0.04, 0.04))
            self.w1 = 0.0
            self.w2 = 0.0
            self.target_x = float(self.np_random.uniform(-1.5, 1.5))
            self.steps = 0
            return self._obs(), {}

        def _obs(self):
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

        def step(self, action):
            force = float(np.clip(action[0], -1.0, 1.0)) * MAX_FORCE
            # Fast RK4 integration
            state = np.array([self.cart_x, self.cart_v, self.t1, self.w1, self.t2, self.w2], dtype=np.float64)
            for _ in range(self.n_substeps):
                def deriv(s):
                    x, xd, t1, w1, t2, w2 = s
                    c1, s1 = math.cos(t1), math.sin(t1)
                    c2, s2 = math.cos(t2), math.sin(t2)
                    c12, s12 = math.cos(t1 - t2), math.sin(t1 - t2)
                    m12 = M1 + M2
                    A = np.array([
                        [CART_MASS + m12, m12 * L1 * c1, M2 * L2 * c2],
                        [m12 * L1 * c1, m12 * (L1 ** 2), M2 * L1 * L2 * c12],
                        [M2 * L2 * c2, M2 * L1 * L2 * c12, M2 * (L2 ** 2)]
                    ], dtype=np.float64)
                    B = np.array([
                        force - FRICTION_CART * xd + m12 * L1 * s1 * (w1 ** 2) + M2 * L2 * s2 * (w2 ** 2),
                        m12 * G * L1 * s1 - M2 * L1 * L2 * s12 * (w2 ** 2) - DAMPING_1 * w1,
                        M2 * G * L2 * s2 + M2 * L1 * L2 * s12 * (w1 ** 2) - DAMPING_2 * w2
                    ], dtype=np.float64)
                    acc = np.linalg.solve(A, B)
                    return np.array([xd, acc[0], w1, acc[1], w2, acc[2]], dtype=np.float64)

                k1 = deriv(state)
                k2 = deriv(state + 0.5 * self.sub_dt * k1)
                k3 = deriv(state + 0.5 * self.sub_dt * k2)
                k4 = deriv(state + self.sub_dt * k3)
                state += (self.sub_dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

            self.cart_x, self.cart_v, self.t1, self.w1, self.t2, self.w2 = state
            self.t1 = (self.t1 + math.pi) % (2.0 * math.pi) - math.pi
            self.t2 = (self.t2 + math.pi) % (2.0 * math.pi) - math.pi
            self.steps += 1

            tip_y = L1 * math.cos(self.t1) + L2 * math.cos(self.t2)
            dist_to_target = abs(self.cart_x - self.target_x)

            reward = 8.0 + 4.0 * (tip_y / (L1 + L2)) - 1.8 * (dist_to_target ** 2) - 0.002 * (self.w1 ** 2 + self.w2 ** 2)
            is_fallen = bool(abs(self.t1) > 0.45 or abs(self.t2) > 0.5 or tip_y < 0.65)
            is_off_track = bool(abs(self.cart_x) > TRACK_LIMIT)
            terminated = is_fallen or is_off_track
            truncated = bool(self.steps >= 1000)

            return self._obs(), reward, terminated, truncated, {}

    return FastTrainEnv()


def pretrain_neural_policy(model, num_samples=10000, epochs=150):
    """
    Initializes the PPO policy network with dynamic target-tracking capabilities
    so it begins fully responsive and balanced from the very first frame.
    """
    print("[AI Warm-up] Synthesizing balance and steering demonstrations...")
    K = np.array([21.392, 36.663, -528.685, -10.788, 802.479, 121.801])
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

        u = -float(np.dot(K, s_err))
        action = np.clip(u / MAX_FORCE, -1.0, 1.0)
        X_data.append(obs)
        Y_data.append([action])

    X_tensor = torch.tensor(np.array(X_data), dtype=torch.float32)
    Y_tensor = torch.tensor(np.array(Y_data), dtype=torch.float32)

    optimizer = torch.optim.Adam(model.policy.parameters(), lr=3e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        dist = model.policy.get_distribution(X_tensor)
        pred = dist.mode()
        loss = loss_fn(pred, Y_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"[AI Warm-up] Completed! Policy initial MSE: {loss.item():.4f}. Neural Network ready!")


if __name__ == "__main__":
    print("=================================================================")
    print("     Interactive Double Pendulum Reinforcement Learning         ")
    print("=================================================================")
    print("Initializing environment and graphics engine...")
    ui_env = InteractiveDoublePendulumEnv(render_mode="human")

    # Vectorized headless environments for parallel PPO training (~4,000 steps/sec)
    num_vec_envs = 4
    vec_train_env = DummyVecEnv([create_training_env for _ in range(num_vec_envs)])

    print("Building PPO Reinforcement Learning Policy...")
    model = PPO(
        "MlpPolicy",
        vec_train_env,
        learning_rate=5e-4,
        n_steps=256,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        verbose=0
    )

    # 1. Warm-start policy so it is immediately responsive to mouse and arrows
    pretrain_neural_policy(model, num_samples=10000, epochs=150)

    # 2. Launch high-speed background training worker (~4,000 steps/sec)
    stop_training = threading.Event()

    def background_training_loop():
        """Continuously trains the PPO policy at maximum throughput in the background."""
        while not stop_training.is_set():
            try:
                model.learn(total_timesteps=1024, reset_num_timesteps=False)
            except Exception:
                break

    train_thread = threading.Thread(target=background_training_loop, daemon=True)
    train_thread.start()
    print("[AI Training] Accelerated background training engine active (~4,000 steps/sec)!")

    print("Launching Interactive Dashboard!")
    print("Features:")
    print("  - [100% Pure RL]: Manual & autonomous control are exclusively executed by the RL neural net.")
    print("  - [Hyper-Responsive Steer]: Mouse movement and Arrow keys smoothly update target position.")
    print("  - [High-Speed PPO Training]: Actively trains thousands of RL steps per second in background.")
    print("  - [Instant Reset on Fall]: Pendulum stays upright; resets immediately if knocked down.")
    print("=================================================================")

    obs, _ = ui_env.reset()

    try:
        while not ui_env.should_close:
            # Predict action exclusively using the trained RL neural network
            action = predict_action(model, obs)
            obs, reward, terminated, truncated, _ = ui_env.step(action)

            # Pass live training step counter to the UI dashboard
            if ui_env.screen is not None:
                ui_env.render(training_steps=model.num_timesteps)

            if terminated or truncated:
                obs, _ = ui_env.reset()

    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
    finally:
        stop_training.set()
        train_thread.join(timeout=1.0)
        ui_env.close()
        vec_train_env.close()
        print("Environment closed cleanly. Goodbye!")
