"""MCR-UAV Live Research Dashboard & Telemetry Monitor.

This module provides a real-time Tkinter and Matplotlib GUI dashboard for
monitoring the PyBullet UAV simulation, Meta-Supervisor reconfiguration vector,
Transformer Context Encoder latent space, disturbances, training metrics, and live plots.
"""

from __future__ import annotations

import csv
import math
import os
import queue
import threading
import time
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import ttk


class MCRUAVLiveMonitor:
    """Real-time research dashboard for MCR-UAV Meta-RL experiment monitoring."""

    def __init__(
        self,
        telemetry_queue: queue.Queue,
        control_events: Dict[str, threading.Event],
        save_dir: str = "meta_rl_results/gui",
        master_seed: int = 42,
    ) -> None:
        self.telemetry_queue = telemetry_queue
        self.control_events = control_events
        self.save_dir = save_dir
        self.current_seed = master_seed
        os.makedirs(self.save_dir, exist_ok=True)

        # Telemetry file
        self.telemetry_csv_path = os.path.join(self.save_dir, f"telemetry_seed_{self.current_seed}.csv")
        self.csv_file = open(self.telemetry_csv_path, "a", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.csv_file)
        if os.path.getsize(self.telemetry_csv_path) == 0:
            self.csv_writer.writerow([
                "timestamp", "seed", "iteration", "episode", "simulation_time",
                "pos_x", "pos_y", "pos_z", "target_x", "target_y", "target_z",
                "vel_x", "vel_y", "vel_z", "tracking_error", "reward",
                "disturbance", "motor_loe", "lambda_RL", "alpha_Q", "alpha_R",
                "alpha_P", "alpha_I", "alpha_D", "H", "success"
            ])
            self.csv_file.flush()

        # Telemetry history for plots (rolling window)
        self.history_len = 300
        self.hist_time = []
        self.hist_error = []
        self.hist_reward = []
        self.hist_cum_reward = []
        self.hist_pos_z = []
        self.hist_target_z = []
        self.hist_lambda_rl = []
        self.hist_alpha_q = []
        self.hist_alpha_r = []
        self.hist_alpha_p = []
        self.hist_alpha_i = []
        self.hist_alpha_d = []
        self.hist_horizon = []
        self.hist_vel_norm = []
        self.hist_rpm_mean = []

        # Current state
        self.last_packet = {}
        self.flight_trail = []
        self.is_running = True
        self.paused = False

        self.after_id = None

        # Build UI
        self._build_ui()

    def _build_ui(self) -> None:
        self.root = tk.Tk()
        self.root.title("MCR-UAV LIVE MONITOR — Meta-Contextual Reconfiguration Dashboard")
        self.root.geometry("1440x960")
        self.root.configure(bg="#121318")

        # Configure dark styling
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure(".", background="#121318", foreground="#e0e0e0", font=("Consolas", 10))
        self.style.configure("TLabel", background="#121318", foreground="#e0e0e0")
        self.style.configure("Header.TLabel", font=("Consolas", 14, "bold"), foreground="#00ffff")
        self.style.configure("SubHeader.TLabel", font=("Consolas", 11, "bold"), foreground="#00d2ff")
        self.style.configure("Badge.TLabel", font=("Consolas", 10, "bold"), background="#1e2230", foreground="#00ff88")
        self.style.configure("Alert.TLabel", font=("Consolas", 10, "bold"), foreground="#ffaa00")

        # Main scrollable or grid container
        main_frame = tk.Frame(self.root, bg="#121318")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

        # ---------------- TOP HEADER BAR ----------------
        header_bar = tk.Frame(main_frame, bg="#1a1d26", height=40, relief=tk.RAISED, bd=1)
        header_bar.pack(fill=tk.X, pady=(0, 6))

        title_lbl = tk.Label(
            header_bar,
            text="⚡ MCR-UAV META-RL REAL-TIME PRODUCTION MONITOR",
            font=("Consolas", 13, "bold"),
            bg="#1a1d26",
            fg="#00ffff"
        )
        title_lbl.pack(side=tk.LEFT, padx=12, pady=6)

        self.status_badge = tk.Label(
            header_bar,
            text="STATUS: INITIALIZING",
            font=("Consolas", 10, "bold"),
            bg="#2a3045",
            fg="#00ff88",
            padx=10,
            pady=3,
            relief=tk.RIDGE
        )
        self.status_badge.pack(side=tk.RIGHT, padx=12, pady=6)

        self.time_lbl = tk.Label(
            header_bar,
            text="Sim Time: 0.00s | Step: 0",
            font=("Consolas", 10),
            bg="#1a1d26",
            fg="#aaaaaa"
        )
        self.time_lbl.pack(side=tk.RIGHT, padx=15, pady=6)

        # ---------------- TOP ROW PANELS (Sim View + Supervisor + Context) ----------------
        top_row = tk.Frame(main_frame, bg="#121318")
        top_row.pack(fill=tk.X, expand=False, pady=(0, 6))

        # Panel 1: 3D Flight Viewport (Left)
        sim_box = tk.LabelFrame(
            top_row,
            text=" PYBULLET UAV 3D SIMULATION VIEWPORT ",
            font=("Consolas", 10, "bold"),
            bg="#181a24",
            fg="#00d2ff",
            bd=1,
            relief=tk.GROOVE
        )
        sim_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))

        self.sim_canvas = tk.Canvas(sim_box, width=420, height=250, bg="#0d0e14", highlightthickness=1, highlightbackground="#252a3a")
        self.sim_canvas.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # Panel 2: Meta-RL Supervisor Reconfiguration Panel (Middle)
        super_box = tk.LabelFrame(
            top_row,
            text=" MCR-UAV META-SUPERVISOR RECONFIGURATION ",
            font=("Consolas", 10, "bold"),
            bg="#181a24",
            fg="#00ff88",
            bd=1,
            relief=tk.GROOVE
        )
        super_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)

        super_inner = tk.Frame(super_box, bg="#181a24")
        super_inner.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

        self.param_labels: Dict[str, tk.Label] = {}
        params_info = [
            ("λ_RL (Blending)", "0.0700", "#00ffff"),
            ("α_Q (State Cost)", "1.0000", "#00ff88"),
            ("α_R (Input Cost)", "1.0000", "#00ff88"),
            ("α_P (PID Prop)", "1.0000", "#ffaa00"),
            ("α_I (PID Integ)", "1.0000", "#ffaa00"),
            ("α_D (PID Deriv)", "1.0000", "#ffaa00"),
            ("MPC Horizon H", "20", "#ff00dd"),
            ("Controller Mode", "MCR-UAV", "#00ffff"),
        ]
        for i, (name, val, col) in enumerate(params_info):
            r = i // 2
            c = (i % 2) * 2
            lbl_title = tk.Label(super_inner, text=f"{name}:", font=("Consolas", 9), bg="#181a24", fg="#aaaaaa", anchor="w")
            lbl_title.grid(row=r, column=c, sticky="w", padx=4, pady=3)

            lbl_val = tk.Label(super_inner, text=val, font=("Consolas", 10, "bold"), bg="#222634", fg=col, width=9, relief=tk.SUNKEN)
            lbl_val.grid(row=r, column=c+1, sticky="w", padx=4, pady=3)
            self.param_labels[name] = lbl_val

        # Panel 3: Context Encoder Latent Space Panel (Right)
        ctx_box = tk.LabelFrame(
            top_row,
            text=" TRANSFORMER CONTEXT ENCODER (H_t=20x52 → z_t∈R^16) ",
            font=("Consolas", 10, "bold"),
            bg="#181a24",
            fg="#ffaa00",
            bd=1,
            relief=tk.GROOVE
        )
        ctx_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 0))

        self.ctx_canvas = tk.Canvas(ctx_box, width=380, height=250, bg="#0d0e14", highlightthickness=1, highlightbackground="#252a3a")
        self.ctx_canvas.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # ---------------- MIDDLE ROW PANELS (Training Status & Disturbance) ----------------
        mid_row = tk.Frame(main_frame, bg="#121318")
        mid_row.pack(fill=tk.X, expand=False, pady=(0, 6))

        # Training Status Box (Left)
        train_box = tk.LabelFrame(
            mid_row,
            text=" TRAINING STATUS & PERFORMANCE METRICS ",
            font=("Consolas", 10, "bold"),
            bg="#181a24",
            fg="#00d2ff",
            bd=1,
            relief=tk.GROOVE
        )
        train_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))

        train_inner = tk.Frame(train_box, bg="#181a24")
        train_inner.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

        self.train_labels: Dict[str, tk.Label] = {}
        train_fields = [
            ("Seed", f"{self.current_seed}"),
            ("Iteration", "1 / 50"),
            ("Episode", "1"),
            ("Current Return", "0.00"),
            ("Mean Return", "0.00"),
            ("Step Reward", "0.00"),
            ("Grad Norm", "0.0000"),
            ("Param Delta", "0.0000"),
            ("Val Success", "0.0%"),
            ("Val RMSE", "0.00 m"),
            ("Val Energy", "0.0"),
            ("Tracking Error", "0.00 m"),
        ]
        for idx, (lbl, val) in enumerate(train_fields):
            r = idx // 4
            c = (idx % 4) * 2
            tk.Label(train_inner, text=f"{lbl}:", font=("Consolas", 8), bg="#181a24", fg="#888888").grid(row=r, column=c, sticky="w", padx=2, pady=2)
            v_lbl = tk.Label(train_inner, text=val, font=("Consolas", 9, "bold"), bg="#181a24", fg="#ffffff", anchor="w")
            v_lbl.grid(row=r, column=c+1, sticky="w", padx=2, pady=2)
            self.train_labels[lbl] = v_lbl

        # Disturbance & Faults Box (Right)
        dist_box = tk.LabelFrame(
            mid_row,
            text=" ACTIVE DISTURBANCE & ACTUATOR DEGRADATION ",
            font=("Consolas", 10, "bold"),
            bg="#181a24",
            fg="#ff5555",
            bd=1,
            relief=tk.GROOVE
        )
        dist_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 0))

        dist_inner = tk.Frame(dist_box, bg="#181a24")
        dist_inner.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

        self.dist_labels: Dict[str, tk.Label] = {}
        dist_fields = [
            ("Disturbance Type", "Nominal"),
            ("Wind / Gust Mag", "0.00 m/s"),
            ("Impulse Force", "0.00 N"),
            ("Sensor Noise", "1.0x (Nominal)"),
            ("Motor LoE", "0.0%"),
            ("Degraded Rotors", "None"),
        ]
        for idx, (lbl, val) in enumerate(dist_fields):
            r = idx // 2
            c = (idx % 2) * 2
            tk.Label(dist_inner, text=f"{lbl}:", font=("Consolas", 8), bg="#181a24", fg="#888888").grid(row=r, column=c, sticky="w", padx=4, pady=2)
            v_lbl = tk.Label(dist_inner, text=val, font=("Consolas", 9, "bold"), bg="#181a24", fg="#ffaa00", anchor="w")
            v_lbl.grid(row=r, column=c+1, sticky="w", padx=4, pady=2)
            self.dist_labels[lbl] = v_lbl

        # ---------------- BOTTOM ROW (Live Matplotlib Plots & Control Bar) ----------------
        plot_box = tk.LabelFrame(
            main_frame,
            text=" LIVE PERFORMANCE & TELEMETRY PLOTS ",
            font=("Consolas", 10, "bold"),
            bg="#181a24",
            fg="#00d2ff",
            bd=1,
            relief=tk.GROOVE
        )
        plot_box.pack(fill=tk.BOTH, expand=True, pady=(0, 6))

        # Embedded Matplotlib Figure (2 rows x 3 cols = 6 plots)
        plt.style.use("dark_background")
        self.fig, self.axs = plt.subplots(2, 3, figsize=(12, 3.8), dpi=85)
        self.fig.patch.set_facecolor("#10121a")
        for ax in self.axs.flat:
            ax.set_facecolor("#0b0c12")
            ax.grid(True, linestyle=":", alpha=0.35, color="#404555")
            ax.tick_params(labelsize=7, colors="#999999")

        self.axs[0, 0].set_title("Tracking Error (m)", fontsize=8, color="#00d2ff")
        self.axs[0, 1].set_title("Altitude: UAV z vs Target z", fontsize=8, color="#00ff88")
        self.axs[0, 2].set_title("Reward & Cumulative Return", fontsize=8, color="#ffcc00")
        self.axs[1, 0].set_title("Reconfig: λ_RL, α_Q, α_R", fontsize=8, color="#ff00dd")
        self.axs[1, 1].set_title("PID Scalings (α_P, α_I, α_D) & H", fontsize=8, color="#00ffff")
        self.axs[1, 2].set_title("Velocity & Mean Motor RPM", fontsize=8, color="#ff7777")
        self.fig.tight_layout()

        self.plot_canvas = FigureCanvasTkAgg(self.fig, master=plot_box)
        self.plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # ---------------- CONTROL BUTTONS BAR ----------------
        ctrl_bar = tk.Frame(main_frame, bg="#181a24", height=42, relief=tk.RAISED, bd=1)
        ctrl_bar.pack(fill=tk.X)

        btn_font = ("Consolas", 9, "bold")
        self.btn_start = tk.Button(ctrl_bar, text="▶ START", bg="#1f6f3e", fg="#ffffff", font=btn_font, padx=12, pady=4, command=self._on_start)
        self.btn_start.pack(side=tk.LEFT, padx=6, pady=4)

        self.btn_pause = tk.Button(ctrl_bar, text="⏸ PAUSE", bg="#855a15", fg="#ffffff", font=btn_font, padx=12, pady=4, command=self._on_pause)
        self.btn_pause.pack(side=tk.LEFT, padx=6, pady=4)

        self.btn_resume = tk.Button(ctrl_bar, text="⏯ RESUME", bg="#1f587a", fg="#ffffff", font=btn_font, padx=12, pady=4, command=self._on_resume)
        self.btn_resume.pack(side=tk.LEFT, padx=6, pady=4)

        self.btn_stop = tk.Button(ctrl_bar, text="⏹ STOP", bg="#8a2020", fg="#ffffff", font=btn_font, padx=12, pady=4, command=self._on_stop)
        self.btn_stop.pack(side=tk.LEFT, padx=6, pady=4)

        self.btn_reset = tk.Button(ctrl_bar, text="🔄 RESET EPISODE", bg="#3a3e52", fg="#ffffff", font=btn_font, padx=10, pady=4, command=self._on_reset)
        self.btn_reset.pack(side=tk.LEFT, padx=6, pady=4)

        # Optional toggles
        self.btn_baseline = tk.Button(ctrl_bar, text="MODE: MCR-UAV", bg="#2a3045", fg="#00ffff", font=btn_font, padx=10, pady=4, command=self._toggle_mode)
        self.btn_baseline.pack(side=tk.RIGHT, padx=10, pady=4)

        # Start periodic GUI refresh timer (~30 FPS)
        self.root.after(33, self._poll_telemetry)

    def _on_start(self) -> None:
        if "start_event" in self.control_events:
            self.control_events["start_event"].set()
        self.status_badge.config(text="STATUS: RUNNING", fg="#00ff88", bg="#1b4d2e")

    def _on_pause(self) -> None:
        self.paused = True
        if "pause_event" in self.control_events:
            self.control_events["pause_event"].set()
        self.status_badge.config(text="STATUS: PAUSED", fg="#ffcc00", bg="#4d3b1b")

    def _on_resume(self) -> None:
        self.paused = False
        if "pause_event" in self.control_events:
            self.control_events["pause_event"].clear()
        self.status_badge.config(text="STATUS: RUNNING", fg="#00ff88", bg="#1b4d2e")

    def _on_stop(self) -> None:
        if "stop_event" in self.control_events:
            self.control_events["stop_event"].set()
        self.status_badge.config(text="STATUS: STOPPED", fg="#ff4444", bg="#4d1b1b")

    def _on_reset(self) -> None:
        if "reset_event" in self.control_events:
            self.control_events["reset_event"].set()

    def _toggle_mode(self) -> None:
        current = self.btn_baseline.cget("text")
        if "MCR-UAV" in current:
            self.btn_baseline.config(text="MODE: BASELINE", fg="#ffaa00")
            if "baseline_event" in self.control_events:
                self.control_events["baseline_event"].set()
        else:
            self.btn_baseline.config(text="MODE: MCR-UAV", fg="#00ffff")
            if "baseline_event" in self.control_events:
                self.control_events["baseline_event"].clear()

    def set_seed(self, seed: int) -> None:
        """Switch seed logging destination."""
        self.current_seed = seed
        self.train_labels["Seed"].config(text=str(seed))
        try:
            self.csv_file.close()
        except Exception:
            pass
        self.telemetry_csv_path = os.path.join(self.save_dir, f"telemetry_seed_{self.current_seed}.csv")
        self.csv_file = open(self.telemetry_csv_path, "a", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.csv_file)
        if os.path.getsize(self.telemetry_csv_path) == 0:
            self.csv_writer.writerow([
                "timestamp", "seed", "iteration", "episode", "simulation_time",
                "pos_x", "pos_y", "pos_z", "target_x", "target_y", "target_z",
                "vel_x", "vel_y", "vel_z", "tracking_error", "reward",
                "disturbance", "motor_loe", "lambda_RL", "alpha_Q", "alpha_R",
                "alpha_P", "alpha_I", "alpha_D", "H", "success"
            ])
            self.csv_file.flush()

    def _poll_telemetry(self) -> None:
        """Drain queue and update UI elements at ~30 FPS."""
        packets_processed = 0
        latest_packet = None

        while not self.telemetry_queue.empty():
            try:
                packet = self.telemetry_queue.get_nowait()
                latest_packet = packet
                packets_processed += 1

                # Log to CSV
                pos = packet.get("position", [0.0, 0.0, 0.0])
                target = packet.get("target", [0.0, 0.0, 0.0])
                vel = packet.get("velocity", [0.0, 0.0, 0.0])
                sim_time = packet.get("sim_time", packet.get("step", 0) * 0.1)

                self.csv_writer.writerow([
                    f"{packet.get('timestamp', time.time()):.3f}",
                    packet.get("seed", self.current_seed),
                    packet.get("iteration", 1),
                    packet.get("episode", 1),
                    f"{sim_time:.3f}",
                    f"{pos[0]:.4f}", f"{pos[1]:.4f}", f"{pos[2]:.4f}",
                    f"{target[0]:.4f}", f"{target[1]:.4f}", f"{target[2]:.4f}",
                    f"{vel[0]:.4f}", f"{vel[1]:.4f}", f"{vel[2]:.4f}",
                    f"{packet.get('tracking_error', 0.0):.4f}",
                    f"{packet.get('reward', 0.0):.4f}",
                    packet.get("disturbance_type", "Nominal"),
                    f"{packet.get('motor_loe', 0.0):.2f}",
                    f"{packet.get('lambda_rl', 0.07):.4f}",
                    f"{packet.get('alpha_q', 1.0):.4f}",
                    f"{packet.get('alpha_r', 1.0):.4f}",
                    f"{packet.get('alpha_p', 1.0):.4f}",
                    f"{packet.get('alpha_i', 1.0):.4f}",
                    f"{packet.get('alpha_d', 1.0):.4f}",
                    packet.get("horizon", 20),
                    1 if packet.get("success", False) else 0
                ])

                # Accumulate plot history
                self.hist_time.append(sim_time)
                self.hist_error.append(packet.get("tracking_error", 0.0))
                self.hist_reward.append(packet.get("reward", 0.0))
                self.hist_cum_reward.append(packet.get("cum_reward", 0.0))
                self.hist_pos_z.append(pos[2])
                self.hist_target_z.append(target[2])
                self.hist_lambda_rl.append(packet.get("lambda_rl", 0.07))
                self.hist_alpha_q.append(packet.get("alpha_q", 1.0))
                self.hist_alpha_r.append(packet.get("alpha_r", 1.0))
                self.hist_alpha_p.append(packet.get("alpha_p", 1.0))
                self.hist_alpha_i.append(packet.get("alpha_i", 1.0))
                self.hist_alpha_d.append(packet.get("alpha_d", 1.0))
                self.hist_horizon.append(packet.get("horizon", 20))
                self.hist_vel_norm.append(float(np.linalg.norm(vel)))
                rpm = packet.get("motor_rpm", [0.0, 0.0, 0.0, 0.0])
                self.hist_rpm_mean.append(float(np.mean(rpm)))

                # Limit history length
                if len(self.hist_time) > self.history_len:
                    self.hist_time = self.hist_time[-self.history_len:]
                    self.hist_error = self.hist_error[-self.history_len:]
                    self.hist_reward = self.hist_reward[-self.history_len:]
                    self.hist_cum_reward = self.hist_cum_reward[-self.history_len:]
                    self.hist_pos_z = self.hist_pos_z[-self.history_len:]
                    self.hist_target_z = self.hist_target_z[-self.history_len:]
                    self.hist_lambda_rl = self.hist_lambda_rl[-self.history_len:]
                    self.hist_alpha_q = self.hist_alpha_q[-self.history_len:]
                    self.hist_alpha_r = self.hist_alpha_r[-self.history_len:]
                    self.hist_alpha_p = self.hist_alpha_p[-self.history_len:]
                    self.hist_alpha_i = self.hist_alpha_i[-self.history_len:]
                    self.hist_alpha_d = self.hist_alpha_d[-self.history_len:]
                    self.hist_horizon = self.hist_horizon[-self.history_len:]
                    self.hist_vel_norm = self.hist_vel_norm[-self.history_len:]
                    self.hist_rpm_mean = self.hist_rpm_mean[-self.history_len:]

            except queue.Empty:
                break

        if packets_processed > 0:
            self.csv_file.flush()

        if latest_packet is not None:
            self.last_packet = latest_packet
            self._update_readouts(latest_packet)
            self._update_sim_view(latest_packet)
            self._update_latent_bars(latest_packet)
            self._update_plots()

        if self.is_running:
            self.after_id = self.root.after(33, self._poll_telemetry)

    def _update_readouts(self, p: Dict[str, Any]) -> None:
        # Time and Status
        step = p.get("step", 0)
        sim_time = p.get("sim_time", step * 0.1)
        self.time_lbl.config(text=f"Sim Time: {sim_time:.2f}s | Step: {step}")
        if "status" in p:
            self.status_badge.config(text=f"STATUS: {p['status']}")

        # Supervisor Readouts
        self.param_labels["λ_RL (Blending)"].config(text=f"{p.get('lambda_rl', 0.07):.4f}")
        self.param_labels["α_Q (State Cost)"].config(text=f"{p.get('alpha_q', 1.0):.4f}")
        self.param_labels["α_R (Input Cost)"].config(text=f"{p.get('alpha_r', 1.0):.4f}")
        self.param_labels["α_P (PID Prop)"].config(text=f"{p.get('alpha_p', 1.0):.4f}")
        self.param_labels["α_I (PID Integ)"].config(text=f"{p.get('alpha_i', 1.0):.4f}")
        self.param_labels["α_D (PID Deriv)"].config(text=f"{p.get('alpha_d', 1.0):.4f}")
        self.param_labels["MPC Horizon H"].config(text=str(p.get("horizon", 20)))

        # Training Readouts
        self.train_labels["Seed"].config(text=str(p.get("seed", self.current_seed)))
        self.train_labels["Iteration"].config(text=f"{p.get('iteration', 1)} / 50")
        self.train_labels["Episode"].config(text=str(p.get("episode", 1)))
        self.train_labels["Current Return"].config(text=f"{p.get('cum_reward', 0.0):.2f}")
        self.train_labels["Mean Return"].config(text=f"{p.get('mean_return', 0.0):.2f}")
        self.train_labels["Step Reward"].config(text=f"{p.get('reward', 0.0):.3f}")
        self.train_labels["Grad Norm"].config(text=f"{p.get('grad_norm', 0.0):.4f}")
        self.train_labels["Param Delta"].config(text=f"{p.get('param_delta', 0.0):.5f}")
        self.train_labels["Val Success"].config(text=f"{p.get('val_success', 0.0):.1%}")
        self.train_labels["Val RMSE"].config(text=f"{p.get('val_rmse', 0.0):.3f} m")
        self.train_labels["Val Energy"].config(text=f"{p.get('val_energy', 0.0):.1f}")
        self.train_labels["Tracking Error"].config(text=f"{p.get('tracking_error', 0.0):.3f} m")

        # Disturbance Readouts
        self.dist_labels["Disturbance Type"].config(text=str(p.get("disturbance_type", "Nominal")))
        self.dist_labels["Wind / Gust Mag"].config(text=f"{p.get('disturbance_magnitude', 0.0):.2f} m/s")
        self.dist_labels["Motor LoE"].config(text=f"{p.get('motor_loe', 0.0):.1f}%")
        self.dist_labels["Degraded Rotors"].config(text=str(p.get("degraded_motors", "None")))

    def _update_sim_view(self, p: Dict[str, Any]) -> None:
        """Render high-contrast 2D/3D flight projection on the canvas."""
        c = self.sim_canvas
        c.delete("all")
        w = c.winfo_width()
        h = c.winfo_height()
        if w < 10 or h < 10:
            w, h = 420, 250

        cx, cy = w / 2, h / 2
        scale = min(w, h) / 5.2  # 5.2 meter world bounds

        # Draw grid
        for d in np.linspace(-2.5, 2.5, 11):
            gx = cx + d * scale
            gy = cy - d * scale
            c.create_line(gx, 0, gx, h, fill="#1c1e29", dash=(1, 3))
            c.create_line(0, gy, w, gy, fill="#1c1e29", dash=(1, 3))

        # Draw coordinate origin
        c.create_line(cx - 8, cy, cx + 8, cy, fill="#3a4055")
        c.create_line(cx, cy - 8, cx, cy + 8, fill="#3a4055")

        # Obstacles
        obs_list = p.get("obstacles", [])
        for obs in obs_list:
            ox, oy = obs.get("x", 0.0), obs.get("y", 0.0)
            hw = obs.get("hx", 0.1) * scale
            hh = obs.get("hy", 0.1) * scale
            x1 = cx + ox * scale - hw
            y1 = cy - oy * scale - hh
            x2 = cx + ox * scale + hw
            y2 = cy - oy * scale + hh
            c.create_rectangle(x1, y1, x2, y2, fill="#7a5520", outline="#ffaa00", width=1)

        # Target Position & Threshold Ring (0.60m)
        tx, ty = p.get("target", [0.0, 0.0, 0.0])[0:2]
        tx_px = cx + tx * scale
        ty_px = cy - ty * scale
        r_thresh = 0.60 * scale
        c.create_oval(tx_px - r_thresh, ty_px - r_thresh, tx_px + r_thresh, ty_px + r_thresh, outline="#00ff88", dash=(3, 3), width=1)
        c.create_oval(tx_px - 5, ty_px - 5, tx_px + 5, ty_px + 5, fill="#ff3333", outline="#ffffff", width=1)
        c.create_text(tx_px, ty_px - 10, text="TARGET", fill="#ff5555", font=("Consolas", 8, "bold"))

        # UAV Position & Trail
        ux, uy, uz = p.get("position", [0.0, 0.0, 0.0])[0:3]
        ux_px = cx + ux * scale
        uy_px = cy - uy * scale

        self.flight_trail.append((ux_px, uy_px))
        if len(self.flight_trail) > 120:
            self.flight_trail.pop(0)

        # Draw trail
        if len(self.flight_trail) > 1:
            for i in range(len(self.flight_trail) - 1):
                c.create_line(
                    self.flight_trail[i][0], self.flight_trail[i][1],
                    self.flight_trail[i+1][0], self.flight_trail[i+1][1],
                    fill="#00adb5", width=1
                )

        # Reference trajectory line
        c.create_line(ux_px, uy_px, tx_px, ty_px, fill="#555577", dash=(2, 2))

        # UAV Body & Heading
        rpy = p.get("rpy", [0.0, 0.0, 0.0])
        yaw = rpy[2]
        u_size = 10
        c.create_oval(ux_px - u_size, uy_px - u_size, ux_px + u_size, uy_px + u_size, fill="#00ffff", outline="#ffffff", width=1)
        # Heading nose line
        hx = ux_px + math.cos(yaw) * (u_size + 8)
        hy = uy_px - math.sin(yaw) * (u_size + 8)
        c.create_line(ux_px, uy_px, hx, hy, fill="#ff00dd", width=2)

        # On-screen HUD info
        c.create_text(8, 12, text=f"UAV: ({ux:.2f}, {uy:.2f}, {uz:.2f})m", fill="#00ffff", font=("Consolas", 8), anchor="w")
        c.create_text(8, 26, text=f"TGT: ({tx:.2f}, {ty:.2f}, {p.get('target', [0,0,0])[2]:.2f})m", fill="#ff5555", font=("Consolas", 8), anchor="w")
        c.create_text(8, 40, text=f"ERR: {p.get('tracking_error', 0.0):.3f}m", fill="#00ff88", font=("Consolas", 8), anchor="w")

    def _update_latent_bars(self, p: Dict[str, Any]) -> None:
        """Render 16 latent context bars for z_t in [-2.0, 2.0]."""
        c = self.ctx_canvas
        c.delete("all")
        w = c.winfo_width()
        h = c.winfo_height()
        if w < 10 or h < 10:
            w, h = 380, 250

        z = p.get("z_t", [0.0] * 16)
        if len(z) < 16:
            z = list(z) + [0.0] * (16 - len(z))

        margin_x = 35
        bar_w = (w - margin_x - 10) / 16
        mid_y = h / 2

        # Draw zero line
        c.create_line(margin_x, mid_y, w - 10, mid_y, fill="#404555", width=1)
        c.create_text(18, mid_y, text="0.0", fill="#777777", font=("Consolas", 7))
        c.create_text(18, 15, text="+2.0", fill="#777777", font=("Consolas", 7))
        c.create_text(18, h - 15, text="-2.0", fill="#777777", font=("Consolas", 7))

        for i in range(16):
            val = float(z[i])
            val_clamped = max(-2.0, min(2.0, val))
            bar_h = (val_clamped / 2.0) * (h / 2 - 20)

            bx = margin_x + i * bar_w + 2
            if bar_h >= 0:
                y1 = mid_y - bar_h
                y2 = mid_y
                color = "#00ff88"
            else:
                y1 = mid_y
                y2 = mid_y - bar_h
                color = "#ff5555"

            c.create_rectangle(bx, y1, bx + bar_w - 4, y2, fill=color, outline="")
            c.create_text(bx + (bar_w - 4)/2, h - 8, text=f"{i}", fill="#666666", font=("Consolas", 6))

    def _update_plots(self) -> None:
        """Refresh 6 real-time performance subplots."""
        if len(self.hist_time) < 2:
            return

        t = self.hist_time

        # Plot 0,0: Tracking Error
        ax00 = self.axs[0, 0]
        ax00.clear()
        ax00.set_facecolor("#0b0c12")
        ax00.grid(True, linestyle=":", alpha=0.3, color="#404555")
        ax00.plot(t, self.hist_error, color="#00d2ff", lw=1.2, label="Error (m)")
        ax00.axhline(0.60, color="#ffaa00", linestyle="--", lw=0.9, label="Thresh (0.6m)")
        ax00.set_title("Tracking Error (m)", fontsize=8, color="#00d2ff")
        ax00.legend(loc="upper right", fontsize=6)
        ax00.tick_params(labelsize=6, colors="#888888")

        # Plot 0,1: Altitude
        ax01 = self.axs[0, 1]
        ax01.clear()
        ax01.set_facecolor("#0b0c12")
        ax01.grid(True, linestyle=":", alpha=0.3, color="#404555")
        ax01.plot(t, self.hist_pos_z, color="#00ff88", lw=1.2, label="UAV z")
        ax01.plot(t, self.hist_target_z, color="#ff5555", linestyle="--", lw=1.0, label="Target z")
        ax01.set_title("Altitude: UAV z vs Target z", fontsize=8, color="#00ff88")
        ax01.legend(loc="upper right", fontsize=6)
        ax01.tick_params(labelsize=6, colors="#888888")

        # Plot 0,2: Reward
        ax02 = self.axs[0, 2]
        ax02.clear()
        ax02.set_facecolor("#0b0c12")
        ax02.grid(True, linestyle=":", alpha=0.3, color="#404555")
        ax02.plot(t, self.hist_reward, color="#ffcc00", lw=1.0, label="Step Rew")
        ax02.set_title("Reward vs Timestep", fontsize=8, color="#ffcc00")
        ax02.tick_params(labelsize=6, colors="#888888")

        # Plot 1,0: Reconfiguration
        ax10 = self.axs[1, 0]
        ax10.clear()
        ax10.set_facecolor("#0b0c12")
        ax10.grid(True, linestyle=":", alpha=0.3, color="#404555")
        ax10.plot(t, self.hist_lambda_rl, color="#00ffff", lw=1.0, label="λ_RL")
        ax10.plot(t, self.hist_alpha_q, color="#00ff88", lw=1.0, label="α_Q")
        ax10.plot(t, self.hist_alpha_r, color="#ff00dd", lw=1.0, label="α_R")
        ax10.set_title("Reconfig: λ_RL, α_Q, α_R", fontsize=8, color="#ff00dd")
        ax10.legend(loc="upper right", fontsize=6)
        ax10.tick_params(labelsize=6, colors="#888888")

        # Plot 1,1: PID Scalings & Horizon
        ax11 = self.axs[1, 1]
        ax11.clear()
        ax11.set_facecolor("#0b0c12")
        ax11.grid(True, linestyle=":", alpha=0.3, color="#404555")
        ax11.plot(t, self.hist_alpha_p, color="#ffaa00", lw=1.0, label="α_P")
        ax11.plot(t, self.hist_alpha_i, color="#00ffaa", lw=1.0, label="α_I")
        ax11.plot(t, self.hist_alpha_d, color="#ff55aa", lw=1.0, label="α_D")
        ax11.set_title("PID Scalings (α_P, α_I, α_D)", fontsize=8, color="#00ffff")
        ax11.legend(loc="upper right", fontsize=6)
        ax11.tick_params(labelsize=6, colors="#888888")

        # Plot 1,2: Velocity & Motor RPM
        ax12 = self.axs[1, 2]
        ax12.clear()
        ax12.set_facecolor("#0b0c12")
        ax12.grid(True, linestyle=":", alpha=0.3, color="#404555")
        ax12.plot(t, self.hist_vel_norm, color="#ff7777", lw=1.0, label="||v|| (m/s)")
        ax12.set_title("Velocity Norm (m/s)", fontsize=8, color="#ff7777")
        ax12.legend(loc="upper right", fontsize=6)
        ax12.tick_params(labelsize=6, colors="#888888")

        self.fig.tight_layout()
        self.plot_canvas.draw_idle()

    def close(self) -> None:
        self.is_running = False
        try:
            self.csv_file.close()
        except Exception:
            pass
        def _cleanup():
            try:
                if hasattr(self, "after_id") and self.after_id is not None:
                    self.root.after_cancel(self.after_id)
                    self.after_id = None
                self.root.quit()
                self.root.destroy()
            except Exception:
                pass
        try:
            self.root.after(0, _cleanup)
        except Exception:
            pass


def start_live_gui(
    telemetry_queue: queue.Queue,
    control_events: Dict[str, threading.Event],
    save_dir: str = "meta_rl_results/gui",
    master_seed: int = 42,
) -> MCRUAVLiveMonitor:
    """Factory helper to construct and return the live monitor."""
    return MCRUAVLiveMonitor(
        telemetry_queue=telemetry_queue,
        control_events=control_events,
        save_dir=save_dir,
        master_seed=master_seed,
    )
