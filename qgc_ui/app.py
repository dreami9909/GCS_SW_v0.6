from __future__ import annotations

import math
import time
import tkinter as tk

from .components import StatusChip, Theme, configure_ttk_styles, flat_button
from .domain import MissionStore, VehicleState
from .tactical import TacticalState
from .tactical_views import DataView, TacticalFlyView
from .views import AnalyzeView, PlanView


class QGCApplication(tk.Tk):
    """QGroundControl-inspired UI shell with an isolated mock vehicle model."""

    def __init__(self, *, start_maximized: bool = True) -> None:
        super().__init__()
        self.title("Tactical Ground Control // Training UI")
        self.geometry("1440x900")
        self.minsize(1180, 720)
        self.configure(background=Theme.WINDOW)
        configure_ttk_styles(self)

        self.vehicle = VehicleState(
            connected=True,
            ready_text="Ready To Fly",
            flight_mode="Hold",
            gps_satellites=12,
            gps_fix="3D Fix",
            battery_percent=87,
            link_percent=98,
        )
        self.mission = MissionStore()
        self.mission.seed_demo()
        self.tactical = TacticalState.demo(
            self.mission.home_latitude,
            self.mission.home_longitude,
        )
        self._active_view = "Fly"
        self._nav_buttons: dict[str, tk.Button] = {}
        self._started_at = time.monotonic()
        self._status_reset_job: str | None = None

        self._build_toolbar()
        self._build_views()
        self._build_footer()
        self.show_view("Fly")
        self._tick_mock_vehicle()

        self.bind("<F1>", lambda _event: self.show_view("Fly"))
        self.bind("<F2>", lambda _event: self.show_view("Plan"))
        self.bind("<F3>", lambda _event: self.show_view("Analyze"))
        self.bind("<F4>", lambda _event: self.show_view("Data"))
        self.bind("<Escape>", lambda _event: self.attributes("-fullscreen", False))

        if start_maximized:
            self.after(20, self._show_main_window)

    def _show_main_window(self) -> None:
        """Show the window and bring it to the foreground on Windows."""
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        usable_height = max(screen_height - 48, self.minsize()[1])
        self.geometry(f"{screen_width}x{usable_height}+0+0")
        self.deiconify()
        self.lift()
        try:
            self.attributes("-topmost", True)
            self.after(500, lambda: self.attributes("-topmost", False))
        except tk.TclError:
            pass

    def _build_toolbar(self) -> None:
        self.toolbar = tk.Frame(self, background=Theme.TOOLBAR, height=58)
        self.toolbar.pack(side=tk.TOP, fill=tk.X)
        self.toolbar.pack_propagate(False)

        data_button = flat_button(
            self.toolbar,
            "▤ DATA",
            lambda: self.show_view("Data"),
            background=Theme.ACCENT,
            foreground="#1c2225",
            active_background="#ffd24d",
            font=("Consolas", 12, "bold"),
            padx=15,
            pady=10,
        )
        data_button.pack(side=tk.LEFT)
        self._nav_buttons["Data"] = data_button

        for name in ("Fly", "Plan", "Analyze"):
            button = flat_button(
                self.toolbar,
                name,
                lambda view=name: self.show_view(view),
                background=Theme.TOOLBAR,
                active_background="#333c41",
                font=("Segoe UI", 11, "bold"),
                padx=18,
                pady=13,
            )
            button.pack(side=tk.LEFT)
            self._nav_buttons[name] = button

        tk.Frame(self.toolbar, background="#40484c", width=1).pack(
            side=tk.LEFT,
            fill=tk.Y,
            padx=(4, 8),
            pady=9,
        )

        self.ready_chip = StatusChip(self.toolbar, "FLIGHT STATUS", width=122)
        self.ready_chip.pack(side=tk.LEFT)
        self.mode_chip = StatusChip(self.toolbar, "FLIGHT MODE", width=95)
        self.mode_chip.pack(side=tk.LEFT)
        self.message_chip = StatusChip(self.toolbar, "MESSAGES", width=78)
        self.message_chip.pack(side=tk.LEFT)

        self.connect_button = flat_button(
            self.toolbar,
            "Mock Link",
            self.toggle_connection,
            background="#30393e",
            active_background="#46535a",
            font=("Segoe UI", 9, "bold"),
            padx=14,
            pady=9,
        )
        self.connect_button.pack(side=tk.RIGHT, padx=(7, 10), pady=8)

        self.battery_chip = StatusChip(self.toolbar, "BATTERY", width=78)
        self.battery_chip.pack(side=tk.RIGHT)
        self.link_chip = StatusChip(self.toolbar, "LINK", width=68)
        self.link_chip.pack(side=tk.RIGHT)
        self.gps_chip = StatusChip(self.toolbar, "GPS", width=82)
        self.gps_chip.pack(side=tk.RIGHT)

    def _build_views(self) -> None:
        self.view_container = tk.Frame(self, background=Theme.WINDOW)
        self.view_container.pack(fill=tk.BOTH, expand=True)
        self.pages = {
            "Fly": TacticalFlyView(
                self.view_container,
                self.mission,
                self.vehicle,
                self.tactical,
                self.set_status,
            ),
            "Plan": PlanView(
                self.view_container,
                self.mission,
                self.vehicle,
                self.set_status,
            ),
            "Analyze": AnalyzeView(self.view_container, self.set_status),
            "Data": DataView(
                self.view_container,
                self.vehicle,
                self.tactical,
                self.set_status,
            ),
        }
        for page in self.pages.values():
            page.place(x=0, y=0, relwidth=1, relheight=1)

    def _build_footer(self) -> None:
        footer = tk.Frame(self, background="#13181b", height=28)
        footer.pack(side=tk.BOTTOM, fill=tk.X)
        footer.pack_propagate(False)
        self.status_label = tk.Label(
            footer,
            text="Training UI ready // synthetic MAVLink source connected",
            background="#13181b",
            foreground="#b7c0c4",
            font=("Consolas", 8),
            anchor="w",
            padx=10,
        )
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.clock_label = tk.Label(
            footer,
            text="",
            background="#13181b",
            foreground="#879398",
            font=("Consolas", 8),
            padx=10,
        )
        self.clock_label.pack(side=tk.RIGHT)

    def show_view(self, name: str) -> None:
        if name not in self.pages:
            return
        self._active_view = name
        self.pages[name].lift()
        for view_name, button in self._nav_buttons.items():
            selected = view_name == name
            button.configure(
                background=(
                    Theme.ACCENT
                    if selected and view_name == "Data"
                    else "#343e43"
                    if selected
                    else Theme.TOOLBAR
                ),
                foreground=(
                    "#172018"
                    if selected and view_name == "Data"
                    else Theme.ACCENT
                    if selected
                    else Theme.LIGHT_TEXT
                ),
            )
        self.set_status(f"{name} View")

    def set_status(self, message: str, *, persistent: bool = False) -> None:
        self.status_label.configure(text=message)
        if self._status_reset_job is not None:
            self.after_cancel(self._status_reset_job)
            self._status_reset_job = None
        if not persistent:
            self._status_reset_job = self.after(
                4500,
                lambda: self.status_label.configure(
                    text=(
                        "Mock vehicle connected"
                        if self.vehicle.connected
                        else "No vehicle connected"
                    )
                ),
            )

    def toggle_connection(self) -> None:
        self.vehicle.connected = not self.vehicle.connected
        if self.vehicle.connected:
            self.vehicle.ready_text = "Ready To Fly"
            self.vehicle.gps_satellites = 12
            self.vehicle.gps_fix = "3D Fix"
            self.vehicle.link_percent = 98
            self.vehicle.battery_percent = max(self.vehicle.battery_percent, 80)
            self.set_status("Mock MAVLink link connected.")
        else:
            self.vehicle.ready_text = "Communication Lost"
            self.vehicle.gps_satellites = 0
            self.vehicle.gps_fix = "No GPS"
            self.vehicle.link_percent = 0
            self.set_status("Mock MAVLink link disconnected.")
        self.update_toolbar()

    def update_toolbar(self) -> None:
        vehicle = self.vehicle
        ready_color = (
            Theme.GREEN
            if vehicle.connected and vehicle.ready_text == "Ready To Fly"
            else Theme.RED
        )
        self.ready_chip.set(vehicle.ready_text, ready_color)
        self.mode_chip.set(vehicle.flight_mode if vehicle.connected else "--")
        self.message_chip.set("0", Theme.GREEN if vehicle.connected else Theme.MUTED)
        self.gps_chip.set(
            f"{vehicle.gps_satellites} sat",
            Theme.GREEN if vehicle.gps_satellites >= 8 else Theme.RED,
        )
        self.link_chip.set(
            f"{vehicle.link_percent}%",
            Theme.GREEN if vehicle.link_percent >= 60 else Theme.RED,
        )
        battery_color = (
            Theme.GREEN
            if vehicle.battery_percent >= 40
            else Theme.ACCENT
            if vehicle.battery_percent >= 20
            else Theme.RED
        )
        self.battery_chip.set(f"{vehicle.battery_percent}%", battery_color)
        self.connect_button.configure(
            text="Mock Link" if vehicle.connected else "Connect",
            foreground=Theme.GREEN if vehicle.connected else Theme.LIGHT_TEXT,
        )

    def _tick_mock_vehicle(self) -> None:
        vehicle = self.vehicle
        elapsed = time.monotonic() - self._started_at
        self.clock_label.configure(text=time.strftime("%H:%M:%S"))
        if vehicle.connected:
            orbit_radius = 0.0012
            vehicle.latitude = self.mission.home_latitude + math.sin(elapsed / 10) * orbit_radius
            vehicle.longitude = self.mission.home_longitude + math.cos(elapsed / 10) * orbit_radius
            vehicle.heading_deg = (elapsed * 12) % 360
            vehicle.roll_deg = math.sin(elapsed / 2.4) * 8
            vehicle.pitch_deg = math.sin(elapsed / 3.1) * 4
            vehicle.relative_altitude_m = 25 + math.sin(elapsed / 4) * 3
            vehicle.ground_speed_mps = 7.2 + math.sin(elapsed / 2) * 0.6
            vehicle.vertical_speed_mps = math.cos(elapsed / 4) * 0.7
            vehicle.flight_time_s = int(elapsed)
            vehicle.link_percent = 96 + int((math.sin(elapsed / 5) + 1) * 1.5)
        self.tactical.tick(elapsed, link_connected=vehicle.connected)
        self.update_toolbar()
        fly_view = self.pages.get("Fly")
        if isinstance(fly_view, TacticalFlyView):
            fly_view.update_display()
        plan_view = self.pages.get("Plan")
        if isinstance(plan_view, PlanView) and self._active_view == "Plan":
            plan_view.map.render()
        data_view = self.pages.get("Data")
        if isinstance(data_view, DataView):
            data_view.update_data()
        self.after(250, self._tick_mock_vehicle)
