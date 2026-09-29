from __future__ import annotations

import json
import math
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable

from .components import Theme, ToolButton, flat_button
from .domain import MissionCommand, MissionStore, VehicleState
from .map_widget import MapWidget


class PlanView(tk.Frame):
    def __init__(
        self,
        master: tk.Misc,
        mission: MissionStore,
        vehicle: VehicleState,
        status_callback: Callable[[str], None],
    ) -> None:
        super().__init__(master, background=Theme.WINDOW)
        self.mission = mission
        self.vehicle = vehicle
        self.status_callback = status_callback
        self.command_var = tk.StringVar(value=MissionCommand.WAYPOINT.value)
        self._refreshing_tree = False
        self._detail_vars = {
            name: tk.StringVar()
            for name in ("latitude", "longitude", "altitude_m", "hold_s")
        }

        self._build_tool_rail()
        self._build_editor()

        self.map = MapWidget(
            self,
            mission,
            vehicle,
            editable=True,
            command_provider=self.current_command,
            status_callback=status_callback,
        )
        self.map.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.tool_rail.lift()
        self.editor.lift()

        mission.subscribe(self.refresh)
        self.refresh()

    def _build_tool_rail(self) -> None:
        self.tool_rail = tk.Frame(self, background=Theme.TOOLBAR_ALT, width=82)
        self.tool_rail.pack(side=tk.LEFT, fill=tk.Y)
        self.tool_rail.pack_propagate(False)

        tk.Label(
            self.tool_rail,
            text="PLAN",
            background=Theme.TOOLBAR_ALT,
            foreground=Theme.ACCENT,
            font=("Segoe UI", 9, "bold"),
            pady=10,
        ).pack(fill=tk.X)

        tools = [
            ("↗", "Takeoff", MissionCommand.TAKEOFF),
            ("●", "Waypoint", MissionCommand.WAYPOINT),
            ("↻", "Loiter", MissionCommand.LOITER),
            ("⌂", "Return", MissionCommand.RTL),
            ("↓", "Land", MissionCommand.LAND),
        ]
        for icon, label, command in tools:
            button = tk.Radiobutton(
                self.tool_rail,
                text=f"{icon}\n{label}",
                value=command.value,
                variable=self.command_var,
                indicatoron=False,
                background=Theme.TOOLBAR_ALT,
                foreground=Theme.LIGHT_TEXT,
                activebackground="#374248",
                activeforeground=Theme.LIGHT_TEXT,
                selectcolor=Theme.ACCENT_DARK,
                relief=tk.FLAT,
                borderwidth=0,
                font=("Segoe UI", 9),
                height=3,
                cursor="hand2",
                command=lambda value=command.value: self.status_callback(
                    f"{value}: click the map to add an item."
                ),
            )
            button.pack(fill=tk.X, pady=1)

        tk.Frame(self.tool_rail, background="#4b555a", height=1).pack(fill=tk.X, pady=5)
        ToolButton(self.tool_rail, "⌖", "Center", self._center_map).pack(fill=tk.X)

    def _build_editor(self) -> None:
        self.editor = tk.Frame(self, background=Theme.PANEL, width=345)
        self.editor.pack(side=tk.RIGHT, fill=tk.Y)
        self.editor.pack_propagate(False)

        header = tk.Frame(self.editor, background="#ffffff", height=62)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        tk.Label(
            header,
            text="Mission",
            background="#ffffff",
            foreground=Theme.TEXT,
            font=("Segoe UI", 18, "bold"),
        ).pack(side=tk.LEFT, padx=15)
        self.sync_label = tk.Label(
            header,
            text="Not synced",
            background="#ffffff",
            foreground=Theme.MUTED,
            font=("Segoe UI", 8),
        )
        self.sync_label.pack(side=tk.RIGHT, padx=14)

        file_row = tk.Frame(self.editor, background=Theme.PANEL)
        file_row.pack(fill=tk.X, padx=10, pady=(10, 4))
        for label, callback in (
            ("New", self.clear),
            ("Open", self.open_file),
            ("Save", self.save_file),
        ):
            flat_button(
                file_row,
                label,
                callback,
                background="#d8dcde",
                foreground=Theme.TEXT,
                active_background="#c8cdcf",
                padx=12,
                pady=6,
            ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        sync_row = tk.Frame(self.editor, background=Theme.PANEL)
        sync_row.pack(fill=tk.X, padx=10, pady=4)
        flat_button(
            sync_row,
            "Upload",
            lambda: self._mock_sync("Upload"),
            background=Theme.BLUE,
            padx=12,
            pady=7,
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        flat_button(
            sync_row,
            "Download",
            lambda: self._mock_sync("Download"),
            background=Theme.BLUE,
            padx=12,
            pady=7,
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        tk.Label(
            self.editor,
            text="Mission Items",
            background=Theme.PANEL,
            foreground=Theme.TEXT,
            font=("Segoe UI", 10, "bold"),
            anchor="w",
        ).pack(fill=tk.X, padx=12, pady=(12, 4))

        tree_frame = tk.Frame(self.editor, background=Theme.PANEL)
        tree_frame.pack(fill=tk.BOTH, padx=10)
        self.tree = ttk.Treeview(
            tree_frame,
            style="QGC.Treeview",
            columns=("sequence", "command", "altitude"),
            show="headings",
            height=8,
            selectmode="browse",
        )
        self.tree.heading("sequence", text="#")
        self.tree.heading("command", text="Command")
        self.tree.heading("altitude", text="Alt")
        self.tree.column("sequence", width=36, anchor="center", stretch=False)
        self.tree.column("command", width=145, anchor="w")
        self.tree.column("altitude", width=62, anchor="e", stretch=False)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)
        scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=scrollbar.set)

        delete_row = tk.Frame(self.editor, background=Theme.PANEL)
        delete_row.pack(fill=tk.X, padx=10, pady=5)
        flat_button(
            delete_row,
            "Delete selected",
            self.delete_selected,
            background="#d8dcde",
            foreground=Theme.TEXT,
            active_background="#c8cdcf",
            pady=6,
        ).pack(side=tk.RIGHT)

        details = tk.LabelFrame(
            self.editor,
            text="  Selected Item  ",
            background=Theme.PANEL,
            foreground=Theme.TEXT,
            font=("Segoe UI", 9, "bold"),
            bd=1,
            relief=tk.GROOVE,
        )
        details.pack(fill=tk.X, padx=10, pady=(5, 8))
        labels = (
            ("Latitude", "latitude"),
            ("Longitude", "longitude"),
            ("Altitude (m)", "altitude_m"),
            ("Hold (s)", "hold_s"),
        )
        for row, (label, key) in enumerate(labels):
            tk.Label(
                details,
                text=label,
                background=Theme.PANEL,
                foreground=Theme.TEXT,
                anchor="w",
            ).grid(row=row, column=0, sticky="w", padx=8, pady=4)
            entry = tk.Entry(
                details,
                textvariable=self._detail_vars[key],
                background="white",
                foreground=Theme.TEXT,
                relief=tk.SOLID,
                borderwidth=1,
                font=("Consolas", 9),
            )
            entry.grid(row=row, column=1, sticky="ew", padx=8, pady=4)
        details.columnconfigure(1, weight=1)
        flat_button(
            details,
            "Apply",
            self.apply_details,
            background=Theme.ACCENT,
            foreground="#212121",
            active_background="#ffd54d",
            pady=6,
        ).grid(row=4, column=0, columnspan=2, sticky="ew", padx=8, pady=8)

        summary = tk.Frame(self.editor, background="#dfe3e5")
        summary.pack(side=tk.BOTTOM, fill=tk.X)
        self.summary_label = tk.Label(
            summary,
            text="",
            background="#dfe3e5",
            foreground=Theme.TEXT,
            justify=tk.LEFT,
            anchor="w",
            font=("Segoe UI", 9),
            padx=12,
            pady=9,
        )
        self.summary_label.pack(fill=tk.X)

    def current_command(self) -> MissionCommand | None:
        try:
            return MissionCommand(self.command_var.get())
        except ValueError:
            return None

    def refresh(self) -> None:
        self._refreshing_tree = True
        try:
            self.tree.delete(*self.tree.get_children())
            for waypoint in self.mission.waypoints:
                self.tree.insert(
                    "",
                    "end",
                    iid=str(waypoint.sequence),
                    values=(
                        waypoint.sequence,
                        waypoint.command.value,
                        f"{waypoint.altitude_m:.0f} m",
                    ),
                )
            if self.mission.selected_sequence is not None:
                item_id = str(self.mission.selected_sequence)
                if self.tree.exists(item_id):
                    self.tree.selection_set(item_id)
                    self.tree.see(item_id)
            selected = self.mission.get_selected()
            values = {
                "latitude": f"{selected.latitude:.7f}" if selected else "",
                "longitude": f"{selected.longitude:.7f}" if selected else "",
                "altitude_m": f"{selected.altitude_m:.1f}" if selected else "",
                "hold_s": f"{selected.hold_s:.1f}" if selected else "",
            }
            for key, value in values.items():
                self._detail_vars[key].set(value)
            distance = self.mission.total_distance_m()
            estimated = self.mission.estimated_time_s()
            self.summary_label.configure(
                text=(
                    f"Mission items  {len(self.mission.waypoints)}\n"
                    f"Distance  {distance / 1000:.2f} km    "
                    f"Estimated  {estimated / 60:.1f} min"
                )
            )
        finally:
            self._refreshing_tree = False

    def _on_tree_select(self, _event: tk.Event) -> None:
        if self._refreshing_tree:
            return
        selection = self.tree.selection()
        if selection:
            sequence = int(selection[0])
            if sequence != self.mission.selected_sequence:
                self.mission.select(sequence)

    def apply_details(self) -> None:
        try:
            latitude = float(self._detail_vars["latitude"].get())
            longitude = float(self._detail_vars["longitude"].get())
            altitude = float(self._detail_vars["altitude_m"].get())
            hold = float(self._detail_vars["hold_s"].get())
        except ValueError:
            messagebox.showerror("Invalid value", "Enter valid numeric values.")
            return
        if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
            messagebox.showerror("Invalid coordinate", "Latitude or longitude is out of range.")
            return
        if altitude < 0 or hold < 0:
            messagebox.showerror("Invalid value", "Altitude and hold time cannot be negative.")
            return
        if self.mission.update_selected(
            latitude=latitude,
            longitude=longitude,
            altitude_m=altitude,
            hold_s=hold,
        ):
            self.status_callback("Mission item updated.")

    def delete_selected(self) -> None:
        if self.mission.delete_selected():
            self.status_callback("Mission item deleted.")

    def clear(self) -> None:
        if self.mission.waypoints and not messagebox.askyesno(
            "New mission",
            "Clear the current mission?",
        ):
            return
        self.mission.clear()
        self.status_callback("New empty mission created.")

    def save_file(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save Plan",
            defaultextension=".plan.json",
            filetypes=[("Python GCS Plan", "*.plan.json"), ("JSON", "*.json")],
        )
        if not path:
            return
        data = {
            "format": "python-qgc-plan",
            "version": 1,
            "home": {
                "latitude": self.mission.home_latitude,
                "longitude": self.mission.home_longitude,
            },
            "items": [
                {
                    "sequence": waypoint.sequence,
                    "command": waypoint.command.value,
                    "latitude": waypoint.latitude,
                    "longitude": waypoint.longitude,
                    "altitude_m": waypoint.altitude_m,
                    "hold_s": waypoint.hold_s,
                }
                for waypoint in self.mission.waypoints
            ],
        }
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        self.status_callback(f"Saved plan: {Path(path).name}")

    def open_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Open Plan",
            filetypes=[("Python GCS Plan", "*.plan.json"), ("JSON", "*.json")],
        )
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            home = data["home"]
            items = data["items"]
            new_items = []
            for index, item in enumerate(items, start=1):
                from .domain import Waypoint

                new_items.append(
                    Waypoint(
                        sequence=index,
                        command=MissionCommand(item["command"]),
                        latitude=float(item["latitude"]),
                        longitude=float(item["longitude"]),
                        altitude_m=float(item.get("altitude_m", 50)),
                        hold_s=float(item.get("hold_s", 0)),
                    )
                )
            self.mission.home_latitude = float(home["latitude"])
            self.mission.home_longitude = float(home["longitude"])
            self.mission.waypoints = new_items
            self.mission.selected_sequence = 1 if new_items else None
            self.mission.notify()
            self.map.reset_view()
            self.status_callback(f"Opened plan: {Path(path).name}")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError) as error:
            messagebox.showerror("Open Plan", f"Could not open plan:\n{error}")

    def _mock_sync(self, operation: str) -> None:
        self.sync_label.configure(text=f"{operation}…", foreground=Theme.ACCENT_DARK)
        self.status_callback(f"{operation}: mock MAVLink transfer started.")
        self.after(900, lambda: self._finish_sync(operation))

    def _finish_sync(self, operation: str) -> None:
        self.sync_label.configure(text="Synced", foreground=Theme.GREEN)
        self.status_callback(f"{operation}: mock transfer complete.")

    def _center_map(self) -> None:
        self.map.reset_view()
        self.status_callback("Map centered on Home.")


class AttitudeIndicator(tk.Canvas):
    def __init__(self, master: tk.Misc, size: int = 210) -> None:
        super().__init__(
            master,
            width=size,
            height=size,
            background="#121719",
            highlightthickness=0,
        )
        self.size = size
        self.roll = 0.0
        self.pitch = 0.0
        self.heading = 0.0
        self.bind("<Configure>", lambda _event: self.render())

    def set_attitude(self, roll: float, pitch: float, heading: float) -> None:
        self.roll = roll
        self.pitch = pitch
        self.heading = heading
        self.render()

    def render(self) -> None:
        self.delete("all")
        size = min(self.winfo_width(), self.winfo_height())
        center = size / 2
        radius = size * 0.45
        pitch_offset = max(-radius * 0.55, min(radius * 0.55, self.pitch * 2.0))
        roll = math.radians(self.roll)
        direction = (math.cos(roll), math.sin(roll))
        normal = (-direction[1], direction[0])
        horizon_center = (
            center + normal[0] * pitch_offset,
            center + normal[1] * pitch_offset,
        )
        length = radius * 1.8
        x1 = horizon_center[0] - direction[0] * length
        y1 = horizon_center[1] - direction[1] * length
        x2 = horizon_center[0] + direction[0] * length
        y2 = horizon_center[1] + direction[1] * length

        self.create_oval(
            center - radius,
            center - radius,
            center + radius,
            center + radius,
            fill="#3b91bd",
            outline="#8f999e",
            width=3,
        )
        ground_points = [
            center - radius,
            center + radius,
            center - radius,
            max(center - radius, min(center + radius, y1)),
            center + radius,
            max(center - radius, min(center + radius, y2)),
            center + radius,
            center + radius,
        ]
        self.create_polygon(*ground_points, fill="#8c6545", outline="")
        self.create_line(x1, y1, x2, y2, fill="white", width=3)
        self.create_line(center - 30, center, center - 7, center, fill=Theme.ACCENT, width=4)
        self.create_line(center + 7, center, center + 30, center, fill=Theme.ACCENT, width=4)
        self.create_polygon(
            center - 7,
            center,
            center,
            center + 5,
            center + 7,
            center,
            fill=Theme.ACCENT,
            outline="",
        )
        self.create_text(
            center,
            13,
            text=f"{self.heading:03.0f}°",
            fill="white",
            font=("Segoe UI", 10, "bold"),
        )
        self.create_text(
            center,
            size - 12,
            text=f"ROLL {self.roll:+.1f}°   PITCH {self.pitch:+.1f}°",
            fill="#d5dde0",
            font=("Consolas", 8),
        )


class FlyView(tk.Frame):
    def __init__(
        self,
        master: tk.Misc,
        mission: MissionStore,
        vehicle: VehicleState,
        status_callback: Callable[[str], None],
    ) -> None:
        super().__init__(master, background=Theme.WINDOW)
        self.vehicle = vehicle
        self.status_callback = status_callback

        self.map = MapWidget(
            self,
            mission,
            vehicle,
            editable=False,
            status_callback=status_callback,
        )
        self.map.pack(fill=tk.BOTH, expand=True)

        action_panel = tk.Frame(self.map, background=Theme.TOOLBAR_ALT)
        action_panel.place(x=14, y=16, anchor="nw")
        actions = [
            ("ARM", self._toggle_arm),
            ("Takeoff", lambda: self._confirm_action("Takeoff")),
            ("Pause", lambda: self._confirm_action("Pause")),
            ("Return", lambda: self._confirm_action("Return to Launch")),
            ("Land", lambda: self._confirm_action("Land")),
        ]
        for label, command in actions:
            flat_button(
                action_panel,
                label,
                command,
                background=Theme.TOOLBAR_ALT,
                active_background="#3b474d",
                padx=16,
                pady=9,
            ).pack(fill=tk.X, pady=1)

        self.instrument_panel = tk.Frame(
            self.map,
            background="#161c1f",
            width=250,
        )
        self.instrument_panel.place(relx=1.0, x=-16, y=16, anchor="ne")
        self.instrument_panel.pack_propagate(False)

        tk.Label(
            self.instrument_panel,
            text="Vehicle 1",
            background="#161c1f",
            foreground=Theme.LIGHT_TEXT,
            font=("Segoe UI", 12, "bold"),
            pady=8,
        ).pack(fill=tk.X)
        self.attitude = AttitudeIndicator(self.instrument_panel, 220)
        self.attitude.pack(padx=10, pady=(0, 8))

        values = tk.Frame(self.instrument_panel, background="#161c1f")
        values.pack(fill=tk.X, padx=10, pady=(0, 10))
        self.value_labels: dict[str, tk.Label] = {}
        fields = (
            ("ALT REL", "altitude"),
            ("GROUND SPD", "speed"),
            ("VERT SPD", "vertical"),
            ("FLIGHT TIME", "time"),
        )
        for row, (title, key) in enumerate(fields):
            tk.Label(
                values,
                text=title,
                background="#161c1f",
                foreground="#89969c",
                font=("Segoe UI", 8),
                anchor="w",
            ).grid(row=row, column=0, sticky="w", pady=3)
            label = tk.Label(
                values,
                text="--",
                background="#161c1f",
                foreground="white",
                font=("Consolas", 11, "bold"),
                anchor="e",
            )
            label.grid(row=row, column=1, sticky="e", padx=(25, 0), pady=3)
            self.value_labels[key] = label
        values.columnconfigure(1, weight=1)

        video = tk.Frame(self.map, background="#111719", width=290, height=160)
        video.place(x=14, rely=1.0, y=-16, anchor="sw")
        video.pack_propagate(False)
        tk.Label(
            video,
            text="▣\nWAITING FOR VIDEO",
            background="#111719",
            foreground="#859299",
            font=("Segoe UI", 10, "bold"),
        ).pack(expand=True)

    def update_vehicle(self) -> None:
        vehicle = self.vehicle
        self.attitude.set_attitude(vehicle.roll_deg, vehicle.pitch_deg, vehicle.heading_deg)
        self.value_labels["altitude"].configure(text=f"{vehicle.relative_altitude_m:5.1f} m")
        self.value_labels["speed"].configure(text=f"{vehicle.ground_speed_mps:5.1f} m/s")
        self.value_labels["vertical"].configure(text=f"{vehicle.vertical_speed_mps:+5.1f} m/s")
        minutes, seconds = divmod(vehicle.flight_time_s, 60)
        self.value_labels["time"].configure(text=f"{minutes:02d}:{seconds:02d}")
        self.map.render()

    def _toggle_arm(self) -> None:
        action = "Disarm" if self.vehicle.armed else "Arm"
        self._confirm_action(action, arm_toggle=True)

    def _confirm_action(self, action: str, arm_toggle: bool = False) -> None:
        if not messagebox.askyesno(
            "Confirm vehicle action",
            f"{action}?\n\nThis prototype uses mock vehicle data only.",
        ):
            return
        if arm_toggle:
            self.vehicle.armed = not self.vehicle.armed
        self.status_callback(f"{action} confirmed (mock only).")


class AnalyzeView(tk.Frame):
    def __init__(self, master: tk.Misc, status_callback: Callable[[str], None]) -> None:
        super().__init__(master, background="#eef0f1")
        title = tk.Frame(self, background="#ffffff", height=74)
        title.pack(fill=tk.X)
        title.pack_propagate(False)
        tk.Label(
            title,
            text="Analyze Tools",
            background="#ffffff",
            foreground=Theme.TEXT,
            font=("Segoe UI", 22, "bold"),
        ).pack(side=tk.LEFT, padx=24)

        body = tk.Frame(self, background="#eef0f1")
        body.pack(fill=tk.BOTH, expand=True, padx=28, pady=26)
        tools = [
            ("Flight Log Download", "Download and manage onboard flight logs."),
            ("MAVLink Inspector", "Inspect live messages, fields and update rates."),
            ("Log Replay", "Replay recorded telemetry through the user interface."),
            ("GeoTag Images", "Attach vehicle positions to survey photographs."),
            ("MAVLink Console", "Open a developer console for supported vehicles."),
            ("Message Log", "Review warnings, errors and vehicle status text."),
        ]
        for index, (name, description) in enumerate(tools):
            card = tk.Frame(
                body,
                background="#ffffff",
                highlightbackground="#d3d7d9",
                highlightthickness=1,
                width=330,
                height=130,
            )
            card.grid(row=index // 3, column=index % 3, padx=8, pady=8, sticky="nsew")
            card.grid_propagate(False)
            tk.Label(
                card,
                text=name,
                background="#ffffff",
                foreground=Theme.TEXT,
                font=("Segoe UI", 12, "bold"),
                anchor="w",
            ).pack(fill=tk.X, padx=16, pady=(16, 4))
            tk.Label(
                card,
                text=description,
                background="#ffffff",
                foreground=Theme.MUTED,
                font=("Segoe UI", 9),
                justify=tk.LEFT,
                wraplength=285,
                anchor="nw",
            ).pack(fill=tk.BOTH, expand=True, padx=16)
            flat_button(
                card,
                "Open",
                lambda value=name: status_callback(f"{value}: coming in a later milestone."),
                background="#e1e5e7",
                foreground=Theme.TEXT,
                active_background="#ced4d7",
                padx=12,
                pady=4,
            ).pack(anchor="e", padx=12, pady=10)
        for column in range(3):
            body.columnconfigure(column, weight=1)
        for row in range(2):
            body.rowconfigure(row, weight=1)


class SettingsView(tk.Frame):
    def __init__(self, master: tk.Misc, status_callback: Callable[[str], None]) -> None:
        super().__init__(master, background="#eef0f1")
        navigation = tk.Frame(self, background=Theme.TOOLBAR_ALT, width=240)
        navigation.pack(side=tk.LEFT, fill=tk.Y)
        navigation.pack_propagate(False)
        tk.Label(
            navigation,
            text="Application Settings",
            background=Theme.TOOLBAR_ALT,
            foreground="white",
            font=("Segoe UI", 15, "bold"),
            padx=18,
            pady=20,
            anchor="w",
        ).pack(fill=tk.X)
        for label in ("General", "Comm Links", "Maps", "Video", "Telemetry", "Offline Maps"):
            flat_button(
                navigation,
                label,
                lambda value=label: status_callback(f"Settings: {value}"),
                background=Theme.TOOLBAR_ALT,
                active_background="#3c484e",
                padx=20,
                pady=11,
            ).pack(fill=tk.X)

        content = tk.Frame(self, background="#ffffff")
        content.pack(fill=tk.BOTH, expand=True, padx=1)
        tk.Label(
            content,
            text="General",
            background="#ffffff",
            foreground=Theme.TEXT,
            font=("Segoe UI", 22, "bold"),
        ).pack(anchor="w", padx=30, pady=(28, 12))
        form = tk.Frame(content, background="#ffffff")
        form.pack(fill=tk.X, padx=30)
        settings = (
            ("Application language", "Korean / English"),
            ("Map provider", "Offline Preview"),
            ("Distance units", "Metric"),
            ("Telemetry log", "Enabled"),
            ("Virtual joystick", "Disabled"),
        )
        for row, (label, value) in enumerate(settings):
            tk.Label(
                form,
                text=label,
                background="#ffffff",
                foreground=Theme.TEXT,
                font=("Segoe UI", 10),
                anchor="w",
            ).grid(row=row, column=0, sticky="w", pady=9)
            tk.Label(
                form,
                text=value,
                background="#f0f2f3",
                foreground=Theme.MUTED,
                font=("Segoe UI", 10),
                width=24,
                padx=10,
                pady=6,
                anchor="w",
            ).grid(row=row, column=1, sticky="e", padx=(50, 0), pady=9)
