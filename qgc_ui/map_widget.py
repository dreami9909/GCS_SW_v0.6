from __future__ import annotations

import base64
import math
import os
import queue
import threading
import tkinter as tk
import urllib.error
import urllib.parse
import urllib.request
from tkinter import ttk
from typing import TYPE_CHECKING, Callable

from .domain import MissionCommand, MissionStore, VehicleState

if TYPE_CHECKING:
    from .tactical import TacticalState, SubjectTrack


COLORS = {
    "map": "#111915",
    "map_grid": "#3c5a4b",
    "minor_road": "#ece9df",
    "major_road": "#f7f2e6",
    "road_edge": "#bab7ad",
    "water": "#9bc4d0",
    "park": "#b9d2ad",
    "route": "#cc7a00",
    "route_shadow": "#fff3d9",
    "selected": "#ffd84d",
    "waypoint": "#f4a000",
    "home": "#28a55f",
    "vehicle": "#d93b33",
    "text": "#202830",
    "team": "#55b8ff",
    "subject": "#ff5349",
    "neutral": "#d6c568",
    "safe_zone": "#58bd75",
    "approach": "#ffbe38",
}


class MapWidget(ttk.Frame):
    """Offline vector-style map surface used while the real map provider is absent."""

    def __init__(
        self,
        master: tk.Misc,
        mission: MissionStore,
        vehicle: VehicleState,
        *,
        editable: bool,
        command_provider: Callable[[], MissionCommand | None] | None = None,
        status_callback: Callable[[str], None] | None = None,
        tactical_state: TacticalState | None = None,
        google_maps_api_key: str | None = None,
    ) -> None:
        super().__init__(master)
        self.mission = mission
        self.vehicle = vehicle
        self.editable = editable
        self.command_provider = command_provider or (lambda: None)
        self.status_callback = status_callback or (lambda _message: None)
        self.tactical_state = tactical_state
        self.center_lat = mission.home_latitude
        self.center_lon = mission.home_longitude
        self.span_lat = 0.021
        self._drag_sequence: int | None = None
        self._pan_origin: tuple[int, int] | None = None
        self._pan_center: tuple[float, float] | None = None
        self._marker_positions: dict[int, tuple[float, float]] = {}
        self._subject_positions: dict[int, tuple[float, float]] = {}
        self.google_maps_api_key = google_maps_api_key or os.getenv(
            "GOOGLE_MAPS_API_KEY", ""
        ).strip()
        self._google_photo: tk.PhotoImage | None = None
        self._google_signature: tuple[float, float, int, int, int] | None = None
        self._google_loaded_signature: tuple[float, float, int, int, int] | None = None
        self._google_request_job: str | None = None
        self._google_inflight = False
        self._google_results: queue.Queue[
            tuple[tuple[float, float, int, int, int], bytes | None, str]
        ] = queue.Queue()

        self.canvas = tk.Canvas(
            self,
            background=COLORS["map"],
            highlightthickness=0,
            cursor="crosshair" if editable else "arrow",
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        controls = tk.Frame(self.canvas, background="#2b3339", bd=0)
        controls.place(relx=1.0, x=-14, y=14, anchor="ne")
        self._map_button(controls, "+", lambda: self.zoom(0.78)).pack(fill=tk.X)
        self._map_button(controls, "-", lambda: self.zoom(1.28)).pack(
            fill=tk.X, pady=(1, 0)
        )
        self._map_button(controls, "C", self.reset_view).pack(fill=tk.X, pady=(1, 0))
        self.scale_label = tk.Label(
            controls,
            text="GRID 000.0 km",
            background="#101713",
            foreground="#92e6aa",
            font=("Consolas", 8, "bold"),
            padx=7,
            pady=5,
        )
        self.scale_label.pack(fill=tk.X, pady=(4, 0))

        self.canvas.bind("<Configure>", self._on_configure)
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        self.canvas.bind("<Button-1>", self._on_left_down)
        self.canvas.bind("<Double-Button-1>", self._on_double_click)
        self.canvas.bind("<B1-Motion>", self._on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_left_up)
        self.canvas.bind("<ButtonPress-3>", self._on_pan_start)
        self.canvas.bind("<B3-Motion>", self._on_pan_move)
        self.canvas.bind("<ButtonRelease-3>", self._on_pan_end)

        mission.subscribe(self.render)
        if self.google_maps_api_key:
            self.after(160, self._poll_google_results)

    @staticmethod
    def _map_button(master: tk.Misc, text: str, command: Callable[[], None]) -> tk.Button:
        return tk.Button(
            master,
            text=text,
            command=command,
            width=3,
            font=("Segoe UI", 11, "bold"),
            foreground="white",
            background="#2b3339",
            activeforeground="white",
            activebackground="#46535b",
            relief=tk.FLAT,
            bd=0,
        )

    @property
    def span_lon(self) -> float:
        width = max(self.canvas.winfo_width(), 1)
        height = max(self.canvas.winfo_height(), 1)
        latitude_correction = max(math.cos(math.radians(self.center_lat)), 0.2)
        return self.span_lat * (width / height) / latitude_correction

    @property
    def grid_distance_km(self) -> float:
        return self.span_lat * 111.32 / 8

    def _google_zoom(self) -> int:
        height = max(self.canvas.winfo_height(), 256)
        estimate = math.log2(360.0 * height / (256.0 * max(self.span_lat, 0.0001)))
        return max(2, min(21, round(estimate)))

    def _on_configure(self, _event: tk.Event) -> None:
        self.render()
        self._schedule_google_map()

    def _schedule_google_map(self) -> None:
        if not self.google_maps_api_key:
            return
        if self._google_request_job is not None:
            self.after_cancel(self._google_request_job)
        self._google_request_job = self.after(350, self._request_google_map)

    def _request_google_map(self) -> None:
        self._google_request_job = None
        if self._google_inflight or not self.google_maps_api_key:
            return
        canvas_width = max(self.canvas.winfo_width(), 360)
        canvas_height = max(self.canvas.winfo_height(), 240)
        request_width = max(180, min(640, math.ceil(canvas_width / 2)))
        request_height = max(180, min(640, math.ceil(canvas_height / 2)))
        signature = (
            round(self.center_lat, 6),
            round(self.center_lon, 6),
            self._google_zoom(),
            request_width,
            request_height,
        )
        self._google_signature = signature
        if signature == self._google_loaded_signature:
            return
        self._google_inflight = True

        def worker() -> None:
            parameters = {
                "center": f"{signature[0]},{signature[1]}",
                "zoom": str(signature[2]),
                "size": f"{signature[3]}x{signature[4]}",
                "scale": "2",
                "maptype": "hybrid",
                "format": "png",
                "key": self.google_maps_api_key,
            }
            url = (
                "https://maps.googleapis.com/maps/api/staticmap?"
                + urllib.parse.urlencode(parameters)
            )
            try:
                request = urllib.request.Request(
                    url,
                    headers={"User-Agent": "PythonGroundControl/0.2"},
                )
                with urllib.request.urlopen(request, timeout=12) as response:
                    data = response.read()
                self._google_results.put((signature, data, ""))
            except (OSError, urllib.error.URLError) as error:
                self._google_results.put((signature, None, str(error)))

        threading.Thread(target=worker, daemon=True).start()

    def _poll_google_results(self) -> None:
        try:
            while True:
                signature, image_data, error = self._google_results.get_nowait()
                self._google_inflight = False
                if image_data is not None and signature == self._google_signature:
                    try:
                        encoded = base64.b64encode(image_data)
                        self._google_photo = tk.PhotoImage(data=encoded)
                        self._google_loaded_signature = signature
                        self.render()
                        self.status_callback("Google Maps background loaded.")
                    except tk.TclError as image_error:
                        self.status_callback(f"Google Maps image error: {image_error}")
                elif error:
                    self.status_callback(
                        "Google Maps unavailable; offline tactical grid is active."
                    )
        except queue.Empty:
            pass
        if self.google_maps_api_key and self.winfo_exists():
            self.after(160, self._poll_google_results)

    def latlon_to_xy(self, latitude: float, longitude: float) -> tuple[float, float]:
        width = max(self.canvas.winfo_width(), 1)
        height = max(self.canvas.winfo_height(), 1)
        x = (longitude - self.center_lon) / self.span_lon * width + width / 2
        y = (self.center_lat - latitude) / self.span_lat * height + height / 2
        return x, y

    def xy_to_latlon(self, x: float, y: float) -> tuple[float, float]:
        width = max(self.canvas.winfo_width(), 1)
        height = max(self.canvas.winfo_height(), 1)
        longitude = (x - width / 2) / width * self.span_lon + self.center_lon
        latitude = self.center_lat - (y - height / 2) / height * self.span_lat
        return latitude, longitude

    def zoom(self, factor: float) -> None:
        self.span_lat = max(0.001, min(1.5, self.span_lat * factor))
        self.render()
        self._schedule_google_map()

    def reset_view(self) -> None:
        self.center_lat = self.mission.home_latitude
        self.center_lon = self.mission.home_longitude
        self.span_lat = 0.021
        self.render()
        self._schedule_google_map()

    def render(self) -> None:
        canvas = self.canvas
        width = max(canvas.winfo_width(), 1)
        height = max(canvas.winfo_height(), 1)
        canvas.delete("map")
        self._marker_positions.clear()
        self._subject_positions.clear()
        self._draw_base_map(width, height)
        self._draw_route()
        self._draw_home()
        for waypoint in self.mission.waypoints:
            self._draw_waypoint(waypoint)
        self._draw_vehicle()
        self._draw_tactical_overlays()
        self._draw_map_labels(width, height)
        self.scale_label.configure(text=f"GRID {self.grid_distance_km:05.1f} km")

    def _draw_base_map(self, width: int, height: int) -> None:
        canvas = self.canvas
        canvas.create_rectangle(0, 0, width, height, fill=COLORS["map"], outline="", tags="map")

        if self._google_photo is not None:
            canvas.create_image(
                width / 2,
                height / 2,
                image=self._google_photo,
                anchor="center",
                tags="map",
            )
            self._draw_tactical_grid(width, height)
            return

        park = [
            width * 0.07,
            height * 0.12,
            width * 0.35,
            height * 0.05,
            width * 0.42,
            height * 0.33,
            width * 0.16,
            height * 0.39,
        ]
        canvas.create_polygon(
            *park,
            fill=COLORS["park"],
            outline="#9eb993",
            width=2,
            tags="map",
        )
        canvas.create_polygon(
            width * 0.78,
            0,
            width,
            0,
            width,
            height * 0.45,
            width * 0.88,
            height * 0.39,
            width * 0.82,
            height * 0.18,
            fill=COLORS["water"],
            outline="",
            tags="map",
        )

        grid_step = max(55, int(min(width, height) / 9))
        for x in range(-grid_step, width + grid_step, grid_step):
            canvas.create_line(
                x,
                0,
                x + height * 0.18,
                height,
                fill=COLORS["map_grid"],
                width=1,
                tags="map",
            )
        for y in range(-grid_step, height + grid_step, grid_step):
            canvas.create_line(
                0,
                y,
                width,
                y - width * 0.05,
                fill=COLORS["map_grid"],
                width=1,
                tags="map",
            )

        major_roads = [
            (-30, height * 0.72, width + 30, height * 0.35),
            (width * 0.42, -20, width * 0.56, height + 20),
        ]
        for x1, y1, x2, y2 in major_roads:
            canvas.create_line(
                x1,
                y1,
                x2,
                y2,
                fill=COLORS["road_edge"],
                width=12,
                tags="map",
            )
            canvas.create_line(
                x1,
                y1,
                x2,
                y2,
                fill=COLORS["major_road"],
                width=9,
                tags="map",
            )
            canvas.create_line(
                x1,
                y1,
                x2,
                y2,
                fill="#e0b94f",
                width=1,
                dash=(10, 8),
                tags="map",
            )

        labels = [
            (0.18, 0.2, "Operations Area"),
            (0.58, 0.22, "Mission Field"),
            (0.69, 0.72, "Runway 01"),
            (0.88, 0.16, "Reservoir"),
        ]
        for rel_x, rel_y, label in labels:
            canvas.create_text(
                width * rel_x,
                height * rel_y,
                text=label,
                fill="#777a75",
                font=("Segoe UI", 9),
                tags="map",
            )
        self._draw_tactical_grid(width, height)

    def _draw_tactical_grid(self, width: int, height: int) -> None:
        for index in range(1, 8):
            x = width * index / 8
            y = height * index / 8
            self.canvas.create_line(
                x,
                0,
                x,
                height,
                fill=COLORS["map_grid"],
                width=1,
                dash=(3, 6),
                tags="map",
            )
            self.canvas.create_line(
                0,
                y,
                width,
                y,
                fill=COLORS["map_grid"],
                width=1,
                dash=(3, 6),
                tags="map",
            )

    def _draw_route(self) -> None:
        points = [self.latlon_to_xy(self.mission.home_latitude, self.mission.home_longitude)]
        points.extend(
            self.latlon_to_xy(waypoint.latitude, waypoint.longitude)
            for waypoint in self.mission.waypoints
        )
        if len(points) < 2:
            return
        flattened = [coordinate for point in points for coordinate in point]
        self.canvas.create_line(
            *flattened,
            fill=COLORS["route_shadow"],
            width=7,
            joinstyle=tk.ROUND,
            tags="map",
        )
        self.canvas.create_line(
            *flattened,
            fill=COLORS["route"],
            width=3,
            joinstyle=tk.ROUND,
            tags="map",
        )

    def _draw_home(self) -> None:
        x, y = self.latlon_to_xy(self.mission.home_latitude, self.mission.home_longitude)
        self.canvas.create_oval(
            x - 14,
            y - 14,
            x + 14,
            y + 14,
            fill=COLORS["home"],
            outline="white",
            width=2,
            tags="map",
        )
        self.canvas.create_text(
            x,
            y,
            text="H",
            fill="white",
            font=("Segoe UI", 10, "bold"),
            tags="map",
        )

    def _draw_waypoint(self, waypoint: object) -> None:
        x, y = self.latlon_to_xy(waypoint.latitude, waypoint.longitude)
        self._marker_positions[waypoint.sequence] = (x, y)
        selected = waypoint.sequence == self.mission.selected_sequence
        radius = 14 if selected else 11
        fill = COLORS["selected"] if selected else COLORS["waypoint"]
        self.canvas.create_oval(
            x - radius,
            y - radius,
            x + radius,
            y + radius,
            fill=fill,
            outline="#5b4300",
            width=3 if selected else 1,
            tags="map",
        )
        self.canvas.create_text(
            x,
            y,
            text=str(waypoint.sequence),
            fill="#252525",
            font=("Segoe UI", 9, "bold"),
            tags="map",
        )

    def _draw_vehicle(self) -> None:
        x, y = self.latlon_to_xy(self.vehicle.latitude, self.vehicle.longitude)
        angle = math.radians(self.vehicle.heading_deg - 90)
        radius = 17
        tip = (x + math.cos(angle) * radius, y + math.sin(angle) * radius)
        left = (
            x + math.cos(angle + 2.45) * radius * 0.72,
            y + math.sin(angle + 2.45) * radius * 0.72,
        )
        right = (
            x + math.cos(angle - 2.45) * radius * 0.72,
            y + math.sin(angle - 2.45) * radius * 0.72,
        )
        self.canvas.create_polygon(
            *tip,
            *left,
            x,
            y,
            *right,
            fill=COLORS["team"],
            outline="white",
            width=2,
            tags="map",
        )
        self.canvas.create_rectangle(
            x - 22,
            y - 19,
            x + 22,
            y + 23,
            outline=COLORS["team"],
            width=2,
            tags="map",
        )
        self.canvas.create_text(
            x,
            y + 31,
            text="SAR-01",
            fill="#d6efff",
            font=("Consolas", 8, "bold"),
            tags="map",
        )

    def _draw_tactical_overlays(self) -> None:
        state = self.tactical_state
        if state is None:
            return

        launcher = next((site for site in state.sites if site.code == "LC"), None)
        if launcher is not None:
            center_x, center_y = self.latlon_to_xy(
                launcher.latitude, launcher.longitude
            )
            edge_x, _ = self.latlon_to_xy(
                launcher.latitude,
                launcher.longitude + 0.006,
            )
            radius = max(abs(edge_x - center_x), 46)
            self.canvas.create_oval(
                center_x - radius,
                center_y - radius,
                center_x + radius,
                center_y + radius,
                outline=COLORS["safe_zone"],
                width=2,
                dash=(8, 6),
                tags="map",
            )
            self.canvas.create_text(
                center_x,
                center_y - radius - 10,
                text="SAFE ZONE",
                fill="#98efad",
                font=("Consolas", 8, "bold"),
                tags="map",
            )

        for site in state.sites:
            self._draw_site_symbol(site.code, site.label, site.latitude, site.longitude)

        for subject in state.subjects:
            self._draw_subject_symbol(subject)

        selected = state.selected_subject
        if selected is not None:
            lead_seconds = 18.0
            heading = math.radians(selected.heading_deg)
            north_m = math.cos(heading) * selected.speed_mps * lead_seconds
            east_m = math.sin(heading) * selected.speed_mps * lead_seconds
            approach_lat = selected.latitude + north_m / 111_320.0
            longitude_scale = max(
                math.cos(math.radians(selected.latitude)) * 111_320.0,
                20_000.0,
            )
            approach_lon = selected.longitude + east_m / longitude_scale
            vehicle_x, vehicle_y = self.latlon_to_xy(
                self.vehicle.latitude, self.vehicle.longitude
            )
            approach_x, approach_y = self.latlon_to_xy(
                approach_lat, approach_lon
            )
            self.canvas.create_line(
                vehicle_x,
                vehicle_y,
                approach_x,
                approach_y,
                fill=COLORS["approach"],
                width=2,
                dash=(9, 5),
                arrow=tk.LAST,
                tags="map",
            )
            self.canvas.create_oval(
                approach_x - 11,
                approach_y - 11,
                approach_x + 11,
                approach_y + 11,
                outline=COLORS["approach"],
                width=2,
                tags="map",
            )
            self.canvas.create_line(
                approach_x - 15,
                approach_y,
                approach_x + 15,
                approach_y,
                fill=COLORS["approach"],
                tags="map",
            )
            self.canvas.create_line(
                approach_x,
                approach_y - 15,
                approach_x,
                approach_y + 15,
                fill=COLORS["approach"],
                tags="map",
            )
            self.canvas.create_text(
                approach_x + 18,
                approach_y - 13,
                anchor="w",
                text="PREDICTED APPROACH",
                fill="#ffd57a",
                font=("Consolas", 8, "bold"),
                tags="map",
            )

    def _draw_site_symbol(
        self,
        code: str,
        label: str,
        latitude: float,
        longitude: float,
    ) -> None:
        x, y = self.latlon_to_xy(latitude, longitude)
        self.canvas.create_rectangle(
            x - 22,
            y - 14,
            x + 22,
            y + 14,
            outline=COLORS["team"],
            fill="#102531",
            width=2,
            tags="map",
        )
        self.canvas.create_line(
            x - 22,
            y - 14,
            x,
            y - 23,
            x + 22,
            y - 14,
            fill=COLORS["team"],
            width=2,
            tags="map",
        )
        self.canvas.create_text(
            x,
            y,
            text=code,
            fill="#d9f3ff",
            font=("Consolas", 8, "bold"),
            tags="map",
        )
        self.canvas.create_text(
            x,
            y + 24,
            text=label,
            fill="#b7deef",
            font=("Malgun Gothic", 8),
            tags="map",
        )

    def _draw_subject_symbol(self, subject: SubjectTrack) -> None:
        x, y = self.latlon_to_xy(subject.latitude, subject.longitude)
        self._subject_positions[subject.track_id] = (x, y)
        selected = (
            self.tactical_state is not None
            and subject.track_id == self.tactical_state.selected_track_id
        )
        radius = 17 if selected else 13
        color = "#ffcf45" if selected else COLORS["subject"]
        self.canvas.create_polygon(
            x,
            y - radius,
            x + radius,
            y,
            x,
            y + radius,
            x - radius,
            y,
            fill="#32120f",
            outline=color,
            width=3 if selected else 2,
            tags="map",
        )
        heading = math.radians(subject.heading_deg - 90)
        self.canvas.create_line(
            x,
            y,
            x + math.cos(heading) * 29,
            y + math.sin(heading) * 29,
            fill=color,
            width=2,
            arrow=tk.LAST,
            tags="map",
        )
        self.canvas.create_text(
            x + 20,
            y - 18,
            anchor="w",
            text=f"PSN-{subject.track_id}",
            fill=color,
            font=("Consolas", 8, "bold"),
            tags="map",
        )

    def _draw_map_labels(self, width: int, height: int) -> None:
        self.canvas.create_rectangle(
            12,
            height - 43,
            218,
            height - 12,
            fill="#ffffff",
            outline="#a9a9a9",
            tags="map",
        )
        self.canvas.create_text(
            24,
            height - 28,
            anchor="w",
            text=f"{self.center_lat:.5f}, {self.center_lon:.5f}",
            fill=COLORS["text"],
            font=("Consolas", 9),
            tags="map",
        )
        self.canvas.create_text(
            width - 14,
            height - 15,
            anchor="se",
            text=(
                "GOOGLE MAPS // HYBRID"
                if self._google_photo is not None
                else "OFFLINE TACTICAL GRID // GOOGLE KEY REQUIRED"
            ),
            fill="#aab7ad",
            font=("Consolas", 8, "bold"),
            tags="map",
        )

    def _nearest_waypoint(self, x: float, y: float) -> int | None:
        nearest: int | None = None
        nearest_distance = 18.0
        for sequence, position in self._marker_positions.items():
            distance = math.hypot(position[0] - x, position[1] - y)
            if distance < nearest_distance:
                nearest = sequence
                nearest_distance = distance
        return nearest

    def _on_left_down(self, event: tk.Event) -> None:
        sequence = self._nearest_waypoint(event.x, event.y)
        if sequence is not None:
            self.mission.select(sequence)
            if self.editable:
                self._drag_sequence = sequence
            return
        if not self.editable:
            return
        command = self.command_provider()
        if command is None:
            self.status_callback("Select a mission command, then click the map.")
            return
        latitude, longitude = self.xy_to_latlon(event.x, event.y)
        waypoint = self.mission.add_waypoint(latitude, longitude, command)
        self.status_callback(f"Added {waypoint.command.value} #{waypoint.sequence}")

    def _on_double_click(self, event: tk.Event) -> None:
        if self.tactical_state is None:
            return
        nearest_track: int | None = None
        nearest_distance = 26.0
        for track_id, position in self._subject_positions.items():
            distance = math.hypot(position[0] - event.x, position[1] - event.y)
            if distance < nearest_distance:
                nearest_track = track_id
                nearest_distance = distance
        if nearest_track is not None:
            self.tactical_state.select_subject(nearest_track)
            self.status_callback(f"Subject PSN-{nearest_track} selected.")
            self.render()

    def _on_left_drag(self, event: tk.Event) -> None:
        if not self.editable or self._drag_sequence is None:
            return
        latitude, longitude = self.xy_to_latlon(event.x, event.y)
        self.mission.move(self._drag_sequence, latitude, longitude)

    def _on_left_up(self, _event: tk.Event) -> None:
        self._drag_sequence = None

    def _on_pan_start(self, event: tk.Event) -> None:
        self._pan_origin = (event.x, event.y)
        self._pan_center = (self.center_lat, self.center_lon)
        self.canvas.configure(cursor="fleur")

    def _on_pan_move(self, event: tk.Event) -> None:
        if self._pan_origin is None or self._pan_center is None:
            return
        width = max(self.canvas.winfo_width(), 1)
        height = max(self.canvas.winfo_height(), 1)
        delta_x = event.x - self._pan_origin[0]
        delta_y = event.y - self._pan_origin[1]
        self.center_lat = self._pan_center[0] + delta_y / height * self.span_lat
        self.center_lon = self._pan_center[1] - delta_x / width * self.span_lon
        self.render()

    def _on_pan_end(self, _event: tk.Event) -> None:
        self._pan_origin = None
        self._pan_center = None
        self.canvas.configure(cursor="crosshair" if self.editable else "arrow")
        self._schedule_google_map()

    def _on_mouse_wheel(self, event: tk.Event) -> None:
        self.zoom(0.86 if event.delta > 0 else 1.16)
