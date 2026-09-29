from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable


class Theme:
    WINDOW = "#07100d"
    TOOLBAR = "#0c1510"
    TOOLBAR_ALT = "#152018"
    PANEL = "#f4f4f4"
    PANEL_ALT = "#e8eaeb"
    BORDER = "#bcc1c3"
    TEXT = "#263238"
    MUTED = "#68747a"
    LIGHT_TEXT = "#ecf0f1"
    ACCENT = "#e7ad27"
    ACCENT_DARK = "#c89200"
    GREEN = "#39d768"
    RED = "#ed4a42"
    BLUE = "#4186b7"


def flat_button(
    master: tk.Misc,
    text: str,
    command: Callable[[], None],
    *,
    background: str = Theme.TOOLBAR_ALT,
    foreground: str = Theme.LIGHT_TEXT,
    active_background: str | None = None,
    font: tuple = ("Segoe UI", 10),
    padx: int = 12,
    pady: int = 8,
) -> tk.Button:
    return tk.Button(
        master,
        text=text,
        command=command,
        background=background,
        foreground=foreground,
        activebackground=active_background or background,
        activeforeground=foreground,
        relief=tk.FLAT,
        borderwidth=0,
        highlightthickness=0,
        font=font,
        padx=padx,
        pady=pady,
        cursor="hand2",
    )


class StatusChip(tk.Frame):
    def __init__(
        self,
        master: tk.Misc,
        title: str,
        value: str = "--",
        *,
        width: int = 90,
    ) -> None:
        super().__init__(master, background=Theme.TOOLBAR)
        self.configure(width=width, height=48)
        self.pack_propagate(False)
        tk.Label(
            self,
            text=title,
            background=Theme.TOOLBAR,
            foreground="#8f9ba1",
            font=("Segoe UI", 7),
        ).pack(anchor="center")
        self.value_label = tk.Label(
            self,
            text=value,
            background=Theme.TOOLBAR,
            foreground=Theme.LIGHT_TEXT,
            font=("Segoe UI", 10, "bold"),
        )
        self.value_label.pack(anchor="center")

    def set(self, value: str, color: str | None = None) -> None:
        self.value_label.configure(text=value, foreground=color or Theme.LIGHT_TEXT)


class ToolButton(tk.Frame):
    def __init__(
        self,
        master: tk.Misc,
        icon: str,
        label: str,
        command: Callable[[], None],
        *,
        width: int = 78,
    ) -> None:
        super().__init__(master, background=Theme.TOOLBAR_ALT, width=width, height=64)
        self.pack_propagate(False)
        self._command = command
        self.icon_label = tk.Label(
            self,
            text=icon,
            background=Theme.TOOLBAR_ALT,
            foreground=Theme.LIGHT_TEXT,
            font=("Segoe UI Symbol", 17),
        )
        self.icon_label.pack(pady=(7, 0))
        self.text_label = tk.Label(
            self,
            text=label,
            background=Theme.TOOLBAR_ALT,
            foreground=Theme.LIGHT_TEXT,
            font=("Segoe UI", 8),
        )
        self.text_label.pack()
        for widget in (self, self.icon_label, self.text_label):
            widget.bind("<Button-1>", lambda _event: self._command())
            widget.bind("<Enter>", lambda _event: self.set_hover(True))
            widget.bind("<Leave>", lambda _event: self.set_hover(False))

    def set_hover(self, hover: bool) -> None:
        color = "#374248" if hover else Theme.TOOLBAR_ALT
        self.configure(background=color)
        self.icon_label.configure(background=color)
        self.text_label.configure(background=color)


def configure_ttk_styles(root: tk.Misc) -> None:
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass
    style.configure(
        "QGC.Treeview",
        background="#ffffff",
        fieldbackground="#ffffff",
        foreground=Theme.TEXT,
        rowheight=29,
        borderwidth=0,
        font=("Segoe UI", 9),
    )
    style.configure(
        "QGC.Treeview.Heading",
        background="#d9dddf",
        foreground=Theme.TEXT,
        font=("Segoe UI", 9, "bold"),
        relief=tk.FLAT,
    )
    style.map(
        "QGC.Treeview",
        background=[("selected", "#f6d979")],
        foreground=[("selected", "#222222")],
    )
    style.configure(
        "QGC.TNotebook",
        background=Theme.PANEL,
        borderwidth=0,
        tabmargins=0,
    )
    style.configure(
        "QGC.TNotebook.Tab",
        background="#d8dcde",
        foreground=Theme.TEXT,
        padding=(14, 8),
        font=("Segoe UI", 9, "bold"),
    )
    style.map(
        "QGC.TNotebook.Tab",
        background=[("selected", Theme.PANEL)],
        foreground=[("selected", "#000000")],
    )
