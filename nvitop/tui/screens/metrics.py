# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
# pylint: disable=invalid-name

from __future__ import annotations

import itertools
import threading
import time
from collections import OrderedDict
from typing import TYPE_CHECKING, ClassVar

from nvitop.tui.library import (
    HOSTNAME,
    IS_SUPERUSER,
    IS_WINDOWS,
    NA,
    USER_CONTEXT,
    USERNAME,
    BufferedHistoryGraph,
    GpuProcess,
    HistoryGraph,
    Selection,
    WideString,
    cut_string,
    wcslen,
)
from nvitop.tui.screens.base import BaseSelectableScreen


if TYPE_CHECKING:
    import curses
    from collections.abc import Callable

    from nvitop.tui.tui import TUI


__all__ = ['ProcessMetricsScreen']


# pylint: disable-next=too-many-branches,too-many-locals
def get_yticks(history: HistoryGraph, y_offset: int) -> list[tuple[int, int]]:
    height = history.height
    baseline = history.baseline
    bound = history.bound
    max_bound = history.max_bound
    scale: float = history.scale  # type: ignore[attr-defined]
    upsidedown = history.upsidedown

    def p2h_f(p: int) -> float:
        return 0.01 * scale * p * (max_bound - baseline) * (height - 1) / (bound - baseline)

    max_height = height - 2
    percentages = (1, 2, 4, 5, 8, 10, 20, 40, 50, 80, 100, 200, 400, 500, 800, 1000)
    h2p = {}
    p2h = {}
    h2e = {}
    for p in percentages:
        h_f = p2h_f(p)
        p2h[p] = h = int(h_f)
        if h not in h2p:
            if h < max_height:
                h2p[h] = p
                h2e[h] = abs(h_f - h) / p
        elif abs(h_f - h) / p < h2e[h]:
            h2p[h] = p
            h2e[h] = abs(h_f - h) / p
    h2p = sorted(h2p.items())
    ticks = []
    if len(h2p) >= 2:
        (hm1, pm1), (h2, p2) = h2p[-2:]
        if height < 12:
            ticks = [(hm1, pm1)] if h2e[hm1] < h2e[h2] else [(h2, p2)]
        else:
            ticks = [(h2, p2)]
            if p2 % 2 == 0:
                p1 = p2 // 2
                h1 = int(p2h_f(p1))
                p3 = 3 * p1
                h3 = int(p2h_f(p3))
                if p1 >= 3:
                    ticks.append((h1, p1))
                    if h2 < h3 < max_height:
                        ticks.append((h3, p3))
    else:
        ticks = list(h2p)
    if not upsidedown:
        ticks = [(height - 1 - h, p) for h, p in ticks]
    return [(h + y_offset, p) for h, p in ticks]


class ProcessMetricsScreen(BaseSelectableScreen):  # pylint: disable=too-many-instance-attributes
    NAME: ClassVar[str] = 'process-metrics'
    SNAPSHOT_INTERVAL: ClassVar[float] = 0.5

    def __init__(self, *, win: curses.window, root: TUI) -> None:
        super().__init__(win, root)

        self.selection: Selection = Selection(self)
        self.npu_utilization: HistoryGraph | None = None
        self.ai_core_utilization: HistoryGraph | None = None
        self.vector_core_utilization: HistoryGraph | None = None
        self.memory_bandwidth_utilization: HistoryGraph | None = None

        self.enabled: bool = False
        self.snapshot_lock = threading.Lock()
        self._snapshot_daemon = threading.Thread(
            name='process-metrics-snapshot-daemon',
            target=self._snapshot_target,
            daemon=True,
        )
        self._daemon_running = threading.Event()

        self.x, self.y = root.x, root.y
        self.width, self.height = root.width, root.height
        self.left_width: int = max(20, (self.width - 3) // 2)
        self.right_width: int = max(20, (self.width - 2) // 2)
        self.upper_height: int = max(5, (self.height - 5 - 3) // 2)
        self.lower_height: int = max(5, (self.height - 5 - 2) // 2)

    @property
    def visible(self) -> bool:
        return self._visible

    @visible.setter
    def visible(self, value: bool) -> None:
        if self._visible != value:
            self.need_redraw = True
            self._visible = value
        if self.visible:
            self._daemon_running.set()
            try:
                self._snapshot_daemon.start()
            except RuntimeError:
                pass
            self.take_snapshots()
        else:
            self.focused = False

    def enable(self, state: bool = True) -> None:
        if not self.selection.is_set() or not state:
            self.disable()
            return

        def format_percent(label: str) -> Callable[[float], str]:
            def formatter(value: float) -> str:
                if value is NA:  # type: ignore[comparison-overlap]
                    return f'{label}: {value}'  # type: ignore[unreachable]
                return f'{label}: {value:.1f}%'

            return formatter

        def format_max_percent(label: str) -> Callable[[float], str]:
            def formatter(value: float) -> str:
                if value is NA:  # type: ignore[comparison-overlap]
                    return f'MAX {label}: {value}'  # type: ignore[unreachable]
                return f'MAX {label}: {value:.1f}%'

            return formatter

        def format_npu(value: float) -> str:
            if value is NA:  # type: ignore[comparison-overlap]
                return f'NPU: {value}'  # type: ignore[unreachable]
            return f'NPU: {value:.1f}%'

        def format_max_npu(value: float) -> str:
            if value is NA:  # type: ignore[comparison-overlap]
                return f'MAX NPU: {value}'  # type: ignore[unreachable]
            return f'MAX NPU: {value:.1f}%'

        with self.snapshot_lock:
            self.npu_utilization = BufferedHistoryGraph(
                interval=1.0,
                upperbound=100.0,
                width=self.left_width,
                height=self.upper_height,
                baseline=0.0,
                upsidedown=False,
                dynamic_bound=True,
                min_bound=10.0,
                init_bound=100.0,
                format=format_npu,
                max_format=format_max_npu,
            )
            self.ai_core_utilization = BufferedHistoryGraph(
                interval=1.0,
                upperbound=100.0,
                width=self.left_width,
                height=self.lower_height,
                baseline=0.0,
                upsidedown=True,
                dynamic_bound=True,
                min_bound=10.0,
                init_bound=100.0,
                format=format_percent('CUBE'),
                max_format=format_max_percent('CUBE'),
            )
            self.vector_core_utilization = BufferedHistoryGraph(
                interval=1.0,
                upperbound=100.0,
                width=self.right_width,
                height=self.upper_height,
                baseline=0.0,
                upsidedown=False,
                dynamic_bound=True,
                min_bound=10.0,
                init_bound=100.0,
                format=format_percent('VEC'),
                max_format=format_max_percent('VEC'),
            )
            self.memory_bandwidth_utilization = BufferedHistoryGraph(
                interval=1.0,
                upperbound=100.0,
                width=self.right_width,
                height=self.lower_height,
                baseline=0.0,
                upsidedown=True,
                dynamic_bound=True,
                min_bound=10.0,
                init_bound=100.0,
                format=format_percent('MBW'),
                max_format=format_max_percent('MBW'),
            )
            self.npu_utilization.scale = 1.0  # type: ignore[attr-defined]
            self.ai_core_utilization.scale = 1.0  # type: ignore[attr-defined]
            self.vector_core_utilization.scale = 1.0  # type: ignore[attr-defined]
            self.memory_bandwidth_utilization.scale = 1.0  # type: ignore[attr-defined]

            self._daemon_running.set()
            try:
                self._snapshot_daemon.start()
            except RuntimeError:
                pass
            self.enabled = True

        self.take_snapshots()
        self.update_size()

    def disable(self) -> None:
        with self.snapshot_lock:
            self._daemon_running.clear()
            self.enabled = False
            self.npu_utilization = None
            self.ai_core_utilization = None
            self.vector_core_utilization = None
            self.memory_bandwidth_utilization = None

    @property
    def process(self) -> GpuProcess:
        return self.selection.process  # type: ignore[return-value]

    @process.setter
    def process(self, value: GpuProcess) -> None:
        self.selection.process = value
        self.enable()

    @classmethod
    def set_snapshot_interval(cls, interval: float) -> None:
        assert interval > 0.0
        interval = float(interval)

        cls.SNAPSHOT_INTERVAL = min(interval / 3.0, 1.0)

    def take_snapshots(self) -> None:
        with self.snapshot_lock:
            if not self.selection.is_set() or not self.enabled:
                return

            with GpuProcess.failsafe():
                device_snapshot = self.process.device.as_snapshot()
                self.process.update_gpu_status()
                self.process.as_snapshot()

                assert self.npu_utilization is not None
                assert self.ai_core_utilization is not None
                assert self.vector_core_utilization is not None
                assert self.memory_bandwidth_utilization is not None
                self.npu_utilization.add(device_snapshot.npu_utilization)
                self.ai_core_utilization.add(device_snapshot.ai_core_utilization)
                self.vector_core_utilization.add(device_snapshot.vector_core_utilization)
                self.memory_bandwidth_utilization.add(device_snapshot.memory_utilization)

    def _snapshot_target(self) -> None:
        while True:
            self._daemon_running.wait()
            self.take_snapshots()
            time.sleep(self.SNAPSHOT_INTERVAL)

    def update_size(self, termsize: tuple[int, int] | None = None) -> tuple[int, int]:
        n_term_lines, n_term_cols = termsize = super().update_size(termsize=termsize)

        self.width = n_term_cols - self.x
        self.height = n_term_lines - self.y
        self.left_width = max(20, (self.width - 3) // 2)
        self.right_width = max(20, (self.width - 2) // 2)
        self.upper_height = max(5, (self.height - 8) // 2)
        self.lower_height = max(5, (self.height - 7) // 2)
        self.need_redraw = True

        with self.snapshot_lock:
            if self.enabled:
                assert self.npu_utilization is not None
                assert self.ai_core_utilization is not None
                assert self.vector_core_utilization is not None
                assert self.memory_bandwidth_utilization is not None
                self.npu_utilization.graph_size = (self.left_width, self.upper_height)
                self.ai_core_utilization.graph_size = (self.left_width, self.lower_height)
                self.vector_core_utilization.graph_size = (self.right_width, self.upper_height)
                self.memory_bandwidth_utilization.graph_size = (
                    self.right_width,
                    self.lower_height,
                )

        return termsize

    def frame_lines(self) -> list[str]:
        line = '│' + ' ' * self.left_width + '│' + ' ' * self.right_width + '│'
        return [
            '╒' + '═' * (self.width - 2) + '╕',
            '│ {} │'.format('Process:'.ljust(self.width - 4)),
            '│ {} │'.format('NPU'.ljust(self.width - 4)),
            '╞' + '═' * (self.width - 2) + '╡',
            '│' + ' ' * (self.width - 2) + '│',
            '╞' + '═' * self.left_width + '╤' + '═' * self.right_width + '╡',
            *([line] * self.upper_height),
            '├' + '─' * self.left_width + '┼' + '─' * self.right_width + '┤',
            *([line] * self.lower_height),
            '╘' + '═' * self.left_width + '╧' + '═' * self.right_width + '╛',
        ]

    def poke(self) -> None:
        if self.visible and not self._daemon_running.is_set():
            self._daemon_running.set()
            try:
                self._snapshot_daemon.start()
            except RuntimeError:
                pass
            self.take_snapshots()

        super().poke()

    def draw(self) -> None:  # pylint: disable=too-many-statements,too-many-locals,too-many-branches
        self.color_reset()

        assert self.npu_utilization is not None
        assert self.ai_core_utilization is not None
        assert self.vector_core_utilization is not None
        assert self.memory_bandwidth_utilization is not None

        if self.need_redraw:
            for y, line in enumerate(self.frame_lines(), start=self.y):
                self.addstr(y, self.x, line)

            context_width = wcslen(USER_CONTEXT)
            if not IS_WINDOWS or len(USER_CONTEXT) == context_width:
                # Do not support windows-curses with wide characters
                username_width = wcslen(USERNAME)
                hostname_width = wcslen(HOSTNAME)
                offset = self.x + self.width - context_width - 2
                self.addstr(self.y + 1, self.x + offset, USER_CONTEXT)
                self.color_at(self.y + 1, self.x + offset, width=context_width, attr='bold')
                self.color_at(
                    self.y + 1,
                    self.x + offset,
                    width=username_width,
                    fg=('yellow' if IS_SUPERUSER else 'magenta'),
                    attr='bold',
                )
                self.color_at(
                    self.y + 1,
                    self.x + offset + username_width + 1,
                    width=hostname_width,
                    fg='green',
                    attr='bold',
                )

            for offset, string in (
                (19, '╴30s├'),
                (34, '╴60s├'),
                (65, '╴120s├'),
                (95, '╴180s├'),
                (125, '╴240s├'),
                (155, '╴300s├'),
            ):
                for x_offset, width in (
                    (self.x + 1 + self.left_width, self.left_width),
                    (self.x + 1 + self.left_width + 1 + self.right_width, self.right_width),
                ):
                    if offset > width:
                        break
                    self.addstr(self.y + self.upper_height + 6, x_offset - offset, string)
                    self.color_at(
                        self.y + self.upper_height + 6,
                        x_offset - offset + 1,
                        width=len(string) - 2,
                        attr='dim',
                    )

        with self.snapshot_lock:
            process = self.process.snapshot
            device = self.process.device.snapshot

            def value_of(value: object) -> str:
                return str(NA if value is None else value)

            def rjust_value(value: object, width: int) -> str:
                return value_of(value).rjust(width)

            columns: OrderedDict[str, str | WideString] = OrderedDict(
                [
                    (' NPU', self.process.device.display_index.rjust(4)),
                    ('PID  ', f'{str(process.pid).rjust(3)} {value_of(process.type)}'),
                    (
                        'USER',
                        WideString(
                            cut_string(
                                WideString(value_of(process.username)).rjust(4),
                                maxlen=32,
                                padstr='+',
                            ),
                        ),
                    ),
                    (' NPU-MEM', rjust_value(process.gpu_memory_human, 8)),
                    ('NPU%', rjust_value(device.npu_utilization_string, 5)),
                    ('CUBE%', rjust_value(device.ai_core_utilization_string, 5)),
                    ('VEC%', rjust_value(device.vector_core_utilization_string, 5)),
                    ('AICPU%', rjust_value(device.ai_cpu_utilization_string, 5)),
                    ('CTRL%', rjust_value(device.control_cpu_utilization_string, 5)),
                    ('MBW%', rjust_value(device.memory_utilization_string, 5)),
                    ('  %CPU', rjust_value(process.cpu_percent_string, 6)),
                    (' %MEM', rjust_value(process.memory_percent_string, 5)),
                    (' TIME', rjust_value(process.running_time_human, 5)),
                ],
            )

            x = self.x + 1
            header = ''
            fields = WideString()
            no_break = True
            for i, (col, raw_value) in enumerate(columns.items()):
                value = WideString(raw_value)
                width = len(value)
                if x + width < self.width - 2:
                    if i == 0:
                        header += col.rjust(width)
                        fields += value
                    else:
                        header += ' ' + col.rjust(width)
                        fields += WideString(' ') + value
                    x = self.x + 1 + len(fields)
                else:
                    no_break = False
                    break

            self.addstr(self.y + 2, self.x + 1, header.ljust(self.width - 2))
            self.addstr(self.y + 4, self.x + 1, str(fields.ljust(self.width - 2)))
            self.color_at(
                self.y + 4,
                self.x + 1,
                width=4,
                fg=self.process.device.snapshot.display_color,
            )

            if no_break:
                x = self.x + 1 + len(fields) + 2
                if x + 4 < self.width - 2:
                    self.addstr(
                        self.y + 2,
                        x,
                        cut_string('COMMAND', self.width - x - 2, padstr='..').ljust(
                            self.width - x - 2,
                        ),
                    )
                    if process.is_zombie or process.no_permissions:
                        self.color(fg='yellow')
                    elif process.is_gone:
                        self.color(fg='red')
                    self.addstr(
                        self.y + 4,
                        x,
                        cut_string(
                            WideString(value_of(process.command)).ljust(self.width - x - 2),
                            self.width - x - 2,
                            padstr='..',
                        ),
                    )

            self.color(fg='cyan')
            for y, line in enumerate(self.npu_utilization.graph, start=self.y + 6):
                self.addstr(y, self.x + 1, line)

            self.color(fg='magenta')
            for y, line in enumerate(
                self.ai_core_utilization.graph,
                start=self.y + self.upper_height + 7,
            ):
                self.addstr(y, self.x + 1, line)

            if self.TERM_256COLOR:
                scale = (
                    self.vector_core_utilization.bound / self.vector_core_utilization.max_bound
                ) / (self.upper_height - 1)
                for i, (y, line) in enumerate(
                    enumerate(self.vector_core_utilization.graph, start=self.y + 6),
                ):
                    self.addstr(
                        y,
                        self.x + self.left_width + 2,
                        line,
                        self.get_fg_bg_attr(fg=(self.upper_height - i - 1) * scale),
                    )

                scale = (
                    self.memory_bandwidth_utilization.bound
                    / self.memory_bandwidth_utilization.max_bound
                ) / (self.lower_height - 1)
                for i, (y, line) in enumerate(
                    enumerate(
                        self.memory_bandwidth_utilization.graph,
                        start=self.y + self.upper_height + 7,
                    ),
                ):
                    self.addstr(
                        y,
                        self.x + self.left_width + 2,
                        line,
                        self.get_fg_bg_attr(fg=i * scale),
                    )
            else:
                self.color(fg=self.process.device.snapshot.memory_display_color)
                for y, line in enumerate(self.vector_core_utilization.graph, start=self.y + 6):
                    self.addstr(y, self.x + self.left_width + 2, line)

                self.color(fg=self.process.device.snapshot.gpu_display_color)
                for y, line in enumerate(
                    self.memory_bandwidth_utilization.graph,
                    start=self.y + self.upper_height + 7,
                ):
                    self.addstr(y, self.x + self.left_width + 2, line)

            self.color_reset()
            self.addstr(self.y + 6, self.x + 1, f' {self.npu_utilization.max_value_string()} ')
            self.addstr(self.y + 7, self.x + 5, f' {self.npu_utilization} ')
            self.addstr(
                self.y + self.upper_height + self.lower_height + 5,
                self.x + 5,
                f' {self.ai_core_utilization} ',
            )
            self.addstr(
                self.y + self.upper_height + self.lower_height + 6,
                self.x + 1,
                ' {} '.format(
                    cut_string(
                        self.ai_core_utilization.max_value_string(),
                        maxlen=self.left_width - 2,
                        padstr='..',
                    ),
                ),
            )
            self.addstr(
                self.y + 6,
                self.x + self.left_width + 2,
                ' {} '.format(
                    cut_string(
                        self.vector_core_utilization.max_value_string(),
                        maxlen=self.right_width - 2,
                        padstr='..',
                    ),
                ),
            )
            self.addstr(
                self.y + 7,
                self.x + self.left_width + 6,
                f' {self.vector_core_utilization} ',
            )
            self.addstr(
                self.y + self.upper_height + self.lower_height + 5,
                self.x + self.left_width + 6,
                f' {self.memory_bandwidth_utilization} ',
            )
            self.addstr(
                self.y + self.upper_height + self.lower_height + 6,
                self.x + self.left_width + 2,
                f' {self.memory_bandwidth_utilization.max_value_string()} ',
            )

            for y in range(self.y + 6, self.y + 6 + self.upper_height):
                self.addstr(y, self.x, '│')
                self.addstr(y, self.x + self.left_width + 1, '│')
            for y in range(
                self.y + self.upper_height + 7,
                self.y + self.upper_height + self.lower_height + 7,
            ):
                self.addstr(y, self.x, '│')
                self.addstr(y, self.x + self.left_width + 1, '│')

            self.color(attr='dim')
            for y, p in itertools.chain(
                get_yticks(self.npu_utilization, self.y + 6),
                get_yticks(self.ai_core_utilization, self.y + self.upper_height + 7),
            ):
                self.addstr(y, self.x, f'├╴{p}% ')
                self.color_at(y, self.x, width=2, attr=0)
            x = self.x + self.left_width + 1
            for y, p in itertools.chain(
                get_yticks(self.vector_core_utilization, self.y + 6),
                get_yticks(
                    self.memory_bandwidth_utilization,
                    self.y + self.upper_height + 7,
                ),
            ):
                self.addstr(y, x, f'├╴{p}% ')
                self.color_at(y, x, width=2, attr=0)

    def destroy(self) -> None:
        super().destroy()
        self._daemon_running.clear()

    def press(self, key: int) -> bool:
        self.root.keymaps.use_keymap('process-metrics')
        return self.root.press(key)
