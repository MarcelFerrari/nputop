# This file is part of nputop, the interactive Huawei-NPU process viewer.
#
# Licensed under the Apache License, Version 2.0.
# ==============================================================================
"""Live classes for Huawei Ascend NPU devices."""

# pylint: disable=missing-function-docstring,too-many-instance-attributes

from __future__ import annotations

import contextlib
import threading
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple

from nvitop.api import libdcmi
from nvitop.api.process import GpuProcess
from nvitop.api.utils import NA, NaType, Snapshot, bytes2human


if TYPE_CHECKING:
    from collections.abc import Generator, Iterable
    from typing_extensions import Self


__all__ = [
    'CudaDevice',
    'CudaMigDevice',
    'Device',
    'MigDevice',
    'PhysicalDevice',
    'normalize_cuda_visible_devices',
    'parse_cuda_visible_devices',
]


class MemoryInfo(NamedTuple):
    """Device memory information in bytes."""

    total: int | NaType
    free: int | NaType
    used: int | NaType
    reserved: int | NaType = NA


class ClockInfos(NamedTuple):
    """Clock information in MHz."""

    graphics: int | NaType
    sm: int | NaType
    memory: int | NaType
    video: int | NaType


class ClockSpeedInfos(NamedTuple):
    """Current and maximum clock information in MHz."""

    current: ClockInfos
    max: ClockInfos


class UtilizationRates(NamedTuple):
    """Device utilization rates in percent."""

    gpu: int | NaType
    memory: int | NaType
    encoder: int | NaType
    decoder: int | NaType


class ThroughputInfo(NamedTuple):
    """Data throughput information in KiB/s."""

    tx: int | NaType
    rx: int | NaType

    @property
    def transmit(self) -> int | NaType:
        return self.tx

    @property
    def receive(self) -> int | NaType:
        return self.rx


_DEVICE_CACHE: list[libdcmi.DcmDevice] | None = None
_DEVICE_CACHE_LOCK = threading.RLock()


def _device_pairs() -> list[libdcmi.DcmDevice]:
    global _DEVICE_CACHE  # pylint: disable=global-statement

    with _DEVICE_CACHE_LOCK:
        if _DEVICE_CACHE is None:
            _DEVICE_CACHE = libdcmi.device_pairs()
        return list(_DEVICE_CACHE)


def _check_return(value: Any, types: type | tuple[type, ...] | None = None) -> bool:
    return libdcmi.dcmiCheckReturn(value, types)


def parse_cuda_visible_devices(cuda_visible_devices: str | None = None) -> list[int]:
    """Compatibility stub: CUDA visibility is not used for Ascend NPUs."""
    del cuda_visible_devices
    try:
        return list(range(Device.count()))
    except libdcmi.DCMIError:
        return []


def normalize_cuda_visible_devices(cuda_visible_devices: str | None = None) -> str:
    """Compatibility stub: return comma-separated NPU UUIDs."""
    del cuda_visible_devices
    try:
        return ','.join(device.uuid() for device in Device.all())
    except libdcmi.DCMIError:
        return ''


class Device:  # pylint: disable=too-many-public-methods
    """Live class for an Ascend NPU device."""

    GPU_PROCESS_CLASS: ClassVar[type[GpuProcess]] = GpuProcess
    cuda: ClassVar[type[CudaDevice]]

    SNAPSHOT_KEYS: ClassVar[list[str]] = [
        'name',
        'uuid',
        'bus_id',
        'memory_info',
        'memory_used',
        'memory_free',
        'memory_total',
        'memory_used_human',
        'memory_free_human',
        'memory_total_human',
        'memory_percent',
        'memory_usage',
        'component_utilization_rates',
        'npu_utilization',
        'ai_core_utilization',
        'vector_core_utilization',
        'ai_cpu_utilization',
        'control_cpu_utilization',
        'memory_bandwidth_utilization',
        'on_chip_memory_utilization',
        'on_chip_memory_bandwidth_utilization',
        'ddr_utilization',
        'utilization_rates',
        'gpu_utilization',
        'memory_utilization',
        'encoder_utilization',
        'decoder_utilization',
        'clock_infos',
        'max_clock_infos',
        'clock_speed_infos',
        'sm_clock',
        'memory_clock',
        'fan_speed',
        'hbm_temperature',
        'temperature',
        'power_usage',
        'power_limit',
        'power_status',
        'pcie_throughput',
        'pcie_tx_throughput',
        'pcie_rx_throughput',
        'pcie_tx_throughput_human',
        'pcie_rx_throughput_human',
        'display_active',
        'display_mode',
        'current_driver_model',
        'persistence_mode',
        'performance_state',
        'health_status',
        'ecc_summary',
        'health_ecc_summary',
        'boot_status',
        'work_mode',
        'total_volatile_uncorrected_ecc_errors',
        'compute_mode',
        'cuda_compute_capability',
        'mig_mode',
    ]

    def __init__(
        self,
        index: int | tuple[int, int] | str | None = None,
        *,
        uuid: str | None = None,
        bus_id: str | None = None,
    ) -> None:
        del bus_id

        pairs = _device_pairs()
        if uuid is not None:
            if not uuid.startswith('NPU-'):
                raise libdcmi.DCMIError(f'Invalid NPU UUID: {uuid!r}')
            _, card_id, device_id = uuid.split('-', 2)
            self._device = libdcmi.DcmDevice(card_id=int(card_id), device_id=int(device_id))
            try:
                self._index = pairs.index(self._device)
            except ValueError:
                raise libdcmi.DCMIError(f'NPU device not found for UUID {uuid!r}') from None
        elif isinstance(index, tuple):
            self._device = libdcmi.DcmDevice(card_id=int(index[0]), device_id=int(index[1]))
            try:
                self._index = pairs.index(self._device)
            except ValueError:
                raise libdcmi.DCMIError(f'NPU device not found: {index!r}') from None
        elif isinstance(index, str) and index.startswith('NPU-'):
            _, card_id, device_id = index.split('-', 2)
            self._device = libdcmi.DcmDevice(card_id=int(card_id), device_id=int(device_id))
            try:
                self._index = pairs.index(self._device)
            except ValueError:
                raise libdcmi.DCMIError(f'NPU device not found for UUID {index!r}') from None
        else:
            if index is None:
                index = 0
            if not isinstance(index, int):
                raise TypeError(
                    f'index must be an integer, tuple, or NPU UUID string, got {index!r}',
                )
            if not 0 <= index < len(pairs):
                raise libdcmi.DCMIError(f'Invalid NPU device index: {index!r}')
            self._index = index
            self._device = pairs[index]

        self._name: str | NaType = NA
        self._uuid: str = f'NPU-{self.card_id}-{self.device_id}'
        self._memory_total: int | NaType = NA
        self._memory_total_human: str | NaType = NA
        self._max_clock_infos = ClockInfos(graphics=NA, sm=NA, memory=NA, video=NA)
        self._component_utilization_cache: libdcmi.ComponentUtilization | None = None
        self._oneshot_depth = 0
        self._lock = threading.RLock()
        self._ident = (self.card_id, self.device_id)
        self._hash: int | None = None

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}('
            f'index={self.index}, '
            f'card_id={self.card_id}, '
            f'device_id={self.device_id}, '
            f'name={self.name()!r}, '
            f'total_memory={self.memory_total_human()}'
            ')'
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Device):
            return NotImplemented
        return self._ident == other._ident

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash(self._ident)
        return self._hash

    def __reduce__(self) -> tuple[type[Device], tuple[int]]:
        return self.__class__, (self.index,)

    @classmethod
    def is_available(cls) -> bool:
        try:
            return cls.count() > 0
        except libdcmi.DCMIError:
            return False

    @staticmethod
    def driver_version() -> str | NaType:
        return libdcmi.driver_version()

    cuda_driver_version = driver_version
    max_cuda_version = driver_version
    cuda_runtime_version = staticmethod(lambda: NA)
    cudart_version = cuda_runtime_version

    @classmethod
    def count(cls) -> int:
        return len(_device_pairs())

    @classmethod
    def all(cls) -> list[Self]:
        return cls.from_indices()

    @classmethod
    def from_indices(cls, indices: int | Iterable[int] | None = None) -> list[Self]:
        if indices is None:
            indices = range(cls.count())
        if isinstance(indices, int):
            indices = [indices]
        return [cls(index) for index in indices]

    from_cuda_visible_devices = all
    from_cuda_indices = from_indices
    parse_cuda_visible_devices = staticmethod(parse_cuda_visible_devices)
    normalize_cuda_visible_devices = staticmethod(normalize_cuda_visible_devices)

    @property
    def index(self) -> int:
        return self._index

    @property
    def nvml_index(self) -> int:
        return self._index

    @property
    def physical_index(self) -> int:
        return self._index

    @property
    def card_id(self) -> int:
        return self._device.card_id

    @property
    def device_id(self) -> int:
        return self._device.device_id

    @property
    def handle(self) -> libdcmi.DcmDevice:
        return self._device

    @property
    def cuda_index(self) -> int:
        return self.index

    def name(self) -> str | NaType:
        if self._name is NA:
            try:
                info = libdcmi.chip_info(self._device)
            except libdcmi.DCMIError:
                self._name = f'Ascend NPU {self.card_id}:{self.device_id}'
            else:
                npu_name = info.get('npu_name', NA)
                chip_name = info.get('chip_name', NA)
                chip_type = info.get('chip_type', NA)
                name = npu_name if npu_name is not NA else chip_name
                self._name = str(name if name is not NA else chip_type)
        return self._name

    def uuid(self) -> str:
        return self._uuid

    def bus_id(self) -> str | NaType:
        info = libdcmi.pcie_info(self._device)
        if isinstance(info, libdcmi.PcieInfo):
            return info.bus_id
        return NA

    def serial(self) -> str | NaType:
        logical_id = self.logic_id()
        if logical_id is not NA:
            return f'logic-{logical_id}'
        return self.uuid()

    def logic_id(self) -> int | NaType:
        return libdcmi.logic_id(self._device)

    def board_info(self) -> libdcmi.BoardInfo | NaType:
        return libdcmi.board_info(self._device)

    def board_id(self) -> int | NaType:
        info = self.board_info()
        return info.board_id if isinstance(info, libdcmi.BoardInfo) else NA

    def slot_id(self) -> int | NaType:
        info = self.board_info()
        return info.slot_id if isinstance(info, libdcmi.BoardInfo) else NA

    def memory_info(self) -> MemoryInfo:
        try:
            info = libdcmi.hbm_info(self._device)
        except libdcmi.DCMIError:
            return MemoryInfo(total=NA, free=NA, used=NA, reserved=NA)
        return MemoryInfo(total=info.total, free=info.free, used=info.used, reserved=NA)

    def memory_total(self) -> int | NaType:
        if self._memory_total is NA:
            self._memory_total = self.memory_info().total
        return self._memory_total

    def memory_used(self) -> int | NaType:
        return self.memory_info().used

    def memory_free(self) -> int | NaType:
        return self.memory_info().free

    def memory_total_human(self) -> str | NaType:
        if self._memory_total_human is NA:
            self._memory_total_human = bytes2human(self.memory_total())
        return self._memory_total_human

    def memory_used_human(self) -> str | NaType:
        return bytes2human(self.memory_used())

    def memory_free_human(self) -> str | NaType:
        return bytes2human(self.memory_free())

    def memory_percent(self) -> float | NaType:
        total, _, used, _ = self.memory_info()
        if _check_return(used, int) and _check_return(total, int) and total > 0:
            return round(100.0 * used / total, 1)
        return NA

    def memory_usage(self) -> str:
        return f'{self.memory_used_human()} / {self.memory_total_human()}'

    def component_utilization_rates(self) -> libdcmi.ComponentUtilization:
        if self._oneshot_depth <= 0:
            return libdcmi.component_utilization(self._device)
        if self._component_utilization_cache is None:
            self._component_utilization_cache = libdcmi.component_utilization(self._device)
        return self._component_utilization_cache

    def npu_utilization(self) -> int | NaType:
        return self.component_utilization_rates().npu

    def ai_core_utilization(self) -> int | NaType:
        return self.component_utilization_rates().ai_core

    def vector_core_utilization(self) -> int | NaType:
        return self.component_utilization_rates().vector_core

    def ai_cpu_utilization(self) -> int | NaType:
        return self.component_utilization_rates().ai_cpu

    def control_cpu_utilization(self) -> int | NaType:
        return self.component_utilization_rates().control_cpu

    def memory_bandwidth_utilization(self) -> int | NaType:
        return self.component_utilization_rates().memory_bandwidth

    def on_chip_memory_utilization(self) -> int | NaType:
        return self.component_utilization_rates().on_chip_memory

    def on_chip_memory_bandwidth_utilization(self) -> int | NaType:
        return self.component_utilization_rates().on_chip_memory_bandwidth

    def ddr_utilization(self) -> int | NaType:
        return self.component_utilization_rates().ddr

    def bar1_memory_info(self) -> MemoryInfo:
        return MemoryInfo(total=NA, free=NA, used=NA, reserved=NA)

    def bar1_memory_total(self) -> int | NaType:
        return NA

    def bar1_memory_used(self) -> int | NaType:
        return NA

    def bar1_memory_free(self) -> int | NaType:
        return NA

    def bar1_memory_total_human(self) -> str | NaType:
        return NA

    def bar1_memory_used_human(self) -> str | NaType:
        return NA

    def bar1_memory_free_human(self) -> str | NaType:
        return NA

    def bar1_memory_percent(self) -> float | NaType:
        return NA

    def bar1_memory_usage(self) -> str:
        return f'{NA} / {NA}'

    def utilization_rates(self) -> UtilizationRates:
        components = self.component_utilization_rates()
        gpu = components.npu
        if gpu is NA:
            gpu = components.ai_core
        try:
            memory = libdcmi.hbm_info(self._device).bandwidth_utilization
        except libdcmi.DCMIError:
            memory = components.on_chip_memory_bandwidth
        if memory is NA:
            memory = components.memory_bandwidth
        return UtilizationRates(gpu=gpu, memory=memory, encoder=NA, decoder=NA)

    def gpu_utilization(self) -> int | NaType:
        return self.utilization_rates().gpu

    gpu_percent = gpu_utilization

    def memory_utilization(self) -> int | NaType:
        return self.utilization_rates().memory

    def encoder_utilization(self) -> int | NaType:
        return NA

    def decoder_utilization(self) -> int | NaType:
        return NA

    def clock_infos(self) -> ClockInfos:
        sm = libdcmi.frequency(self._device, libdcmi.DCMI_FREQ_AICORE_CURRENT)
        memory = libdcmi.frequency(self._device, libdcmi.DCMI_FREQ_ON_CHIP_MEMORY)
        if memory is NA:
            try:
                memory = libdcmi.hbm_info(self._device).memory_clock
            except libdcmi.DCMIError:
                memory = NA
        vector = (
            libdcmi.frequency(self._device, libdcmi.DCMI_FREQ_VECTOR_CORE_CURRENT)
            if libdcmi.unsafe_metric_enabled('component')
            else NA
        )
        return ClockInfos(graphics=sm, sm=sm, memory=memory, video=vector)

    clocks = clock_infos

    def max_clock_infos(self) -> ClockInfos:
        sm = libdcmi.frequency(self._device, libdcmi.DCMI_FREQ_AICORE_MAX)
        self._max_clock_infos = ClockInfos(graphics=sm, sm=sm, memory=NA, video=NA)
        return self._max_clock_infos

    max_clocks = max_clock_infos

    def clock_speed_infos(self) -> ClockSpeedInfos:
        return ClockSpeedInfos(current=self.clock_infos(), max=self.max_clock_infos())

    def graphics_clock(self) -> int | NaType:
        return self.clock_infos().graphics

    def sm_clock(self) -> int | NaType:
        return self.clock_infos().sm

    def memory_clock(self) -> int | NaType:
        return self.clock_infos().memory

    def video_clock(self) -> int | NaType:
        return NA

    def max_graphics_clock(self) -> int | NaType:
        return self.max_clock_infos().graphics

    def max_sm_clock(self) -> int | NaType:
        return self.max_clock_infos().sm

    def max_memory_clock(self) -> int | NaType:
        return NA

    def max_video_clock(self) -> int | NaType:
        return NA

    def fan_speed(self) -> int | NaType:
        return NA

    def hbm_temperature(self) -> int | NaType:
        return libdcmi.hbm_temperature(self._device)

    def temperature(self) -> int | NaType:
        value = libdcmi.temperature(self._device)
        if value is NA:
            try:
                value = libdcmi.hbm_info(self._device).temperature
            except libdcmi.DCMIError:
                value = NA
        return value

    def power_usage(self) -> int | NaType:
        return libdcmi.power_usage(self._device)

    power_draw = power_usage

    def power_limit(self) -> int | NaType:
        return NA

    def power_status(self) -> str:
        power_usage = self.power_usage()
        if _check_return(power_usage, int):
            power_usage = f'{round(power_usage / 1000)}W'
        return str(power_usage)

    def pcie_throughput(self) -> ThroughputInfo:
        return ThroughputInfo(tx=NA, rx=NA)

    def pcie_tx_throughput(self) -> int | NaType:
        return NA

    def pcie_rx_throughput(self) -> int | NaType:
        return NA

    def pcie_tx_throughput_human(self) -> str | NaType:
        return NA

    def pcie_rx_throughput_human(self) -> str | NaType:
        return NA

    def nvlink_link_count(self) -> int:
        return 0

    def nvlink_throughput(self, interval: float | None = None) -> list[ThroughputInfo]:
        del interval
        return []

    def nvlink_total_tx_throughput(self) -> int | NaType:
        return NA

    def nvlink_total_rx_throughput(self) -> int | NaType:
        return NA

    def nvlink_mean_tx_throughput(self) -> int | NaType:
        return NA

    def nvlink_mean_rx_throughput(self) -> int | NaType:
        return NA

    def nvlink_tx_throughput(self) -> list[int | NaType]:
        return []

    def nvlink_rx_throughput(self) -> list[int | NaType]:
        return []

    def display_active(self) -> str | NaType:
        logical_id = self.logic_id()
        if _check_return(logical_id, int):
            return f'L{logical_id}'
        return NA

    def display_mode(self) -> str | NaType:
        return NA

    def current_driver_model(self) -> str | NaType:
        return libdcmi.driver_health_status()

    driver_model = current_driver_model

    def persistence_mode(self) -> str | NaType:
        return self.health_status()

    def performance_state(self) -> str | NaType:
        return self._format_clock_short(self.sm_clock())

    @staticmethod
    def _format_clock_short(clock: int | NaType) -> str | NaType:
        if not _check_return(clock, int):
            return NA
        if clock >= 1000:
            return f'{clock / 1000.0:.1f}G'
        return f'{clock}M'

    def health_status(self) -> str | NaType:
        return libdcmi.health_status(self._device)

    def ecc_summary(self) -> str | NaType:
        parts = []
        for label, device_type in (
            ('H', libdcmi.DCMI_DEVICE_TYPE_HBM),
            ('D', libdcmi.DCMI_DEVICE_TYPE_DDR),
        ):
            info = libdcmi.ecc_info(self._device, device_type)
            if isinstance(info, libdcmi.EccInfo):
                parts.append(f'{label}{info.single_bit_errors}/{info.double_bit_errors}')
        return ' '.join(parts) if parts else NA

    def health_ecc_summary(self) -> str | NaType:
        health = self.health_status()
        ecc = self.ecc_summary()
        if health is NA:
            return ecc
        if ecc is NA:
            return health
        return f'{health} {ecc}'

    def boot_status(self) -> str | NaType:
        return libdcmi.boot_status(self._device)

    def work_mode(self) -> str | NaType:
        return libdcmi.work_mode(self._device)

    def total_volatile_uncorrected_ecc_errors(self) -> str | NaType:
        return self.health_ecc_summary()

    def compute_mode(self) -> str | NaType:
        return self.work_mode()

    def cuda_compute_capability(self) -> tuple[int, int] | NaType:
        return NA

    def is_mig_device(self) -> bool:
        return False

    def mig_mode(self) -> str | NaType:
        return NA

    def is_mig_mode_enabled(self) -> bool:
        return False

    def max_mig_device_count(self) -> int:
        return 0

    def mig_devices(self) -> list[MigDevice]:
        return []

    def is_leaf_device(self) -> bool:
        return True

    def to_leaf_devices(self) -> list[Device]:
        return [self]

    def gpu_instance_id(self) -> int | NaType:
        return NA

    def compute_instance_id(self) -> int | NaType:
        return NA

    def processes(self) -> dict[int, GpuProcess]:
        processes = {}
        for proc in libdcmi.running_processes(self._device):
            gpu_process = self.GPU_PROCESS_CLASS(
                pid=proc.pid,
                device=self,
                gpu_memory=proc.memory,
                type='C',
            )
            gpu_process.set_gpu_utilization(NA, NA, NA, NA)
            processes[proc.pid] = gpu_process
        return processes

    def as_snapshot(self) -> Snapshot:
        with self.oneshot():
            return Snapshot(
                real=self,
                index=self.index,
                physical_index=self.physical_index,
                card_id=self.card_id,
                device_id=self.device_id,
                **{key: getattr(self, key)() for key in self.SNAPSHOT_KEYS},
            )

    @contextlib.contextmanager
    def oneshot(self) -> Generator[None]:
        with self._lock:
            self._oneshot_depth += 1
            self._component_utilization_cache = None
            try:
                yield
            finally:
                self._oneshot_depth -= 1
                if self._oneshot_depth == 0:
                    self._component_utilization_cache = None


class PhysicalDevice(Device):
    """Compatibility alias for a physical Ascend NPU."""


class MigDevice(Device):
    """Compatibility stub. Ascend NPUs are exposed as physical devices in this fork."""


class CudaDevice(Device):
    """Compatibility stub for callers that still import CudaDevice."""


class CudaMigDevice(CudaDevice, MigDevice):
    """Compatibility stub for callers that still import CudaMigDevice."""


Device.cuda = CudaDevice
