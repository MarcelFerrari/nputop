"""Small ctypes wrapper for Huawei Ascend DCMI."""

# pylint: disable=missing-function-docstring,too-few-public-methods,invalid-name

from __future__ import annotations

import ctypes
import os
import threading
from typing import Any, NamedTuple

from nvitop.api.utils import NA, NaType


__all__ = [
    'BoardInfo',
    'ClockSample',
    'ComponentUtilization',
    'DCMIError',
    'DCMILibraryNotFound',
    'DCMINotSupported',
    'DcmDevice',
    'EccInfo',
    'HbmInfo',
    'PcieInfo',
    'ProcessInfo',
    'board_info',
    'boot_status',
    'chip_info',
    'component_utilization',
    'dcmiCheckReturn',
    'device_count',
    'device_pairs',
    'driver_health_status',
    'driver_version',
    'ecc_info',
    'frequency',
    'hbm_info',
    'hbm_temperature',
    'health_status',
    'logic_id',
    'pcie_info',
    'power_usage',
    'running_processes',
    'temperature',
    'unsafe_metric_enabled',
    'utilization_rate',
    'work_mode',
]


DCMI_SUCCESS = 0
MAX_CARD_NUM = 64
MAX_PROC_NUM = 32
MAX_CHIP_NAME_LEN = 32

# Values from the Atlas DCMI API reference.
DCMI_FREQ_MEMORY = 1
DCMI_FREQ_CTRL_CPU = 2
DCMI_FREQ_ON_CHIP_MEMORY = 6
DCMI_FREQ_AICORE_CURRENT = 7
DCMI_FREQ_AICORE_MAX = 9
DCMI_FREQ_VECTOR_CORE_CURRENT = 12

DCMI_UTILIZATION_RATE_MEMORY = 1
DCMI_UTILIZATION_RATE_AICORE = 2
DCMI_UTILIZATION_RATE_AICPU = 3
DCMI_UTILIZATION_RATE_CTRL_CPU = 4
DCMI_UTILIZATION_RATE_MEMORY_BANDWIDTH = 5
DCMI_UTILIZATION_RATE_ON_CHIP_MEMORY = 6
DCMI_UTILIZATION_RATE_DDR = 8
DCMI_UTILIZATION_RATE_ON_CHIP_MEMORY_BANDWIDTH = 10
DCMI_UTILIZATION_RATE_VECTOR_CORE = 12
DCMI_UTILIZATION_RATE_NPU = 13

DCMI_DEVICE_TYPE_DDR = 0
DCMI_DEVICE_TYPE_HBM = 2

UNSAFE_METRICS_ENV = 'NPUTOP_DCMI_ENABLE_UNSAFE_METRICS'
DEFAULT_OPTIONAL_METRICS = frozenset(
    {
        'board',
        'boot',
        'chip-v2',
        'component',
        'driver-health',
        'health',
        'logic',
        'pcie',
        'work',
    },
)


class DCMIError(RuntimeError):
    """Base exception for DCMI failures."""

    def __init__(self, message: str, code: int | None = None) -> None:
        super().__init__(message if code is None else f'{message} (DCMI error {code})')
        self.code = code


class DCMILibraryNotFound(DCMIError):
    """Raised when libdcmi cannot be loaded."""


class DCMINotSupported(DCMIError):
    """Raised when the loaded DCMI library does not expose a requested symbol."""


class DcmDevice(NamedTuple):
    """A physical Ascend NPU address."""

    card_id: int
    device_id: int


class ClockSample(NamedTuple):
    """NPU clock sample in MHz."""

    current: int | NaType
    maximum: int | NaType
    memory: int | NaType


class HbmInfo(NamedTuple):
    """On-chip memory information in bytes and percent."""

    total: int | NaType
    used: int | NaType
    free: int | NaType
    bandwidth_utilization: int | NaType
    memory_clock: int | NaType
    temperature: int | NaType


class PcieInfo(NamedTuple):
    """PCIe identity and BDF information."""

    device_id: int
    vendor_id: int
    subvendor_id: int
    subdevice_id: int
    bdf_bus_id: int
    bdf_device_id: int
    bdf_function_id: int

    @property
    def bus_id(self) -> str:
        return f'0000:{self.bdf_bus_id:02x}:{self.bdf_device_id:02x}.{self.bdf_function_id}'


class BoardInfo(NamedTuple):
    """Board and slot information."""

    board_id: int
    pcb_id: int
    bom_id: int
    slot_id: int


class EccInfo(NamedTuple):
    """ECC status and counters."""

    enabled: bool | NaType
    single_bit_errors: int | NaType
    double_bit_errors: int | NaType
    total_single_bit_errors: int | NaType
    total_double_bit_errors: int | NaType
    single_bit_isolated_pages: int | NaType
    double_bit_isolated_pages: int | NaType


class ComponentUtilization(NamedTuple):
    """Device component utilization rates in percent."""

    npu: int | NaType
    ai_core: int | NaType
    vector_core: int | NaType
    ai_cpu: int | NaType
    control_cpu: int | NaType
    memory: int | NaType
    memory_bandwidth: int | NaType
    on_chip_memory: int | NaType
    on_chip_memory_bandwidth: int | NaType
    ddr: int | NaType


class ProcessInfo(NamedTuple):
    """Process resource information reported by DCMI."""

    pid: int
    memory: int | NaType


class _DcmiChipInfo(ctypes.Structure):
    _fields_ = [
        ('chip_type', ctypes.c_ubyte * MAX_CHIP_NAME_LEN),
        ('chip_name', ctypes.c_ubyte * MAX_CHIP_NAME_LEN),
        ('chip_ver', ctypes.c_ubyte * MAX_CHIP_NAME_LEN),
        ('aicore_cnt', ctypes.c_uint),
    ]


class _DcmiChipInfoV2(ctypes.Structure):
    _fields_ = [
        ('chip_type', ctypes.c_ubyte * MAX_CHIP_NAME_LEN),
        ('chip_name', ctypes.c_ubyte * MAX_CHIP_NAME_LEN),
        ('chip_ver', ctypes.c_ubyte * MAX_CHIP_NAME_LEN),
        ('aicore_cnt', ctypes.c_uint),
        ('npu_name', ctypes.c_ubyte * MAX_CHIP_NAME_LEN),
    ]


class _DcmiPcieInfo(ctypes.Structure):
    _fields_ = [
        ('deviceid', ctypes.c_uint),
        ('venderid', ctypes.c_uint),
        ('subvenderid', ctypes.c_uint),
        ('subdeviceid', ctypes.c_uint),
        ('bdf_deviceid', ctypes.c_uint),
        ('bdf_busid', ctypes.c_uint),
        ('bdf_funcid', ctypes.c_uint),
    ]


class _DcmiBoardInfo(ctypes.Structure):
    _fields_ = [
        ('board_id', ctypes.c_uint),
        ('pcb_id', ctypes.c_uint),
        ('bom_id', ctypes.c_uint),
        ('slot_id', ctypes.c_uint),
    ]


class _DcmiHbmInfo(ctypes.Structure):
    _fields_ = [
        ('memory_size', ctypes.c_ulonglong),  # MB
        ('freq', ctypes.c_uint),  # MHz
        ('memory_usage', ctypes.c_ulonglong),  # MB
        ('temp', ctypes.c_int),
        ('bandwith_util_rate', ctypes.c_uint),
    ]


class _DcmiEccInfo(ctypes.Structure):
    _fields_ = [
        ('enable_fag', ctypes.c_int),
        ('single_bit_error_cnt', ctypes.c_uint),
        ('double_bit_error_cnt', ctypes.c_uint),
        ('total_single_bit_error_cnt', ctypes.c_uint),
        ('total_double_bit_error_cnt', ctypes.c_uint),
        ('single_bit_isolated_pages_cnt', ctypes.c_uint),
        ('double_bit_isolated_pages_cnt', ctypes.c_uint),
    ]


class _DcmiProcMemInfo(ctypes.Structure):
    _fields_ = [
        ('proc_id', ctypes.c_int),
        ('proc_mem_usage', ctypes.c_ulong),  # bytes
    ]


_lib: ctypes.CDLL | None = None
_initialized = False
_prototypes_configured = False
_lock = threading.RLock()


def dcmiCheckReturn(retval: Any, types: type | tuple[type, ...] | None = None, /) -> bool:
    """Check whether a value is applicable and optionally has one of the requested types."""
    if types is None:
        return retval != NA
    return retval != NA and isinstance(retval, types)


def _decode_bytes(value: ctypes.Array[ctypes.c_ubyte]) -> str:
    raw = bytes(value)
    raw = raw.split(b'\0', 1)[0]
    return raw.decode('utf-8', errors='replace')


def _library_names() -> list[str]:
    configured = os.getenv('NPUTOP_DCMI_LIBRARY')
    if configured:
        return [configured]
    return ['libdcmi.so']


def unsafe_metric_enabled(name: str) -> bool:
    """Return whether a metric probe that may vary by DCMI version is enabled."""
    normalized = name.lower()
    if normalized == 'ecc':
        return False

    value = os.getenv(UNSAFE_METRICS_ENV, '')
    if not value:
        return normalized in DEFAULT_OPTIONAL_METRICS
    tokens = {token.strip().lower() for token in value.replace(';', ',').split(',')}
    if 'none' in tokens or 'off' in tokens or 'false' in tokens or '0' in tokens:
        return False
    return normalized in DEFAULT_OPTIONAL_METRICS or 'all' in tokens or normalized in tokens


def _load_library() -> ctypes.CDLL:
    global _lib  # pylint: disable=global-statement

    if _lib is not None:
        return _lib

    errors: list[str] = []
    for name in _library_names():
        _lib = _try_load_library(name, errors)
        if _lib is not None:
            return _lib

    details = '; '.join(errors) if errors else 'no candidate library names'
    raise DCMILibraryNotFound(
        'Huawei Ascend DCMI library not found. '
        'Install the Ascend driver/runtime or set NPUTOP_DCMI_LIBRARY to libdcmi.so. '
        f'Attempted: {details}',
    )


def _try_load_library(name: str, errors: list[str]) -> ctypes.CDLL | None:
    try:
        return ctypes.CDLL(name)
    except OSError as ex:
        errors.append(f'{name}: {ex}')
        return None


def _set_prototype(
    lib: ctypes.CDLL,
    name: str,
    restype: Any,
    argtypes: list[Any],
) -> None:
    try:
        func = getattr(lib, name)
    except AttributeError:
        return
    func.restype = restype
    func.argtypes = argtypes


def _configure_prototypes(lib: ctypes.CDLL) -> None:
    global _prototypes_configured  # pylint: disable=global-statement

    if _prototypes_configured:
        return

    _set_prototype(lib, 'dcmi_init', ctypes.c_int, [])
    _set_prototype(lib, 'dcmi_get_driver_version', ctypes.c_int, [ctypes.c_char_p, ctypes.c_uint])
    _set_prototype(
        lib,
        'dcmi_get_card_list',
        ctypes.c_int,
        [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_int],
    )
    _set_prototype(
        lib,
        'dcmi_get_device_num_in_card',
        ctypes.c_int,
        [ctypes.c_int, ctypes.POINTER(ctypes.c_int)],
    )
    _set_prototype(
        lib,
        'dcmi_get_device_chip_info',
        ctypes.c_int,
        [ctypes.c_int, ctypes.c_int, ctypes.POINTER(_DcmiChipInfo)],
    )
    _set_prototype(
        lib,
        'dcmi_get_device_chip_info_v2',
        ctypes.c_int,
        [ctypes.c_int, ctypes.c_int, ctypes.POINTER(_DcmiChipInfoV2)],
    )
    _set_prototype(
        lib,
        'dcmi_get_device_pcie_info',
        ctypes.c_int,
        [ctypes.c_int, ctypes.c_int, ctypes.POINTER(_DcmiPcieInfo)],
    )
    _set_prototype(
        lib,
        'dcmi_get_device_board_info',
        ctypes.c_int,
        [ctypes.c_int, ctypes.c_int, ctypes.POINTER(_DcmiBoardInfo)],
    )
    _set_prototype(
        lib,
        'dcmi_get_device_logic_id',
        ctypes.c_int,
        [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int],
    )
    for name in ('dcmi_get_device_hbm_info', 'dcmi_get_hbm_info'):
        _set_prototype(
            lib,
            name,
            ctypes.c_int,
            [ctypes.c_int, ctypes.c_int, ctypes.POINTER(_DcmiHbmInfo)],
        )
    _set_prototype(
        lib,
        'dcmi_get_device_frequency',
        ctypes.c_int,
        [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_uint)],
    )
    _set_prototype(
        lib,
        'dcmi_get_device_utilization_rate',
        ctypes.c_int,
        [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_uint)],
    )
    _set_prototype(
        lib,
        'dcmi_get_device_temperature',
        ctypes.c_int,
        [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)],
    )
    _set_prototype(
        lib,
        'dcmi_get_device_power_info',
        ctypes.c_int,
        [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)],
    )
    _set_prototype(
        lib,
        'dcmi_get_device_health',
        ctypes.c_int,
        [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_uint)],
    )
    _set_prototype(lib, 'dcmi_get_driver_health', ctypes.c_int, [ctypes.POINTER(ctypes.c_uint)])
    _set_prototype(
        lib,
        'dcmi_get_device_ecc_info',
        ctypes.c_int,
        [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(_DcmiEccInfo)],
    )
    _set_prototype(
        lib,
        'dcmi_get_device_boot_status',
        ctypes.c_int,
        [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)],
    )
    _set_prototype(
        lib,
        'dcmi_get_npu_work_mode',
        ctypes.c_int,
        [ctypes.c_int, ctypes.POINTER(ctypes.c_ubyte)],
    )
    _set_prototype(
        lib,
        'dcmi_get_device_resource_info',
        ctypes.c_int,
        [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(_DcmiProcMemInfo),
            ctypes.POINTER(ctypes.c_int),
        ],
    )

    _prototypes_configured = True


def _lazy_init() -> ctypes.CDLL:
    global _initialized  # pylint: disable=global-statement

    lib = _load_library()
    _configure_prototypes(lib)
    if not _initialized:
        init = getattr(lib, 'dcmi_init', None)
        if init is not None:
            ret = init()
            if ret != DCMI_SUCCESS:
                raise DCMIError('Failed to initialize DCMI', ret)
        _initialized = True
    return lib


def _call(name: str, *args: Any) -> int:
    with _lock:
        lib = _lazy_init()
        try:
            func = getattr(lib, name)
        except AttributeError as ex:
            raise DCMINotSupported(f'DCMI function {name} is not available') from ex
        ret = func(*args)
        if ret != DCMI_SUCCESS:
            raise DCMIError(f'DCMI function {name} failed', ret)
        return ret


def driver_version() -> str | NaType:
    """Return the Ascend driver version."""
    buffer = ctypes.create_string_buffer(64)
    try:
        _call('dcmi_get_driver_version', buffer, ctypes.sizeof(buffer))
    except DCMIError:
        return NA
    return buffer.value.decode('utf-8', errors='replace') or NA


def device_pairs() -> list[DcmDevice]:
    """Return all visible DCMI card/device pairs."""
    card_num = ctypes.c_int()
    card_list = (ctypes.c_int * MAX_CARD_NUM)()
    _call('dcmi_get_card_list', ctypes.byref(card_num), card_list, MAX_CARD_NUM)

    devices: list[DcmDevice] = []
    for offset in range(max(0, card_num.value)):
        card_id = int(card_list[offset])
        device_num = ctypes.c_int()
        _call('dcmi_get_device_num_in_card', card_id, ctypes.byref(device_num))
        devices.extend(
            DcmDevice(card_id=card_id, device_id=device_id)
            for device_id in range(max(0, device_num.value))
        )
    return devices


def device_count() -> int:
    """Return the number of visible NPU devices."""
    return len(device_pairs())


def chip_info(device: DcmDevice) -> dict[str, str | int | NaType]:
    """Return chip info for a device."""
    if unsafe_metric_enabled('chip-v2'):
        try:
            info_v2 = _DcmiChipInfoV2()
            _call(
                'dcmi_get_device_chip_info_v2',
                device.card_id,
                device.device_id,
                ctypes.byref(info_v2),
            )
            return {
                'chip_type': _decode_bytes(info_v2.chip_type) or NA,
                'chip_name': _decode_bytes(info_v2.chip_name) or NA,
                'chip_version': _decode_bytes(info_v2.chip_ver) or NA,
                'aicore_count': int(info_v2.aicore_cnt),
                'npu_name': _decode_bytes(info_v2.npu_name) or NA,
            }
        except DCMIError:
            pass

    info = _DcmiChipInfo()
    _call('dcmi_get_device_chip_info', device.card_id, device.device_id, ctypes.byref(info))
    return {
        'chip_type': _decode_bytes(info.chip_type) or NA,
        'chip_name': _decode_bytes(info.chip_name) or NA,
        'chip_version': _decode_bytes(info.chip_ver) or NA,
        'aicore_count': int(info.aicore_cnt),
        'npu_name': NA,
    }


def pcie_info(device: DcmDevice) -> PcieInfo | NaType:
    """Return PCIe information for a device."""
    if not unsafe_metric_enabled('pcie'):
        return NA
    info = _DcmiPcieInfo()
    try:
        _call('dcmi_get_device_pcie_info', device.card_id, device.device_id, ctypes.byref(info))
    except DCMIError:
        return NA
    return PcieInfo(
        device_id=int(info.deviceid),
        vendor_id=int(info.venderid),
        subvendor_id=int(info.subvenderid),
        subdevice_id=int(info.subdeviceid),
        bdf_bus_id=int(info.bdf_busid),
        bdf_device_id=int(info.bdf_deviceid),
        bdf_function_id=int(info.bdf_funcid),
    )


def board_info(device: DcmDevice) -> BoardInfo | NaType:
    """Return board and slot information for a device."""
    if not unsafe_metric_enabled('board'):
        return NA
    info = _DcmiBoardInfo()
    try:
        _call('dcmi_get_device_board_info', device.card_id, device.device_id, ctypes.byref(info))
    except DCMIError:
        return NA
    return BoardInfo(
        board_id=int(info.board_id),
        pcb_id=int(info.pcb_id),
        bom_id=int(info.bom_id),
        slot_id=int(info.slot_id),
    )


def logic_id(device: DcmDevice) -> int | NaType:
    """Return the Ascend logical device ID."""
    if not unsafe_metric_enabled('logic'):
        return NA
    value = ctypes.c_int()
    try:
        _call('dcmi_get_device_logic_id', ctypes.byref(value), device.card_id, device.device_id)
    except DCMIError:
        return NA
    return int(value.value)


def hbm_info(device: DcmDevice) -> HbmInfo:
    """Return HBM/on-chip memory details for a device."""
    info = _DcmiHbmInfo()
    try:
        _call('dcmi_get_device_hbm_info', device.card_id, device.device_id, ctypes.byref(info))
    except DCMINotSupported:
        _call('dcmi_get_hbm_info', device.card_id, device.device_id, ctypes.byref(info))

    total = int(info.memory_size) * 1024 * 1024
    used = int(info.memory_usage) * 1024 * 1024
    return HbmInfo(
        total=total,
        used=used,
        free=max(0, total - used),
        bandwidth_utilization=int(info.bandwith_util_rate),
        memory_clock=int(info.freq),
        temperature=int(info.temp),
    )


def frequency(device: DcmDevice, freq_type: int) -> int | NaType:
    """Return a device frequency in MHz."""
    value = ctypes.c_uint()
    try:
        _call(
            'dcmi_get_device_frequency',
            device.card_id,
            device.device_id,
            freq_type,
            ctypes.byref(value),
        )
    except DCMIError:
        return NA
    return int(value.value)


def utilization_rate(device: DcmDevice, utilization_type: int) -> int | NaType:
    """Return a device utilization rate in percent."""
    value = ctypes.c_uint()
    try:
        _call(
            'dcmi_get_device_utilization_rate',
            device.card_id,
            device.device_id,
            utilization_type,
            ctypes.byref(value),
        )
    except DCMIError:
        return NA
    return int(value.value)


def component_utilization(device: DcmDevice) -> ComponentUtilization:
    """Return utilization for the useful Ascend compute/memory components."""
    extended = unsafe_metric_enabled('component')
    return ComponentUtilization(
        npu=utilization_rate(device, DCMI_UTILIZATION_RATE_NPU),
        ai_core=utilization_rate(device, DCMI_UTILIZATION_RATE_AICORE),
        vector_core=(
            utilization_rate(device, DCMI_UTILIZATION_RATE_VECTOR_CORE) if extended else NA
        ),
        ai_cpu=utilization_rate(device, DCMI_UTILIZATION_RATE_AICPU) if extended else NA,
        control_cpu=utilization_rate(device, DCMI_UTILIZATION_RATE_CTRL_CPU) if extended else NA,
        memory=utilization_rate(device, DCMI_UTILIZATION_RATE_MEMORY) if extended else NA,
        memory_bandwidth=(
            utilization_rate(device, DCMI_UTILIZATION_RATE_MEMORY_BANDWIDTH) if extended else NA
        ),
        on_chip_memory=(
            utilization_rate(device, DCMI_UTILIZATION_RATE_ON_CHIP_MEMORY) if extended else NA
        ),
        on_chip_memory_bandwidth=(
            utilization_rate(device, DCMI_UTILIZATION_RATE_ON_CHIP_MEMORY_BANDWIDTH)
            if extended
            else NA
        ),
        ddr=utilization_rate(device, DCMI_UTILIZATION_RATE_DDR) if extended else NA,
    )


def temperature(device: DcmDevice) -> int | NaType:
    """Return chip temperature in Celsius."""
    value = ctypes.c_int()
    try:
        _call('dcmi_get_device_temperature', device.card_id, device.device_id, ctypes.byref(value))
    except DCMIError:
        return NA
    return int(value.value)


def hbm_temperature(device: DcmDevice) -> int | NaType:
    """Return HBM/on-chip memory temperature in Celsius."""
    try:
        return hbm_info(device).temperature
    except DCMIError:
        return NA


def power_usage(device: DcmDevice) -> int | NaType:
    """Return power draw in milliwatts."""
    value = ctypes.c_int()
    try:
        _call('dcmi_get_device_power_info', device.card_id, device.device_id, ctypes.byref(value))
    except DCMIError:
        return NA
    return int(value.value) * 100


def _health_to_string(value: int) -> str | NaType:
    return {
        0: 'OK',
        1: 'MIN',
        2: 'MAJ',
        3: 'CRIT',
        0xFFFFFFFF: 'MISS',
    }.get(value, NA)


def health_status(device: DcmDevice) -> str | NaType:
    """Return the device health status as a compact string."""
    if not unsafe_metric_enabled('health'):
        return NA
    value = ctypes.c_uint()
    try:
        _call('dcmi_get_device_health', device.card_id, device.device_id, ctypes.byref(value))
    except DCMIError:
        return NA
    return _health_to_string(int(value.value))


def driver_health_status() -> str | NaType:
    """Return the driver health status as a compact string."""
    if not unsafe_metric_enabled('driver-health'):
        return NA
    value = ctypes.c_uint()
    try:
        _call('dcmi_get_driver_health', ctypes.byref(value))
    except DCMIError:
        return NA
    return _health_to_string(int(value.value))


def ecc_info(device: DcmDevice, device_type: int = DCMI_DEVICE_TYPE_HBM) -> EccInfo | NaType:
    """Return ECC information for HBM or DDR."""
    del device, device_type
    return NA


def boot_status(device: DcmDevice) -> str | NaType:
    """Return a compact boot status string."""
    if not unsafe_metric_enabled('boot'):
        return NA
    value = ctypes.c_int()
    try:
        _call('dcmi_get_device_boot_status', device.card_id, device.device_id, ctypes.byref(value))
    except DCMIError:
        return NA
    return {
        0: 'UNINIT',
        1: 'BIOS',
        2: 'OS',
        3: 'READY',
        16: 'DCMI',
    }.get(int(value.value), str(int(value.value)))


def work_mode(device: DcmDevice) -> str | NaType:
    """Return the NPU card work mode."""
    if not unsafe_metric_enabled('work'):
        return NA
    value = ctypes.c_ubyte()
    try:
        _call('dcmi_get_npu_work_mode', device.card_id, ctypes.byref(value))
    except DCMIError:
        return NA
    return {0: 'AMP', 1: 'SMP'}.get(int(value.value), str(int(value.value)))


def running_processes(device: DcmDevice) -> list[ProcessInfo]:
    """Return host PIDs and NPU memory usage for processes on the device."""
    proc_info = (_DcmiProcMemInfo * MAX_PROC_NUM)()
    proc_num = ctypes.c_int(0)
    try:
        _call(
            'dcmi_get_device_resource_info',
            device.card_id,
            device.device_id,
            proc_info,
            ctypes.byref(proc_num),
        )
    except DCMIError:
        return []

    processes = []
    for i in range(min(MAX_PROC_NUM, max(0, proc_num.value))):
        pid = int(proc_info[i].proc_id)
        if pid > 0:
            processes.append(ProcessInfo(pid=pid, memory=int(proc_info[i].proc_mem_usage)))
    return processes
