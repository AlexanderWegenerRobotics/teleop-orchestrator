"""Read-only attach to the sim's named-shared-memory camera ring buffers (see
teleop-simulator/include/pipeline/shared_memory.hpp -- SharedFrameBuffer /
SharedMemoryWriter). Windows-only, matching the sim's own implementation.

Layout (no padding -- all header fields are 4-byte aligned uint32):
    write_idx    u32   (atomic on the writer side; plain read here is fine,
                         see _read_header)
    frame_count  u32
    width        u32
    height       u32
    slots        u8[SHM_N_SLOTS][SHM_MAX_W * SHM_MAX_H * SHM_CHANNELS]

We only ever attach to a mapping the sim process creates -- never create one
ourselves (OpenFileMapping fails cleanly if it doesn't exist yet, unlike
mmap.mmap's create-or-open semantics), so orchestrator/sim start order
doesn't matter: try_open()/read() just fail soft until the sim shows up.
"""

from __future__ import annotations

import ctypes
import sys
import time
from typing import Optional

import numpy as np

_IS_WINDOWS = sys.platform == "win32"
if _IS_WINDOWS:
    import ctypes.wintypes as wt

SHM_N_SLOTS = 3
SHM_MAX_W = 1280
SHM_MAX_H = 960
SHM_CHANNELS = 3
_HEADER_SIZE = 16  # write_idx, frame_count, width, height (u32 each)
_SLOT_SIZE = SHM_MAX_W * SHM_MAX_H * SHM_CHANNELS
_BUFFER_SIZE = _HEADER_SIZE + SHM_N_SLOTS * _SLOT_SIZE

_FILE_MAP_READ = 0x0004

# Deferred to platform check below, not import time: this module (and
# anything that merely imports LiveSource alongside it) should still be
# importable on non-Windows dev machines/CI for testing the rest of the live/
# package -- only actually constructing/using a SharedFrameReader requires
# Windows, matching the sim's own SharedMemoryWriter.
_kernel32 = None
if _IS_WINDOWS:
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.OpenFileMappingW.restype = wt.HANDLE
    _kernel32.OpenFileMappingW.argtypes = [wt.DWORD, wt.BOOL, wt.LPCWSTR]
    _kernel32.MapViewOfFile.restype = ctypes.c_void_p
    _kernel32.MapViewOfFile.argtypes = [wt.HANDLE, wt.DWORD, wt.DWORD, wt.DWORD, ctypes.c_size_t]
    # Without explicit argtypes, ctypes assumes a bare Python int argument is
    # a 32-bit C int -- self._view is a real 64-bit pointer, so close() would
    # overflow converting it (that's the OverflowError this fixes).
    _kernel32.UnmapViewOfFile.restype = wt.BOOL
    _kernel32.UnmapViewOfFile.argtypes = [ctypes.c_void_p]
    _kernel32.CloseHandle.restype = wt.BOOL
    _kernel32.CloseHandle.argtypes = [wt.HANDLE]


class SharedFrameReader:
    """Latest-frame reader for one camera's SHM ring buffer (shm_name in
    pipeline_config_local.yaml, e.g. "avatar_cam", "avatar_wrist_cam_left").
    """

    def __init__(self, name: str, max_staleness_s: float = 1.0):
        if not _IS_WINDOWS:
            raise RuntimeError("SharedFrameReader is Windows-only, matching the sim's SharedMemoryWriter")
        self.name = name
        self.max_staleness_s = max_staleness_s
        self._handle: Optional[int] = None
        self._view: Optional[int] = None
        self._last_frame_count = 0
        self._last_new_frame_time = 0.0

    @property
    def is_open(self) -> bool:
        """Whether we're currently attached to the sim's mapping."""
        return self._view is not None

    def try_open(self) -> bool:
        """Attempts to attach to the sim's mapping; safe to call repeatedly
        before the sim process exists. Returns whether it's open now."""
        if self._view is not None:
            return True
        handle = _kernel32.OpenFileMappingW(_FILE_MAP_READ, False, f"Local\\{self.name}")
        if not handle:
            return False
        view = _kernel32.MapViewOfFile(handle, _FILE_MAP_READ, 0, 0, _BUFFER_SIZE)
        if not view:
            _kernel32.CloseHandle(handle)
            return False
        self._handle = handle
        self._view = view
        self._last_new_frame_time = time.monotonic()
        return True

    def _read_header(self) -> tuple[int, int, int, int]:
        """Returns (write_idx, frame_count, width, height); a plain (non-atomic)
        read of 4-byte-aligned uint32s, which is all we need -- worst case we
        see a value one write behind, and the next poll catches up."""
        fields = (ctypes.c_uint32 * 4).from_address(self._view)
        return fields[0], fields[1], fields[2], fields[3]

    def read(self) -> Optional[np.ndarray]:
        """Returns the latest completed frame as an [H, W, 3] uint8 array
        (copied out of shared memory), or None if: not attached yet (sim not
        started), no frame written yet, or the feed has gone stale (sim died
        or hung -- see max_staleness_s) -- callers must not treat None as
        "frame is black", only as "no frame available this tick"."""
        if self._view is None and not self.try_open():
            return None

        write_idx, frame_count, width, height = self._read_header()

        now = time.monotonic()
        if frame_count != self._last_frame_count:
            self._last_frame_count = frame_count
            self._last_new_frame_time = now
        elif now - self._last_new_frame_time > self.max_staleness_s:
            return None

        if frame_count == 0 or width == 0 or height == 0:
            return None

        # Reader-side slot choice matches SharedMemoryReader::read(): the
        # writer bumps write_idx only after memcpy'ing into slots[write_idx %
        # N], so "one behind" is always a fully completed write, never torn.
        slot = (write_idx - 1) % SHM_N_SLOTS
        slot_addr = self._view + _HEADER_SIZE + slot * _SLOT_SIZE
        size = width * height * SHM_CHANNELS
        raw = (ctypes.c_uint8 * size).from_address(slot_addr)
        return np.ctypeslib.as_array(raw).reshape(height, width, SHM_CHANNELS).copy()

    def close(self) -> None:
        """Detaches from the mapping; safe to call multiple times."""
        if self._view is not None:
            _kernel32.UnmapViewOfFile(self._view)
            self._view = None
        if self._handle is not None:
            _kernel32.CloseHandle(self._handle)
            self._handle = None

    def __enter__(self) -> "SharedFrameReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
