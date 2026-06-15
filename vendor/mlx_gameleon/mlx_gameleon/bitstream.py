from __future__ import annotations

import json
import struct
from dataclasses import dataclass

import numpy as np

MAGIC = b'MLXGAMELEON01'


@dataclass(frozen=True, slots=True)
class GeometryPayload:
    pos_q: float
    base_coords: np.ndarray
    base_occupancy: np.ndarray
    stage_occupancy: list[np.ndarray]
    attribute_latents: list[np.ndarray] | None = None


def pack_payload(payload: GeometryPayload) -> bytes:
    header = {
        'pos_q': float(payload.pos_q),
        'base_rows': int(payload.base_coords.shape[0]),
        'stages': [
            int(stage.shape[0]) for stage in payload.stage_occupancy
        ],
        'attribute_latents': [
            list(map(int, latent.shape))
            for latent in payload.attribute_latents or []
        ],
    }
    header_bytes = json.dumps(header, separators=(',', ':')).encode()
    chunks = [
        payload.base_coords.astype(np.int32, copy=False).tobytes(),
        payload.base_occupancy.astype(np.uint8, copy=False).tobytes(),
    ]
    chunks.extend(
        stage.astype(np.uint8, copy=False).tobytes()
        for stage in payload.stage_occupancy
    )
    chunks.extend(
        latent.astype(np.int8, copy=False).tobytes()
        for latent in payload.attribute_latents or []
    )
    stream = bytearray(MAGIC)
    stream += struct.pack('<I', len(header_bytes))
    stream += header_bytes
    for chunk in chunks:
        stream += struct.pack('<I', len(chunk))
        stream += chunk
    return bytes(stream)


def unpack_payload(stream: bytes) -> GeometryPayload:
    if not stream.startswith(MAGIC):
        raise ValueError('invalid mlx-gameleon geometry stream')
    cursor = len(MAGIC)
    header_len = struct.unpack('<I', stream[cursor : cursor + 4])[0]
    cursor += 4
    header = json.loads(stream[cursor : cursor + header_len])
    cursor += header_len

    def next_chunk() -> bytes:
        nonlocal cursor
        size = struct.unpack('<I', stream[cursor : cursor + 4])[0]
        cursor += 4
        chunk = stream[cursor : cursor + size]
        cursor += size
        return chunk

    base_rows = int(header['base_rows'])
    base_coords = np.frombuffer(next_chunk(), dtype=np.int32).reshape(
        (base_rows, 3)
    )
    base_occupancy = np.frombuffer(next_chunk(), dtype=np.uint8).reshape(
        (base_rows,)
    )
    stages = []
    for rows in header['stages']:
        stages.append(
            np.frombuffer(next_chunk(), dtype=np.uint8).reshape(
                (int(rows),)
            )
        )
    attribute_latents = []
    for shape in header.get('attribute_latents', []):
        rows, channels = map(int, shape)
        attribute_latents.append(
            np.frombuffer(next_chunk(), dtype=np.int8)
            .reshape((rows, channels))
            .copy()
        )
    return GeometryPayload(
        pos_q=float(header['pos_q']),
        base_coords=base_coords.copy(),
        base_occupancy=base_occupancy.copy(),
        stage_occupancy=[stage.copy() for stage in stages],
        attribute_latents=attribute_latents or None,
    )
