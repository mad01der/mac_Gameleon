from __future__ import annotations

import argparse
import statistics

from mlx_gameleon.codec import GameleonCodec, GameleonGeometryCodec
from mlx_gameleon.data import make_dummy_point_cloud


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--points', type=int, default=10000)
    parser.add_argument('--channels', type=int, default=32)
    parser.add_argument('--kernel-size', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--iters', type=int, default=3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument(
        '--with-attributes',
        action='store_true',
        help='run combined geometry + attribute decode workload',
    )
    args = parser.parse_args()

    xyz = make_dummy_point_cloud(args.points, seed=args.seed)
    codec_cls = (
        GameleonCodec if args.with_attributes else GameleonGeometryCodec
    )
    codec = codec_cls(
        channels=args.channels,
        kernel_size=args.kernel_size,
    )

    last_stream = b''
    for _ in range(args.warmup):
        encoded = codec.compress(xyz)
        last_stream = encoded.byte_stream
        codec.decompress(last_stream)

    enc_times = []
    dec_times = []
    layers = 0
    stream_size = 0
    attr_rows = 0
    for _ in range(args.iters):
        encoded = codec.compress(xyz)
        decoded_result = codec.decompress(encoded.byte_stream)
        if args.with_attributes:
            decoded, attributes, dec_time = decoded_result
            attr_rows = int(attributes.features.shape[0])
        else:
            decoded, dec_time = decoded_result
        enc_times.append(encoded.seconds * 1000)
        dec_times.append(dec_time * 1000)
        layers = encoded.layers
        stream_size = len(encoded.byte_stream)
        if decoded.shape[0] != xyz.shape[0]:
            raise RuntimeError(
                f'decode point count mismatch: {decoded.shape[0]} != {xyz.shape[0]}'
            )

    print(
        'points,channels,layers,stream_bytes,attribute_rows,encode_ms,'
        'decode_ms,encode_std_ms,decode_std_ms'
    )
    print(
        f'{args.points},{args.channels},{layers},{stream_size},{attr_rows},'
        f'{statistics.mean(enc_times):.3f},{statistics.mean(dec_times):.3f},'
        f'{_std(enc_times):.3f},{_std(dec_times):.3f}'
    )


def _std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


if __name__ == '__main__':
    main()
