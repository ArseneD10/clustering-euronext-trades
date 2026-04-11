import mlx.core as mx


def tensor_to_int8(array):
    a_min, a_max = mx.min(array), mx.max(array)
    s = (a_max - a_min) / 255
    z = mx.round(-128 - a_min / s)

    q = mx.round(array / s + z)
    q = mx.clip(q, -128, 127)
    return q.astype(mx.int8)