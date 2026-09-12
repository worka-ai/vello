// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

// `feColorMatrix`, applied to unpremultiplied colour.
//
// The parameter block stores a u32 header followed by twenty f32 matrix values in row-major
// order, so matrix value `i` lives in u32 slot `i + 1` of the block. Keep this in sync with
// `GpuColorMatrix` in `vello_gpu/src/filter.rs`.

fn apply_color_matrix(color: vec4<f32>, matrix: array<f32, 20>) -> vec4<f32> {
    var unpremultiplied = vec4<f32>(0.0);
    if color.a > 0.0 {
        unpremultiplied = vec4<f32>(min(color.rgb / color.a, vec3<f32>(1.0)), color.a);
    }

    var out = vec4<f32>(0.0);
    for (var row = 0u; row < 4u; row += 1u) {
        let base = row * 5u;
        out[row] = clamp(
            matrix[base] * unpremultiplied.r
                + matrix[base + 1u] * unpremultiplied.g
                + matrix[base + 2u] * unpremultiplied.b
                + matrix[base + 3u] * unpremultiplied.a
                + matrix[base + 4u],
            0.0,
            1.0,
        );
    }

    return vec4<f32>(out.rgb * out.a, out.a);
}
