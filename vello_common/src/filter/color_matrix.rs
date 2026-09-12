// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `feColorMatrix` filter primitive.

use crate::peniko::color::PremulRgba8;

/// A 4x5 colour matrix applied to unpremultiplied RGBA.
///
/// Rows are R, G, B, A; columns are R, G, B, A and a constant offset in `[0, 1]` units. Every
/// CSS filter function except `blur` lowers to one of these.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ColorMatrix {
    /// The matrix in row-major order.
    pub matrix: [f32; 20],
}

impl ColorMatrix {
    /// Create a new colour matrix filter.
    pub fn new(matrix: [f32; 20]) -> Self {
        Self { matrix }
    }

    /// Apply the matrix to one unpremultiplied colour with channels in `[0, 1]`.
    ///
    /// The result is clamped to `[0, 1]`, as the specification requires.
    #[inline]
    pub fn apply_unpremultiplied(&self, rgba: [f32; 4]) -> [f32; 4] {
        let m = &self.matrix;
        let [r, g, b, a] = rgba;
        let row = |i: usize| {
            (m[i] * r + m[i + 1] * g + m[i + 2] * b + m[i + 3] * a + m[i + 4]).clamp(0.0, 1.0)
        };
        [row(0), row(5), row(10), row(15)]
    }

    /// Apply the matrix to one premultiplied 8-bit pixel.
    ///
    /// Colour matrices are defined on unpremultiplied colour, so the pixel is unpremultiplied,
    /// transformed and premultiplied again. Fully transparent pixels still go through the
    /// matrix because an alpha row or offset may make them visible.
    #[inline]
    pub fn apply_premul_rgba8(&self, pixel: PremulRgba8) -> PremulRgba8 {
        let alpha = f32::from(pixel.a) / 255.0;
        let unpremultiply = |c: u8| {
            if pixel.a == 0 {
                0.0
            } else {
                (f32::from(c) / 255.0 / alpha).min(1.0)
            }
        };
        let [r, g, b, a] = self.apply_unpremultiplied([
            unpremultiply(pixel.r),
            unpremultiply(pixel.g),
            unpremultiply(pixel.b),
            alpha,
        ]);
        let to_u8 = |v: f32| (v * 255.0 + 0.5) as u8;
        PremulRgba8 {
            r: to_u8(r * a),
            g: to_u8(g * a),
            b: to_u8(b * a),
            a: to_u8(a),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ColorMatrix;
    use crate::peniko::color::PremulRgba8;

    const IDENTITY: [f32; 20] = [
        1.0, 0.0, 0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, 0.0, 0.0, //
        0.0, 0.0, 1.0, 0.0, 0.0, //
        0.0, 0.0, 0.0, 1.0, 0.0,
    ];

    #[test]
    fn identity_leaves_premultiplied_pixels_unchanged() {
        let pixel = PremulRgba8 {
            r: 100,
            g: 50,
            b: 25,
            a: 200,
        };
        assert_eq!(ColorMatrix::new(IDENTITY).apply_premul_rgba8(pixel), pixel);
    }

    #[test]
    fn full_desaturation_produces_equal_channels() {
        let matrix = crate::filter_effects::FilterFunction::Saturate { amount: 0.0 }.to_primitive();
        let crate::filter_effects::FilterPrimitive::ColorMatrix { matrix } = matrix else {
            panic!("saturate lowers to a colour matrix");
        };
        let out = ColorMatrix::new(matrix).apply_unpremultiplied([1.0, 0.0, 0.0, 1.0]);
        assert!((out[0] - out[1]).abs() < 1e-3 && (out[1] - out[2]).abs() < 1e-3);
    }
}
