// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `feColorMatrix` filter primitive implementation.

use vello_common::filter::color_matrix::ColorMatrix;
use vello_common::pixmap::Pixmap;

use super::FilterEffect;
use crate::filter::context::ScratchBuffer;

impl FilterEffect for ColorMatrix {
    fn execute_lowp(&self, pixmap: &mut Pixmap, _: &mut ScratchBuffer) {
        apply(self, pixmap);
    }

    fn execute_highp(&self, pixmap: &mut Pixmap, _: &mut ScratchBuffer) {
        apply(self, pixmap);
    }
}

// A colour matrix is a per-pixel operation with no neighbourhood, so both precision paths share
// one implementation. It works in unpremultiplied colour, which the 8-bit pipeline has to
// leave anyway.
fn apply(matrix: &ColorMatrix, pixmap: &mut Pixmap) {
    for pixel in pixmap.data_mut() {
        *pixel = matrix.apply_premul_rgba8(*pixel);
    }
    // The alpha row or offset can introduce or remove transparency.
    pixmap.recompute_may_have_transparency();
}

#[cfg(test)]
mod tests {
    use super::ColorMatrix;
    use crate::filter::FilterEffect;
    use crate::filter::context::ScratchBuffer;
    use vello_common::filter_effects::{FilterFunction, FilterPrimitive};
    use vello_common::peniko::color::PremulRgba8;
    use vello_common::pixmap::Pixmap;

    fn matrix(function: FilterFunction) -> ColorMatrix {
        let FilterPrimitive::ColorMatrix { matrix } = function.to_primitive() else {
            panic!("expected a colour matrix");
        };
        ColorMatrix::new(matrix)
    }

    #[test]
    fn brightness_scales_opaque_colour() {
        let mut pixmap = Pixmap::new(1, 1);
        pixmap.set_pixel(
            0,
            0,
            PremulRgba8 {
                r: 100,
                g: 50,
                b: 20,
                a: 255,
            },
        );
        matrix(FilterFunction::Brightness { amount: 2.0 })
            .execute_lowp(&mut pixmap, &mut ScratchBuffer::new());
        assert_eq!(
            pixmap.sample(0, 0),
            PremulRgba8 {
                r: 200,
                g: 100,
                b: 40,
                a: 255
            }
        );
    }

    #[test]
    fn opacity_scales_alpha_and_keeps_colour_premultiplied() {
        let mut pixmap = Pixmap::new(1, 1);
        pixmap.set_pixel(
            0,
            0,
            PremulRgba8 {
                r: 255,
                g: 0,
                b: 0,
                a: 255,
            },
        );
        matrix(FilterFunction::Opacity { amount: 0.5 })
            .execute_lowp(&mut pixmap, &mut ScratchBuffer::new());
        let out = pixmap.sample(0, 0);
        assert_eq!(out.a, 128);
        assert_eq!(out.r, 128, "premultiplied red must track alpha");
    }
}
