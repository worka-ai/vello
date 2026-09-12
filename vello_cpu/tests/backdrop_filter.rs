// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! CSS `backdrop-filter` on the CPU renderer.

use vello_cpu::color::palette::css;
use vello_cpu::filter_effects::{Filter, FilterFunction};
use vello_cpu::kurbo::{Rect, Shape};
use vello_cpu::peniko::color::PremulRgba8;
use vello_cpu::{Pixmap, RenderContext, Resources};

const W: u16 = 64;
const H: u16 = 16;

fn render(ctx: &mut RenderContext) -> Pixmap {
    let mut pixmap = Pixmap::new(W, H);
    let mut resources = Resources::new();
    ctx.flush();
    ctx.render(&mut pixmap, &mut resources);
    pixmap
}

fn split_red_blue(ctx: &mut RenderContext) {
    ctx.set_paint(css::RED);
    ctx.fill_rect(&Rect::new(0.0, 0.0, 32.0, f64::from(H)));
    ctx.set_paint(css::BLUE);
    ctx.fill_rect(&Rect::new(32.0, 0.0, f64::from(W), f64::from(H)));
}

fn clip(x0: f64, x1: f64) -> vello_cpu::kurbo::BezPath {
    Rect::new(x0, 0.0, x1, f64::from(H)).to_path(0.1)
}

#[test]
fn blur_mixes_across_the_boundary_inside_the_clip_only() {
    let mut ctx = RenderContext::new(W, H);
    split_red_blue(&mut ctx);
    ctx.apply_backdrop_filter(
        &clip(16.0, 48.0),
        Filter::from_function(FilterFunction::Blur { radius: 4.0 }),
    );
    let pixmap = render(&mut ctx);

    let boundary = pixmap.sample(32, 8);
    assert!(
        boundary.r > 40 && boundary.b > 40,
        "the boundary inside the clip should mix red and blue, got {boundary:?}"
    );
    assert_eq!(
        pixmap.sample(4, 8),
        PremulRgba8 {
            r: 255,
            g: 0,
            b: 0,
            a: 255
        }
    );
    assert_eq!(
        pixmap.sample(60, 8),
        PremulRgba8 {
            r: 0,
            g: 0,
            b: 255,
            a: 255
        }
    );
}

#[test]
fn colour_filters_apply_inside_the_clip_only() {
    let mut ctx = RenderContext::new(W, H);
    split_red_blue(&mut ctx);
    ctx.apply_backdrop_filter(
        &clip(0.0, 16.0),
        Filter::from_function(FilterFunction::Grayscale { amount: 1.0 }),
    );
    let pixmap = render(&mut ctx);

    let inside = pixmap.sample(8, 8);
    assert!(
        inside.r.abs_diff(inside.g) <= 1 && inside.g.abs_diff(inside.b) <= 1,
        "grayscale must equalise channels inside the clip, got {inside:?}"
    );
    assert_eq!(
        pixmap.sample(24, 8),
        PremulRgba8 {
            r: 255,
            g: 0,
            b: 0,
            a: 255
        }
    );
}

#[test]
fn chained_functions_apply_in_order() {
    let mut ctx = RenderContext::new(W, H);
    split_red_blue(&mut ctx);
    ctx.apply_backdrop_filter(
        &clip(16.0, 48.0),
        Filter::from_functions([
            FilterFunction::Blur { radius: 4.0 },
            FilterFunction::Grayscale { amount: 1.0 },
        ]),
    );
    let pixmap = render(&mut ctx);

    let boundary = pixmap.sample(32, 8);
    assert!(
        boundary.r.abs_diff(boundary.b) <= 1 && boundary.r > 0,
        "blurred then desaturated, the boundary should be a neutral grey, got {boundary:?}"
    );
}

#[test]
fn a_backdrop_inside_a_layer_sees_only_that_layers_content() {
    let mut ctx = RenderContext::new(W, H);
    ctx.set_paint(css::RED);
    ctx.fill_rect(&Rect::new(0.0, 0.0, f64::from(W), f64::from(H)));

    ctx.push_opacity_layer(1.0);
    ctx.set_paint(css::LIME);
    ctx.fill_rect(&Rect::new(0.0, 0.0, 16.0, f64::from(H)));
    // The clip covers the lime rect and a transparent part of the layer. The red root is not
    // part of this layer's backdrop, so it must not be desaturated.
    ctx.apply_backdrop_filter(
        &clip(0.0, 40.0),
        Filter::from_function(FilterFunction::Grayscale { amount: 1.0 }),
    );
    ctx.pop_layer();
    let pixmap = render(&mut ctx);

    assert_eq!(
        pixmap.sample(32, 8),
        PremulRgba8 {
            r: 255,
            g: 0,
            b: 0,
            a: 255
        },
        "content outside the layer must not be filtered"
    );
    let lime = pixmap.sample(8, 8);
    assert!(
        lime.r.abs_diff(lime.g) <= 1,
        "the layer's own content must be filtered, got {lime:?}"
    );
}

#[test]
fn a_backdrop_over_an_earlier_filter_layer_includes_it() {
    let mut ctx = RenderContext::new(W, H);
    ctx.push_filter_layer(Filter::from_function(FilterFunction::Blur { radius: 1.0 }));
    ctx.set_paint(css::BLUE);
    ctx.fill_rect(&Rect::new(8.0, 4.0, 56.0, 12.0));
    ctx.pop_layer();

    // The backdrop's snapshot references the filter layer above, which is created before it.
    // It must be rasterised first or the backdrop comes out empty.
    ctx.apply_backdrop_filter(
        &clip(16.0, 48.0),
        Filter::from_function(FilterFunction::Grayscale { amount: 1.0 }),
    );
    let pixmap = render(&mut ctx);

    let inside = pixmap.sample(32, 8);
    assert!(
        inside.a > 200,
        "the filtered backdrop must not be empty, got {inside:?}"
    );
    assert!(
        inside.r.abs_diff(inside.b) <= 1,
        "and it must be desaturated, got {inside:?}"
    );
}
