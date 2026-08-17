// Copyright 2023 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::SegmentCount;

use super::{
    BinHeader, Clip, ClipBbox, ClipBic, ClipElement, DrawBbox, DrawMonoid, Layout, LineSoup, Path,
    PathBbox, PathMonoid, PathSegment, Tile,
};
use bytemuck::{Pod, Zeroable};

const TILE_WIDTH: u32 = 16;
const TILE_HEIGHT: u32 = 16;
// Keep in sync with `vello_shaders/shader/shared/ptcl.wgsl`.
const PTCL_INITIAL_ALLOC: u32 = 64;

// TODO: Obtain these from the vello_shaders crate
pub(crate) const PATH_REDUCE_WG: u32 = 256;
const PATH_BBOX_WG: u32 = 256;
const FLATTEN_WG: u32 = 256;
const CLIP_REDUCE_WG: u32 = 256;

/// Counters for tracking dynamic allocation on the GPU.
///
/// This must be kept in sync with the struct in `shader/shared/bump.wgsl`
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
#[repr(C)]
pub struct BumpAllocators {
    pub failed: u32,
    // Final needed dynamic size of the buffers. If any of these are larger
    // than the corresponding `_size` element reallocation needs to occur.
    pub binning: u32,
    pub ptcl: u32,
    pub tile: u32,
    pub seg_counts: u32,
    pub segments: u32,
    pub blend: u32,
    pub lines: u32,
}

#[derive(Default)]
pub struct BumpAllocatorMemory {
    pub total: u32,
    pub binning: BufferSize<u32>,
    pub ptcl: BufferSize<u32>,
    pub tile: BufferSize<Tile>,
    pub seg_counts: BufferSize<SegmentCount>,
    pub segments: BufferSize<PathSegment>,
    pub blend: BufferSize<u32>,
    pub lines: BufferSize<LineSoup>,
}

impl BumpAllocators {
    pub fn memory(&self) -> BumpAllocatorMemory {
        let binning = BufferSize::new(self.binning);
        let ptcl = BufferSize::new(self.ptcl);
        let tile = BufferSize::new(self.tile);
        let seg_counts = BufferSize::new(self.seg_counts);
        let segments = BufferSize::new(self.segments);
        let blend = BufferSize::new(self.blend);
        let lines = BufferSize::new(self.lines);
        BumpAllocatorMemory {
            total: binning.size_in_bytes()
                + ptcl.size_in_bytes()
                + tile.size_in_bytes()
                + seg_counts.size_in_bytes()
                + segments.size_in_bytes()
                + blend.size_in_bytes()
                + lines.size_in_bytes(),
            binning,
            ptcl,
            tile,
            seg_counts,
            segments,
            blend,
            lines,
        }
    }

    pub fn max_with(self, other: Self) -> Self {
        Self {
            failed: self.failed | other.failed,
            binning: self.binning.max(other.binning),
            ptcl: self.ptcl.max(other.ptcl),
            tile: self.tile.max(other.tile),
            seg_counts: self.seg_counts.max(other.seg_counts),
            segments: self.segments.max(other.segments),
            blend: self.blend.max(other.blend),
            lines: self.lines.max(other.lines),
        }
    }

    pub fn with_margin(self, margin_percent: u32) -> Self {
        Self {
            failed: self.failed,
            binning: grow_by_percent(self.binning, margin_percent),
            ptcl: grow_by_percent(self.ptcl, margin_percent),
            tile: grow_by_percent(self.tile, margin_percent),
            seg_counts: grow_by_percent(self.seg_counts, margin_percent),
            segments: grow_by_percent(self.segments, margin_percent),
            blend: grow_by_percent(self.blend, margin_percent),
            lines: grow_by_percent(self.lines, margin_percent),
        }
    }

    pub fn exceeds(self, sizes: &BufferSizes, layout: &Layout) -> bool {
        self.binning > sizes.bin_data.len().saturating_sub(layout.bin_data_start)
            || self.ptcl > sizes.ptcl.len()
            || self.tile > sizes.tiles.len()
            || self.seg_counts > sizes.seg_counts.len()
            || self.segments > sizes.segments.len()
            || self.blend > sizes.blend_spill.len()
            || self.lines > sizes.lines.len()
    }
}

impl std::fmt::Display for BumpAllocatorMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "\n \
                 \tTotal:\t\t\t{} bytes ({:.2} KB | {:.2} MB)\n\
                 \tBinning\t\t\t{} elements ({} bytes)\n\
                 \tPTCL\t\t\t{} elements ({} bytes)\n\
                 \tTile:\t\t\t{} elements ({} bytes)\n\
                 \tSegment Counts:\t\t{} elements ({} bytes)\n\
                 \tSegments:\t\t{} elements ({} bytes)\n\
                 \tBlend:\t\t\t{} elements ({} bytes)\n\
                 \tLines:\t\t\t{} elements ({} bytes)",
            self.total,
            self.total as f32 / (1 << 10) as f32,
            self.total as f32 / (1 << 20) as f32,
            self.binning.len(),
            self.binning.size_in_bytes(),
            self.ptcl.len(),
            self.ptcl.size_in_bytes(),
            self.tile.len(),
            self.tile.size_in_bytes(),
            self.seg_counts.len(),
            self.seg_counts.size_in_bytes(),
            self.segments.len(),
            self.segments.size_in_bytes(),
            self.blend.len(),
            self.blend.size_in_bytes(),
            self.lines.len(),
            self.lines.size_in_bytes()
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BufferSizingMode {
    VelloEstimate,
    CallerEstimate,
    ExactUpperBound,
}

impl Default for BufferSizingMode {
    fn default() -> Self {
        Self::CallerEstimate
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DynamicBufferPolicy {
    pub sizing: BufferSizingMode,
    pub safety_margin_percent: u32,
    pub allow_grow_retry: bool,
    pub max_dynamic_bytes: Option<u64>,
}

impl Default for DynamicBufferPolicy {
    fn default() -> Self {
        Self {
            sizing: BufferSizingMode::CallerEstimate,
            safety_margin_percent: 25,
            allow_grow_retry: true,
            max_dynamic_bytes: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct TargetProfile {
    pub width_px: u32,
    pub height_px: u32,
    pub scale_factor: f32,
    pub dirty_tiles: Option<u32>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct TileCoverageProfile {
    pub tile_width: u32,
    pub tile_height: u32,
    pub target_tiles: u32,
    pub visible_tiles: u32,
    pub total_draw_tile_coverage: u32,
    pub total_path_tile_coverage: u32,
    pub max_ops_per_tile: u32,
    pub max_blend_depth: u32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SceneComplexityProfile {
    pub draw_ops: u32,
    pub clip_ops: u32,
    pub max_clip_depth: u32,
    pub path_ops: u32,
    pub path_points: u32,
    pub estimated_path_segments: u32,
    pub glyph_runs: u32,
    pub glyphs: u32,
    pub images: u32,
    pub image_bytes: u64,
    pub blend_ops: u32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RenderWorkloadProfile {
    pub target: TargetProfile,
    pub coverage: TileCoverageProfile,
    pub scene: SceneComplexityProfile,
    pub policy: DynamicBufferPolicy,
}

/// Storage of indirect dispatch size values.
///
/// The original plan was to reuse [`BumpAllocators`], but the WebGPU compatible
/// usage list rules forbid that being used as indirect counts while also
/// bound as writable.
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
#[repr(C)]
pub struct IndirectCount {
    pub count_x: u32,
    pub count_y: u32,
    pub count_z: u32,
    pub pad0: u32,
}

/// Uniform render configuration data used by all GPU stages.
///
/// This data structure must be kept in sync with the definition in
/// `shaders/shared/config.wgsl`.
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
#[repr(C)]
pub struct ConfigUniform {
    /// Width of the scene in tiles.
    pub width_in_tiles: u32,
    /// Height of the scene in tiles.
    pub height_in_tiles: u32,
    /// Width of the target in pixels.
    pub target_width: u32,
    /// Height of the target in pixels.
    pub target_height: u32,
    /// The base background color applied to the target before any blends.
    pub base_color: u32,
    /// Layout of packed scene data.
    pub layout: Layout,
    /// Size of line soup buffer allocation (in [`LineSoup`]s)
    pub lines_size: u32,
    /// Size of binning buffer allocation (in `u32`s).
    pub binning_size: u32,
    /// Size of tile buffer allocation (in [`Tile`]s).
    pub tiles_size: u32,
    /// Size of segment count buffer allocation (in [`SegmentCount`]s).
    pub seg_counts_size: u32,
    /// Size of segment buffer allocation (in [`PathSegment`]s).
    pub segments_size: u32,
    /// Size of blend spill buffer (in `u32` pixels).
    // TODO: Maybe store in TILE_WIDTH * TILE_HEIGHT blocks of pixels instead?
    pub blend_size: u32,
    /// Size of per-tile command list buffer allocation (in `u32`s).
    pub ptcl_size: u32,
}

/// CPU side setup and configuration.
#[derive(Default)]
pub struct RenderConfig {
    /// GPU side configuration.
    pub gpu: ConfigUniform,
    /// Workgroup counts for all compute pipelines.
    pub workgroup_counts: WorkgroupCounts,
    /// Sizes of all buffer resources.
    pub buffer_sizes: BufferSizes,
}

impl RenderConfig {
    pub fn new(layout: &Layout, width: u32, height: u32, base_color: &peniko::Color) -> Self {
        Self::new_with_profile(layout, width, height, base_color, None, None)
    }

    pub fn new_with_profile(
        layout: &Layout,
        width: u32,
        height: u32,
        base_color: &peniko::Color,
        profile: Option<&RenderWorkloadProfile>,
        minimum_bump: Option<BumpAllocators>,
    ) -> Self {
        let new_width = width.next_multiple_of(TILE_WIDTH);
        let new_height = height.next_multiple_of(TILE_HEIGHT);
        let width_in_tiles = new_width / TILE_WIDTH;
        let height_in_tiles = new_height / TILE_HEIGHT;
        let n_path_tags = layout.path_tags_size();
        let workgroup_counts =
            WorkgroupCounts::new(layout, width_in_tiles, height_in_tiles, n_path_tags);
        let buffer_sizes =
            BufferSizes::new_with_profile(layout, &workgroup_counts, profile, minimum_bump);
        Self {
            gpu: ConfigUniform {
                width_in_tiles,
                height_in_tiles,
                target_width: width,
                target_height: height,
                base_color: base_color.premultiply().to_rgba8().to_u32(),
                lines_size: buffer_sizes.lines.len(),
                binning_size: buffer_sizes.bin_data.len() - layout.bin_data_start,
                tiles_size: buffer_sizes.tiles.len(),
                seg_counts_size: buffer_sizes.seg_counts.len(),
                segments_size: buffer_sizes.segments.len(),
                blend_size: buffer_sizes.blend_spill.len(),
                ptcl_size: buffer_sizes.ptcl.len(),
                layout: *layout,
            },
            workgroup_counts,
            buffer_sizes,
        }
    }
}

/// Type alias for a workgroup size.
pub type WorkgroupSize = (u32, u32, u32);

/// Computed sizes for all dispatches.
#[derive(Copy, Clone, Debug, Default)]
pub struct WorkgroupCounts {
    pub use_large_path_scan: bool,
    pub path_reduce: WorkgroupSize,
    pub path_reduce2: WorkgroupSize,
    pub path_scan1: WorkgroupSize,
    pub path_scan: WorkgroupSize,
    pub bbox_clear: WorkgroupSize,
    pub flatten: WorkgroupSize,
    pub draw_reduce: WorkgroupSize,
    pub draw_leaf: WorkgroupSize,
    pub clip_reduce: WorkgroupSize,
    pub clip_leaf: WorkgroupSize,
    pub binning: WorkgroupSize,
    pub tile_alloc: WorkgroupSize,
    pub path_count_setup: WorkgroupSize,
    // Note: `path_count` must use an indirect dispatch
    pub backdrop: WorkgroupSize,
    pub coarse: WorkgroupSize,
    pub path_tiling_setup: WorkgroupSize,
    // Note: `path_tiling` must use an indirect dispatch
    pub fine: WorkgroupSize,
}

impl WorkgroupCounts {
    pub fn new(
        layout: &Layout,
        width_in_tiles: u32,
        height_in_tiles: u32,
        n_path_tags: u32,
    ) -> Self {
        let n_paths = layout.n_paths;
        let n_draw_objects = layout.n_draw_objects;
        let n_clips = layout.n_clips;
        let path_tag_padded = align_up(n_path_tags, 4 * PATH_REDUCE_WG);
        let path_tag_wgs = path_tag_padded / (4 * PATH_REDUCE_WG);
        let use_large_path_scan = path_tag_wgs > PATH_REDUCE_WG;
        let reduced_size = if use_large_path_scan {
            align_up(path_tag_wgs, PATH_REDUCE_WG)
        } else {
            path_tag_wgs
        };
        let draw_object_wgs = n_draw_objects.div_ceil(PATH_BBOX_WG);
        let draw_monoid_wgs = draw_object_wgs.min(PATH_BBOX_WG);
        let flatten_wgs = n_path_tags.div_ceil(FLATTEN_WG);
        let clip_reduce_wgs = n_clips.saturating_sub(1) / CLIP_REDUCE_WG;
        let clip_wgs = n_clips.div_ceil(CLIP_REDUCE_WG);
        let path_wgs = n_paths.div_ceil(PATH_BBOX_WG);
        let width_in_bins = width_in_tiles.div_ceil(16);
        let height_in_bins = height_in_tiles.div_ceil(16);
        Self {
            use_large_path_scan,
            path_reduce: (path_tag_wgs, 1, 1),
            path_reduce2: (PATH_REDUCE_WG, 1, 1),
            path_scan1: (reduced_size / PATH_REDUCE_WG, 1, 1),
            path_scan: (path_tag_wgs, 1, 1),
            bbox_clear: (draw_object_wgs, 1, 1),
            flatten: (flatten_wgs, 1, 1),
            draw_reduce: (draw_monoid_wgs, 1, 1),
            draw_leaf: (draw_monoid_wgs, 1, 1),
            clip_reduce: (clip_reduce_wgs, 1, 1),
            clip_leaf: (clip_wgs, 1, 1),
            binning: (draw_object_wgs, 1, 1),
            tile_alloc: (path_wgs, 1, 1),
            path_count_setup: (1, 1, 1),
            backdrop: (path_wgs, 1, 1),
            coarse: (width_in_bins, height_in_bins, 1),
            path_tiling_setup: (1, 1, 1),
            fine: (width_in_tiles, height_in_tiles, 1),
        }
    }
}

/// Typed buffer size primitive.
#[derive(Copy, Clone, Eq, Default, Debug)]
pub struct BufferSize<T: Sized> {
    len: u32,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Sized> BufferSize<T> {
    /// Creates a new buffer size from number of elements.
    pub const fn new(len: u32) -> Self {
        Self {
            // Each buffer binding must be large enough to hold at least one element to avoid
            // triggering validation errors.
            //
            // Note: not using `Ord::max` here because it doesn't support const eval yet (except
            // in nightly)
            len: if len > 0 { len } else { 1 },
            _phantom: std::marker::PhantomData,
        }
    }

    /// Creates a new buffer size from size in bytes.
    pub const fn from_size_in_bytes(size: u32) -> Self {
        Self::new(size / size_of::<T>() as u32)
    }

    /// Returns the number of elements.
    #[expect(clippy::len_without_is_empty, reason = "The buffer can never be empty")]
    pub const fn len(self) -> u32 {
        self.len
    }

    /// Returns the size in bytes.
    pub const fn size_in_bytes(self) -> u32 {
        size_of::<T>() as u32 * self.len
    }

    /// Returns the size in bytes aligned up to the given value.
    pub const fn aligned_in_bytes(self, alignment: u32) -> u32 {
        align_up(self.size_in_bytes(), alignment)
    }
}

impl<T: Sized> PartialEq for BufferSize<T> {
    fn eq(&self, other: &Self) -> bool {
        self.len == other.len
    }
}

impl<T: Sized> PartialOrd for BufferSize<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.len.partial_cmp(&other.len)
    }
}

/// Computed sizes for all buffers.
#[derive(Copy, Clone, Debug, Default)]
pub struct BufferSizes {
    // Known size buffers
    pub path_reduced: BufferSize<PathMonoid>,
    pub path_reduced2: BufferSize<PathMonoid>,
    pub path_reduced_scan: BufferSize<PathMonoid>,
    pub path_monoids: BufferSize<PathMonoid>,
    pub path_bboxes: BufferSize<PathBbox>,
    pub draw_reduced: BufferSize<DrawMonoid>,
    pub draw_monoids: BufferSize<DrawMonoid>,
    pub info: BufferSize<u32>,
    pub clip_inps: BufferSize<Clip>,
    pub clip_els: BufferSize<ClipElement>,
    pub clip_bics: BufferSize<ClipBic>,
    pub clip_bboxes: BufferSize<ClipBbox>,
    pub draw_bboxes: BufferSize<DrawBbox>,
    pub bump_alloc: BufferSize<BumpAllocators>,
    pub indirect_count: BufferSize<IndirectCount>,
    pub bin_headers: BufferSize<BinHeader>,
    pub paths: BufferSize<Path>,
    // Bump allocated buffers
    pub lines: BufferSize<LineSoup>,
    pub bin_data: BufferSize<u32>,
    pub tiles: BufferSize<Tile>,
    pub seg_counts: BufferSize<SegmentCount>,
    pub segments: BufferSize<PathSegment>,
    pub blend_spill: BufferSize<u32>,
    pub ptcl: BufferSize<u32>,
}

impl BufferSizes {
    pub fn new(layout: &Layout, workgroups: &WorkgroupCounts) -> Self {
        Self::new_with_profile(layout, workgroups, None, None)
    }

    pub fn new_with_profile(
        layout: &Layout,
        workgroups: &WorkgroupCounts,
        profile: Option<&RenderWorkloadProfile>,
        minimum_bump: Option<BumpAllocators>,
    ) -> Self {
        let n_paths = layout.n_paths;
        let n_draw_objects = layout.n_draw_objects;
        let n_clips = layout.n_clips;
        let path_tag_wgs = workgroups.path_reduce.0;
        let reduced_size = if workgroups.use_large_path_scan {
            align_up(path_tag_wgs, PATH_REDUCE_WG)
        } else {
            path_tag_wgs
        };
        let path_reduced = BufferSize::new(reduced_size);
        let path_reduced2 = BufferSize::new(PATH_REDUCE_WG);
        let path_reduced_scan = BufferSize::new(reduced_size);
        let path_monoids = BufferSize::new(path_tag_wgs * PATH_REDUCE_WG);
        let path_bboxes = BufferSize::new(n_paths);
        let binning_wgs = workgroups.binning.0;
        let draw_monoid_wgs = workgroups.draw_reduce.0;
        let draw_reduced = BufferSize::new(draw_monoid_wgs);
        let draw_monoids = BufferSize::new(n_draw_objects);
        let info = BufferSize::new(layout.bin_data_start);
        let clip_inps = BufferSize::new(n_clips);
        let clip_els = BufferSize::new(n_clips);
        let clip_bics = BufferSize::new(n_clips / CLIP_REDUCE_WG);
        let clip_bboxes = BufferSize::new(n_clips);
        let draw_bboxes = BufferSize::new(n_paths);
        let bump_alloc = BufferSize::new(1);
        let indirect_count = BufferSize::new(1);
        let bin_headers = BufferSize::new(binning_wgs * 256);
        let n_paths_aligned = align_up(n_paths, 256);
        let paths = BufferSize::new(n_paths_aligned);

        let has_profile = profile.is_some();
        let profile = profile.copied().unwrap_or_default();
        let policy = if has_profile {
            profile.policy
        } else {
            DynamicBufferPolicy {
                sizing: BufferSizingMode::VelloEstimate,
                ..DynamicBufferPolicy::default()
            }
        };
        let minimum_bump = minimum_bump
            .unwrap_or_default()
            .with_margin(policy.safety_margin_percent);

        let tile_count = workgroups.fine.0.saturating_mul(workgroups.fine.1).max(1);
        let bin_count = workgroups
            .coarse
            .0
            .saturating_mul(workgroups.coarse.1)
            .max(1);
        let profiled_target_tiles = profile.coverage.target_tiles.max(tile_count);
        let visible_tiles = profile
            .coverage
            .visible_tiles
            .max(profiled_target_tiles)
            .max(1);
        let draw_tile_coverage = profile
            .coverage
            .total_draw_tile_coverage
            .max(n_draw_objects)
            .max(visible_tiles);
        let path_tile_coverage = profile
            .coverage
            .total_path_tile_coverage
            .max(n_paths)
            .max(1);
        let blend_depth = profile
            .coverage
            .max_blend_depth
            .max(profile.scene.max_clip_depth);
        let blend_ops = profile.scene.blend_ops.max(blend_depth);
        let glyph_pressure = profile
            .scene
            .glyphs
            .saturating_mul(24)
            .saturating_add(profile.scene.glyph_runs.saturating_mul(64));
        let path_pressure = profile
            .scene
            .estimated_path_segments
            .max(profile.scene.path_points.saturating_mul(2))
            .max(glyph_pressure);
        let caller_path_complexity = path_pressure
            .max(n_paths.saturating_mul(8))
            .max(n_draw_objects.saturating_mul(8))
            .max(1);
        let layout_path_data_words = layout.draw_tag_base.saturating_sub(layout.path_data_base);
        let layout_path_tag_words = layout.path_data_base.saturating_sub(layout.path_tag_base);
        let layout_path_complexity = layout_path_data_words
            .saturating_add(layout_path_tag_words)
            .max(n_paths.saturating_mul(8))
            .max(n_draw_objects.saturating_mul(8))
            .max(1);
        let path_complexity = match policy.sizing {
            BufferSizingMode::VelloEstimate => layout_path_complexity,
            BufferSizingMode::CallerEstimate | BufferSizingMode::ExactUpperBound => {
                layout_path_complexity.max(caller_path_complexity)
            }
        };
        let estimate_margin = if has_profile {
            policy.safety_margin_percent
        } else {
            0
        };
        let default_floor = |profiled, unprofiled| {
            if has_profile { profiled } else { unprofiled }
        };

        let bin_data_len = layout
            .bin_data_start
            .saturating_add(dynamic_buffer_len_with_margin(
                bin_count.saturating_mul(64),
                draw_tile_coverage
                    .saturating_mul(4)
                    .max(path_complexity.saturating_mul(2)),
                minimum_bump.binning.max(default_floor(1 << 14, 1 << 18)),
                estimate_margin,
            ));
        // Tile allocation happens after glyph runs have been resolved, so each glyph outline is a
        // draw object even though the caller retained one text operation. Account for both total
        // retained draw coverage and the resolved glyph count instead of relying on path coverage.
        let caller_tile_demand = draw_tile_coverage
            .saturating_mul(2)
            .saturating_add(profile.scene.glyphs.saturating_mul(16));
        let tiles = BufferSize::new(dynamic_buffer_len_with_margin(
            caller_tile_demand,
            path_tile_coverage
                .saturating_mul(4)
                .max(n_paths.saturating_mul(16))
                .max(visible_tiles),
            minimum_bump.tile.max(default_floor(1 << 14, 1 << 21)),
            estimate_margin,
        ));
        let lines = BufferSize::new(dynamic_buffer_len_with_margin(
            path_complexity.saturating_mul(4),
            path_tile_coverage.saturating_mul(2),
            minimum_bump.lines.max(default_floor(1 << 14, 1 << 21)),
            estimate_margin,
        ));
        let seg_counts = BufferSize::new(dynamic_buffer_len_with_margin(
            tiles.len(),
            path_complexity.saturating_mul(2),
            minimum_bump.seg_counts.max(default_floor(1 << 14, 1 << 21)),
            estimate_margin,
        ));
        let segments = BufferSize::new(
            dynamic_buffer_len_with_margin(
                lines.len(),
                path_complexity.saturating_mul(4),
                minimum_bump.segments.max(default_floor(1 << 14, 1 << 21)),
                estimate_margin,
            )
            .max(lines.len()),
        );
        let blend_spill = BufferSize::new(dynamic_buffer_len_with_margin(
            visible_tiles.saturating_mul(blend_depth.saturating_sub(4)),
            blend_ops.saturating_mul(TILE_WIDTH * TILE_HEIGHT),
            minimum_bump.blend.max(default_floor(1 << 12, 1 << 20)),
            estimate_margin,
        ));
        let ptcl_static = tile_count.saturating_mul(PTCL_INITIAL_ALLOC);
        let unprofiled_ptcl_floor = (1_u32 << 23).saturating_sub(ptcl_static);
        let ptcl_dynamic = dynamic_buffer_len_with_margin(
            draw_tile_coverage.saturating_mul(8),
            visible_tiles
                .saturating_mul(profile.coverage.max_ops_per_tile.max(1).saturating_mul(4)),
            minimum_bump
                .ptcl
                .max(default_floor(1 << 15, unprofiled_ptcl_floor)),
            estimate_margin,
        );
        let ptcl = BufferSize::new(ptcl_static.saturating_add(ptcl_dynamic));
        let bin_data = BufferSize::new(bin_data_len);
        Self {
            path_reduced,
            path_reduced2,
            path_reduced_scan,
            path_monoids,
            path_bboxes,
            draw_reduced,
            draw_monoids,
            info,
            clip_inps,
            clip_els,
            clip_bics,
            clip_bboxes,
            draw_bboxes,
            bump_alloc,
            indirect_count,
            lines,
            bin_headers,
            paths,
            bin_data,
            tiles,
            seg_counts,
            segments,
            blend_spill,
            ptcl,
        }
    }
}

fn dynamic_buffer_len(a: u32, b: u32, floor: u32) -> u32 {
    align_up(a.max(b).max(floor), 256)
}

fn dynamic_buffer_len_with_margin(a: u32, b: u32, floor: u32, margin_percent: u32) -> u32 {
    dynamic_buffer_len(grow_by_percent(a.max(b), margin_percent), 0, floor)
}

fn grow_by_percent(value: u32, percent: u32) -> u32 {
    if value == 0 {
        return 0;
    }
    let grown = (value as u64)
        .saturating_mul(100_u64.saturating_add(percent as u64))
        .div_ceil(100);
    grown.min(u32::MAX as u64) as u32
}

const fn align_up(len: u32, alignment: u32) -> u32 {
    let mask = alignment - 1;
    len.saturating_add(mask) & !mask
}

#[cfg(test)]
mod tests {
    use super::*;

    fn representative_text_layout() -> Layout {
        Layout {
            n_draw_objects: 27,
            n_paths: 17,
            n_clips: 2,
            bin_data_start: 64,
            path_tag_base: 0,
            path_data_base: 128,
            draw_tag_base: 1_024,
            draw_data_base: 1_128,
            transform_base: 1_256,
            style_base: 1_384,
        }
    }

    fn representative_text_profile(margin_percent: u32) -> RenderWorkloadProfile {
        RenderWorkloadProfile {
            target: TargetProfile {
                width_px: 1_720,
                height_px: 1_023,
                scale_factor: 1.0,
                dirty_tiles: None,
            },
            coverage: TileCoverageProfile {
                tile_width: TILE_WIDTH,
                tile_height: TILE_HEIGHT,
                target_tiles: 6_912,
                visible_tiles: 6_912,
                total_draw_tile_coverage: 11_127,
                total_path_tile_coverage: 2_961,
                max_ops_per_tile: 4,
                max_blend_depth: 2,
            },
            scene: SceneComplexityProfile {
                draw_ops: 27,
                path_ops: 17,
                estimated_path_segments: 40,
                glyph_runs: 10,
                glyphs: 326,
                blend_ops: 2,
                ..SceneComplexityProfile::default()
            },
            policy: DynamicBufferPolicy {
                safety_margin_percent: margin_percent,
                ..DynamicBufferPolicy::default()
            },
        }
    }

    #[test]
    fn unprofiled_rendering_keeps_vello_conservative_floors() {
        let layout = representative_text_layout();
        let workgroups = WorkgroupCounts::new(&layout, 108, 64, layout.path_tags_size());
        let sizes = BufferSizes::new_with_profile(&layout, &workgroups, None, None);

        assert_eq!(sizes.bin_data.len() - layout.bin_data_start, 1 << 18);
        assert_eq!(sizes.tiles.len(), 1 << 21);
        assert_eq!(sizes.lines.len(), 1 << 21);
        assert_eq!(sizes.seg_counts.len(), 1 << 21);
        assert_eq!(sizes.segments.len(), 1 << 21);
        assert_eq!(sizes.blend_spill.len(), 1 << 20);
        assert_eq!(sizes.ptcl.len(), 1 << 23);
    }

    #[test]
    fn profiled_tiles_include_draw_coverage_glyphs_and_margin() {
        let layout = representative_text_layout();
        let workgroups = WorkgroupCounts::new(&layout, 108, 64, layout.path_tags_size());
        let without_margin = BufferSizes::new_with_profile(
            &layout,
            &workgroups,
            Some(&representative_text_profile(0)),
            None,
        );
        let with_margin = BufferSizes::new_with_profile(
            &layout,
            &workgroups,
            Some(&representative_text_profile(25)),
            None,
        );

        assert_eq!(without_margin.tiles.len(), 27_648);
        assert_eq!(with_margin.tiles.len(), 34_560);
        assert!(with_margin.tiles > without_margin.tiles);
    }
}
