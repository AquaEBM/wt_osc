#![feature(portable_simd, new_uninit, as_array_of_cells)]

extern crate alloc;

mod cluster;
mod oscillator;
mod voice;

mod basic_shapes;
pub mod wavetable;

use alloc::sync::Arc;
use core::{array, cell::Cell, iter, num::NonZeroUsize};
use plugin_util::{
    math::*,
    simd::{prelude::*, Simd, SimdElement, StdFloat},
    simd_util::*,
    smoothing::*,
};
use polygraph::{buffer::Buffers, processor::Processor};

use wavetable::BandLimitedWaveTables;

use cluster::WTOscVoiceCluster;

pub const MAX_UNISON: usize = 16;
pub const NUM_VOICE_OSCILLATORS: usize = enclosing_div(MAX_UNISON, FLOATS_PER_VECTOR);

#[derive(Default)]
pub struct WTOscGlobalState {
    pub table: Box<BandLimitedWaveTables>,
    pub starting_phases: [Float; NUM_VOICE_OSCILLATORS],
    pub num_frames: Float,
    pub sr: f32,
}

#[derive(Default)]
pub struct WTOsc {
    global_state: WTOscGlobalState,
    clusters: Box<[WTOscVoiceCluster]>,
}

impl Processor<FLOATS_PER_VECTOR> for WTOsc {
    fn audio_io_layout(&self) -> (usize, usize) {
        (0, 1)
    }

    fn process(&mut self, buffers: Buffers<Simd<f32, FLOATS_PER_VECTOR>>, cluster_idx: usize) {
        if let Some(output_buf) = buffers.get_output(0) {
            let table = self.global_state.table.as_ref();
            if table.num_frames() != 0 {
                let cluster = &mut self.clusters[cluster_idx];

                cluster.set_params_smoothed(&self.global_state, buffers.buffer_size());

                for sample in output_buf.iter() {
                    sample.set(cluster.process(table));
                }
            }
        }
    }

    fn initialize(&mut self, sr: f32, _max_buffer_size: usize, max_num_clusters: usize) {
        self.global_state.sr = sr;
        self.clusters = iter::repeat_with(Default::default)
            .take(max_num_clusters)
            .collect()
    }

    fn reset(&mut self) {
        self.clusters.iter_mut().for_each(WTOscVoiceCluster::reset)
    }

    fn activate_voice(&mut self, cluster_idx: usize, voice_idx: usize, note: u8) {
        self.clusters[cluster_idx].activate_voice(voice_idx, &self.global_state, note);
    }

    fn deactivate_voice(&mut self, cluster_idx: usize, voice_idx: usize) {
        self.clusters[cluster_idx].deactivate_voice(voice_idx);
    }

    fn move_state(
        &mut self,
        (from_cluster, from_voice): (usize, usize),
        (to_cluster, to_voice): (usize, usize),
    ) {
        let clusters = Cell::from_mut(self.clusters.as_mut()).as_slice_of_cells();

        WTOscVoiceCluster::move_state(
            &clusters[from_cluster],
            from_voice,
            &clusters[to_cluster],
            to_voice,
        );
    }
}