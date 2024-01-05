#![feature(portable_simd, new_uninit, const_float_bits_conv)]

mod basic_shapes;
pub mod wavetable;
extern crate alloc;

pub use alloc::sync::Arc;
use core::{num::NonZeroUsize, mem, iter};
pub use plugin_util::{
    math::*,
    simd::{
        f32x2,
        prelude::{SimdFloat, SimdOrd},
        Simd, StdFloat,
    },
    simd_util::*,
    smoothing::*,
};
use polygraph::{
    lender::LenderReciever,
    stereo_util::{
        as_mut_stereo_sample_array, splat_slot, swap_stereo, triangular_pan_weights,
        STEREO_VOICES_PER_VECTOR,
    }, processor::Processor,
};
pub use std::path::Path;
use wavetable::BandLimitedWaveTables;

mod cluster;
use cluster::WTOscVoiceCluster;

mod oscillator;
mod voice;

pub const MAX_UNISON: usize = 16;
pub const NUM_VOICE_OSCILLATORS: usize = enclosing_div(MAX_UNISON, FLOATS_PER_VECTOR);

pub trait WTOscParams {
    fn initialize(&mut self, sr: f32, max_buffer_size: usize);

    fn update_smoothers(&mut self, num_samples: NonZeroUsize);

    fn tick_n(&mut self, inc: Float);

    fn get_detune(&self, cluster_idx: usize) -> Float;

    fn get_transpose(&self, cluster_idx: usize) -> Float;

    fn get_norm_frame(&self, cluster_idx: usize) -> Float;

    fn get_random(&self, cluster_idx: usize) -> Float;

    fn get_level(&self, cluster_idx: usize) -> Float;

    fn get_stereo_amount(&self, cluster_idx: usize) -> Float;

    fn get_norm_pan(&self, cluster_idx: usize) -> Float;

    fn get_num_unison_voices(&self, cluster_idx: usize) -> [usize; STEREO_VOICES_PER_VECTOR];

    fn get_starting_phases(&self) -> &[Float; NUM_VOICE_OSCILLATORS];
}

pub struct WTOsc<T> {
    params: T,
    table_reciever: Option<LenderReciever<BandLimitedWaveTables>>,
    sr: f32,
    table: Arc<BandLimitedWaveTables>,
    clusters: Box<[WTOscVoiceCluster]>,
    fnum_frames: Float,
}

impl<T: Default> Default for WTOsc<T> {
    fn default() -> Self {
        Self::with_params(T::default())
    }
}

impl<T> WTOsc<T> {
    pub fn with_params(params: T) -> Self {
        Self {
            params,
            table_reciever: None,
            sr: 44100.,
            table: BandLimitedWaveTables::with_frame_count(0),
            fnum_frames: Simd::splat(0.),
            clusters: Box::new([]),
        }
    }

    pub fn set_table_reciever(
        &mut self,
        reciever: LenderReciever<BandLimitedWaveTables>,
    ) -> Option<LenderReciever<BandLimitedWaveTables>> {
        self.table_reciever.replace(reciever)
    }
}

impl<T: WTOscParams> WTOsc<T> {

    pub fn replace_table(
        &mut self,
        new_table: Arc<BandLimitedWaveTables>,
    ) -> Arc<BandLimitedWaveTables> {
        self.fnum_frames = Simd::splat(new_table.num_frames() as f32);

        for (i, cluster) in self.clusters.iter_mut().enumerate() {
            let norm_frame = self.params.get_norm_frame(i);
            for (_j, voice) in cluster.voices.iter_mut().enumerate() {
                voice.set_frame_instantly(norm_frame, self.fnum_frames)
            }
        }

        mem::replace(&mut self.table, new_table)
    }

    pub fn update_param_smoothers(&mut self, num_samples: NonZeroUsize) {
        self.params.update_smoothers(num_samples);

        if let Some(reciever) = self.table_reciever.as_mut() {
            if let Some(table) = reciever.recv_latest() {
                self.replace_table(table);
            }
        }
    }

    pub fn params(&self) -> &T {
        &self.params
    }   
}

impl<T: WTOscParams> Processor<FLOATS_PER_VECTOR> for WTOsc<T> {

    fn process(
        &mut self,
        buffers: polygraph::Buffers<Simd<f32, FLOATS_PER_VECTOR>>,
        cluster_idx: usize,
        params_changed: Option<NonZeroUsize>,
    ) {
        if let Some(num_samples) = params_changed {
            self.update_param_smoothers(num_samples);
        }

        if let Some(output_buf) = buffers.get_output(0) {

            let table = self.table.as_ref();
            let cluster = &mut self.clusters[cluster_idx];

            let inc = Simd::splat(1. / output_buf.len() as f32);

            cluster.set_params_smoothed(&self.params, cluster_idx, inc);

            for sample in output_buf.iter() {
                sample.set(cluster.process(table));
            }
        }
    }

    fn initialize(&mut self, sr: f32, max_buffer_size: usize) {
        self.params.initialize(sr, max_buffer_size);
    }

    fn reset(&mut self) {
        self.clusters.iter_mut().for_each(WTOscVoiceCluster::reset)
    }

    fn set_max_polyphony(&mut self, num_clusters: usize) {
        self.clusters = iter::repeat_with(Default::default).take(num_clusters).collect()
    }

    fn activate_cluster(&mut self, index: usize) {
        self.clusters[index].activate(&self.params, index);
    }

    fn deactivate_cluster(&mut self, index: usize) {
        self.clusters[index].deactivate()
    }

    fn activate_voice(&mut self, cluster_idx: usize, voice_idx: usize, note: u8) {
        self.clusters[cluster_idx].activate_voice(&self.params, cluster_idx, voice_idx, note, self.sr);
    }

    fn deactivate_voice(&mut self, cluster_idx: usize, voice_idx: usize) {
        self.clusters[cluster_idx].deactivate_voice(voice_idx);
    }
}