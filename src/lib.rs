#![feature(portable_simd, new_uninit, const_float_bits_conv)]

mod basic_shapes;
pub mod wavetable;
extern crate alloc;

pub use alloc::sync::Arc;
use core::{cell::Cell, num::NonZeroUsize, mem};
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
    },
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

    pub fn set_table_reciever(
        &mut self,
        reciever: LenderReciever<BandLimitedWaveTables>,
    ) -> Option<LenderReciever<BandLimitedWaveTables>> {
        self.table_reciever.replace(reciever)
    }

    pub fn activate_voice(
        &mut self,
        cluster_idx: usize,
        voice_idx: usize,
        note: u8,
    ) -> Option<bool> {
        self.clusters.get_mut(cluster_idx).map(|cluster| {
            cluster.activate_voice(&self.params, cluster_idx, voice_idx, note, self.sr)
        })
    }

    pub fn deactivate_voice(&mut self, cluster_idx: usize, voice_idx: usize) -> Option<bool> {
        self.clusters
            .get_mut(cluster_idx)
            .map(|cluster| cluster.deactivate_voice(voice_idx))
    }

    pub fn activate_cluster(&mut self, index: usize) -> bool {
        if let Some(cluster) = self.clusters.get_mut(index) {
            cluster.activate(&self.params, index);
            return true;
        }

        false
    }

    pub fn deactivate_cluster(&mut self, index: usize) -> bool {
        if let Some(cluster) = self.clusters.get_mut(index) {
            cluster.deactivate();
            return true;
        }
        false
    }

    pub fn update_smoothers(&mut self, inc: Float) {
        let param_values = &mut self.params;

        param_values.tick_n(inc);

        self.clusters
            .iter_mut()
            .enumerate()
            .for_each(|(i, cluster)| cluster.set_params_smoothed(param_values, i, inc))
    }

    pub fn reset(&mut self) {
        self.clusters.iter_mut().for_each(WTOscVoiceCluster::reset);
    }

    pub fn load_wavetable_non_realtime(&mut self, path: impl AsRef<Path>) {
        self.table = BandLimitedWaveTables::from_file(path);
    }

    pub fn initialize(&mut self, sr: f32, max_buffer_size: usize) {
        self.params.initialize(sr, max_buffer_size);
    }

    pub fn update_param_smoothers(&mut self, num_samples: NonZeroUsize) {
        self.params.update_smoothers(num_samples);

        if let Some(reciever) = self.table_reciever.as_mut() {
            if let Some(table) = reciever.recv_latest() {
                self.replace_table(table);
            }
        }
    }

    pub fn process_buffer(
        &mut self,
        mut active_cluster_idxs: impl Iterator<Item = usize>,
        buffer: &[Cell<Float>],
    ) -> bool {
        let table = self.table.as_ref();

        if let Some(i) = active_cluster_idxs.next() {
            let cluster = unsafe { self.clusters.get_unchecked_mut(i) };

            for sample in buffer {
                sample.set(cluster.process(table));
            }
        } else {
            return false;
        }

        for i in active_cluster_idxs {
            let cluster = unsafe { self.clusters.get_unchecked_mut(i) };

            for sample in buffer {
                sample.set(sample.get() + cluster.process(table));
            }
        }

        true
    }

    pub fn params(&self) -> &T {
        &self.params
    }
}
