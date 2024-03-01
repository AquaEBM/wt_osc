#![feature(portable_simd, new_uninit, as_array_of_cells)]

extern crate alloc;

mod cluster;
mod oscillator;
mod voice;

mod basic_shapes;
pub mod wavetable;

use alloc::sync::Arc;
use cluster::{WTOscClusterNormParams, WTOscVoiceCluster};
use core::{any::Any, array, cell::Cell, iter, mem, num::NonZeroUsize};
use polygraph::{
    buffer::Buffers,
    processor::{Params, Processor},
    simd_util::{
        math::*,
        simd::{prelude::*, Simd, StdFloat},
        smoothing::*,
        *,
    },
};
use voice::VoiceParams;
use wavetable::BandLimitedWaveTables;

const MAX_UNISON: usize = 16;
const OSCS_PER_VOICE: usize = enclosing_div(MAX_UNISON, FLOATS_PER_VECTOR);
const NUM_PARAMS: u64 = 9;
const MAX_PARAM_INDEX: u64 = NUM_PARAMS - 1;

#[derive(Default)]
pub struct WTOsc {
    table: Box<BandLimitedWaveTables>,
    starting_phases: [Float; OSCS_PER_VOICE],
    sr: f32,
    log2_alpha: f32,
    scratch_buffer: Box<[Float]>,
    clusters: Box<[WTOscVoiceCluster]>,
    params: Box<[WTOscClusterNormParams]>,
}

impl Processor for WTOsc {
    type Sample = Float;

    fn audio_io_layout(&self) -> (usize, usize) {
        (0, 1)
    }

    fn process(&mut self, buffers: Buffers<Float>, cluster_idx: usize, voice_mask: &TMask) {
        let table = self.table.as_ref();

        if let Some((output_buf, num_frames)) = buffers
            .get_output(0)
            .zip(NonZeroUsize::new(table.num_frames()))
        {
            let buffer_size = buffers.buffer_size().get();
            let smooth_dt = Float::splat(1.0 / buffer_size as f32);

            let cluster = &mut self.clusters[cluster_idx];
            let cluster_params = &mut self.params[cluster_idx];

            cluster_params.tick_n(self.log2_alpha, buffer_size);

            let num_frames_f = Float::splat(num_frames.get() as f32);

            let scratch_buffer = &mut self.scratch_buffer[..buffer_size];

            for (voice_index, voice) in cluster.voices_mut()
                .iter_mut()
                .enumerate()
                .zip(voice_mask.to_array().into_iter().step_by(2))
                .filter_map(|(data, active)| active.then_some(data))
            {
                let (voice_params, num_oscs) =
                    VoiceParams::new(voice_index, cluster_params).unwrap();

                let (first_osc, other_oscs) = unsafe { voice.get_unchecked_mut(..num_oscs.get()) }
                    .split_first_mut()
                    .unwrap();

                let mask = first_osc.set_params_smoothed(&voice_params, 0, num_frames_f, smooth_dt);
                let voice_samples = split_stereo_cell_slice(output_buf)
                    .iter()
                    .skip(voice_index)
                    .step_by(STEREO_VOICES_PER_VECTOR);

                if OSCS_PER_VOICE > 1 {
                    for sample in scratch_buffer.iter_mut() {
                        *sample = unsafe { first_osc.tick_all(table, mask) };
                    }

                    for (osc, osc_index) in other_oscs.iter_mut().zip(1..) {
                        let mask = osc.set_params_smoothed(
                            &voice_params,
                            osc_index,
                            num_frames_f,
                            smooth_dt,
                        );

                        for sample in scratch_buffer.iter_mut() {
                            *sample += unsafe { osc.tick_all(table, mask) };
                        }
                    }

                    for (out_sample, &scratch) in voice_samples.zip(scratch_buffer.iter()) {
                        out_sample.set(sum_to_stereo_sample(scratch));
                    }
                } else {
                    for out_sample in voice_samples {
                        let output = unsafe { first_osc.tick_all(table, mask) };
                        out_sample.set(sum_to_stereo_sample(output));
                    }
                }
            }

            cluster.set_weights_smoothed(cluster_params, smooth_dt);

            for poly_sample in output_buf {
                let (normal, flipped) = cluster.get_sample_weights();
                cluster.tick_weight_smoothers();
                let sample = poly_sample.get();
                let out = sample * normal + swap_stereo(sample) * flipped;
                poly_sample.set(out);
            }
        }
    }

    fn initialize(&mut self, sr: f32, max_buffer_size: usize, max_num_clusters: usize) {
        self.sr = sr;

        // reach the target value (0.999%) in approximately 20ms
        const BASE_LOG2_ALPHA: f32 = -500.0;

        self.log2_alpha = BASE_LOG2_ALPHA / sr;

        self.clusters = iter::repeat_with(Default::default)
            .take(max_num_clusters)
            .collect();

        self.params = iter::repeat_with(Default::default)
            .take(max_num_clusters)
            .collect();

        self.scratch_buffer = unsafe {
            Box::new_uninit_slice(if OSCS_PER_VOICE > 1 {
                max_buffer_size
            } else {
                0
            })
            .assume_init()
        };
    }

    fn set_param(
        &mut self,
        cluster_idx: usize,
        voice_mask: &TMask,
        param_id: u64,
        norm_val: Float,
    ) {
        self.params[cluster_idx].set_param_target(param_id, norm_val, voice_mask);
    }

    fn custom_event(&mut self, event: &mut dyn Any) {
        if let Some(wt) = event.downcast_mut::<Box<BandLimitedWaveTables>>() {
            let ratio = Simd::splat(wt.num_frames() as f32 / self.table.num_frames() as f32);
            for cluster in self.clusters.iter_mut() {
                cluster.scale_frames(ratio);
            }

            mem::swap(wt, &mut self.table);
        }

        if let Some(starting_phases) = event.downcast_mut::<[f32; MAX_UNISON]>() {
            self.starting_phases
                .iter_mut()
                .flat_map(Simd::as_mut_array)
                .zip(starting_phases.iter())
                .for_each(|(i, &o)| *i = o);
        }
    }

    fn reset(&mut self, cluster_idx: usize, voice_mask: &TMask) {
        let random = &self.params[cluster_idx].random.current;
        self.clusters[cluster_idx].reset_phases(voice_mask, random, &self.starting_phases);
    }

    fn move_state(
        &mut self,
        (from_cluster, from_voice): (usize, usize),
        (to_cluster, to_voice): (usize, usize),
    ) {
        (from_voice < STEREO_VOICES_PER_VECTOR && to_voice < STEREO_VOICES_PER_VECTOR)
            .then(|| {
                let clusters = Cell::from_mut(self.clusters.as_mut()).as_slice_of_cells();
                let params = Cell::from_mut(self.params.as_mut()).as_slice_of_cells();

                unsafe {
                    WTOscVoiceCluster::move_state_unchecked(
                        &clusters[from_cluster],
                        from_voice,
                        &clusters[to_cluster],
                        to_voice,
                    );

                    WTOscClusterNormParams::move_state_unchecked(
                        &params[from_cluster],
                        from_voice,
                        &params[to_cluster],
                        to_voice,
                    );
                }
            })
            .expect("out of bounds voice indices")
    }

    fn set_voice_note(&mut self, cluster_idx: usize, voice_mask: &TMask, note: Float) {
        let c4_phase_delta = Simd::splat(440. / self.sr);
        let new_phase_delta = c4_phase_delta * semitones_to_ratio(note - Simd::splat(69.0));

        let params = &mut self.params[cluster_idx];

        let ratio = voice_mask.select(new_phase_delta / params.phase_delta, Simd::splat(1.0));

        params.set_base_phase_delta(new_phase_delta, voice_mask);

        self.clusters[cluster_idx].scale_phase_deltas(ratio);
    }

    fn set_all_params(
        &mut self,
        cluster_idx: usize,
        voice_mask: &<Self::Sample as SimdFloat>::Mask,
        params: Params,
    ) {
        let cluster_params = &mut self.params[cluster_idx];

        for param_id in 0..NUM_PARAMS {
            cluster_params.set_param_instantly(param_id, params.get_param(param_id), voice_mask);
        }

        let num_frames_f = Simd::splat(self.table.num_frames() as f32);

        self.clusters[cluster_idx].set_params(cluster_params, num_frames_f, voice_mask);
    }
}

#[cfg(test)]
mod tests {

    pub fn test() {

    }
}