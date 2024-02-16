use super::*;
use core::mem;
use voice::VoiceParams;

pub struct OscillatorParams<'a> {
    index: usize,
    params: &'a VoiceParams<'a>,
}

impl<'a> OscillatorParams<'a> {
    pub fn new(index: usize, params: &'a VoiceParams<'a>) -> Self {
        Self { index, params }
    }

    fn starting_phases(&self) -> UInt {
        let phases = unsafe {
            self.params
                .global_state
                .starting_phases
                .get_unchecked(self.index)
        };
        let adjusted_phases = phases * self.params.random;
        flp_to_fxp(adjusted_phases)
    }

    fn get_params(&self) -> (Float, Float, TMask) {
        let half_f = Float::splat(0.5);
        let one_u = UInt::splat(1);
        let last_voice_pair_idx =
            UInt::splat(((MAX_UNISON + (MAX_UNISON & 1) >> 1) - 1).max(1) as u32);
        let last_voice_pair_idx_f = last_voice_pair_idx.cast::<f32>();
        let max_float_bit_index = UInt::splat(mem::size_of::<f32>() as u32 * 8 - 1);
        let counting = UInt::from_array(array::from_fn(|i| i as u32));
        let counting_by2 = counting >> one_u;

        let v_osc_index = UInt::splat((self.index * FLOATS_PER_VECTOR) as u32);
        let voice_indices = v_osc_index + counting;
        let voice_pair_indices = v_osc_index + counting_by2;
        let pair_detunes = last_voice_pair_idx - voice_pair_indices;
        let sign_mask = (voice_indices ^ voice_pair_indices) << max_float_bit_index;

        let num_voices = self.params.num_voices;
        let half_num_voices_f = num_voices.cast() * half_f;
        let abs_norm_detunes = (half_num_voices_f - pair_detunes.cast()) / half_num_voices_f;
        let norm_detunes = Float::from_bits(abs_norm_detunes.to_bits() ^ sign_mask);

        let base_phase_delta = self.params.phase_delta * self.unison_stack_mult();
        let detune_semitones = self
            .params
            .detune
            .mul_add(norm_detunes, self.params.transpose);
        let detune_ratio = semitones_to_ratio(detune_semitones);
        let phase_delta = base_phase_delta * detune_ratio;

        let num_osc_voices = num_voices + (num_voices & one_u);
        let mask = num_osc_voices.simd_gt(voice_indices);

        let norm_voice_spread = voice_pair_indices.cast::<f32>() / last_voice_pair_idx_f;

        let norm_frame =
            norm_voice_spread.mul_add(self.frame_spread(), self.params.base_norm_frame);

        let norm_frame_clamped = norm_frame.simd_clamp(Simd::splat(0.00001), Simd::splat(0.99999));

        let frame = norm_frame_clamped * self.params.global_state.num_frames;

        (phase_delta, frame, mask)
    }

    fn unison_stack_mult(&self) -> Float {
        Float::splat(1.)
    }

    fn frame_spread(&self) -> Float {
        Float::splat(0.)
    }
}

#[derive(Default, Clone, Copy)]
pub struct Oscillator {
    phase_delta: LogSmoother,
    phase: UInt,
    frame: LinearSmoother,
    active_voices_mask: TMask,
}

impl Oscillator {
    #[inline]
    fn set_phase(&mut self, phase: UInt) {
        self.phase = phase;
    }

    #[inline]
    pub fn reset(&mut self) {
        self.set_phase(UInt::splat(0));
    }

    fn update_mask_and_phases(&mut self, starting_phases: UInt, active_voices_mask: TMask) {
        let newly_activated_voices = self.active_voices_mask ^ active_voices_mask;
        self.phase = newly_activated_voices.select(starting_phases, self.phase);
        self.active_voices_mask = active_voices_mask;
    }

    pub fn set_params_instantly(&mut self, params: OscillatorParams) {
        let (phase_delta, frame, mask) = params.get_params();
        self.update_mask_and_phases(params.starting_phases(), mask);
        self.phase_delta.set_val_instantly(phase_delta);
        self.frame.set_val_instantly(frame);
    }

    pub fn set_params_smoothed(&mut self, params: OscillatorParams, inc: Float) {
        let (phase_delta, frame, mask) = params.get_params();
        self.update_mask_and_phases(params.starting_phases(), mask);
        self.phase_delta.set_target(phase_delta, inc);
        self.frame.set_target(frame, inc);
    }

    #[inline]
    fn get_frame_index(&self) -> UInt {
        unsafe { self.frame.get_current().to_int_unchecked() }
    }

    #[inline]
    pub fn advance_and_resample(&mut self, table: &BandLimitedWaveTables) -> Float {
        self.phase_delta.tick1();
        let phase_delta = flp_to_fxp(*self.phase_delta.get_current());
        self.phase += phase_delta;
        let frame_idx = self.get_frame_index();
        table.resample_select(phase_delta, frame_idx, self.phase, self.active_voices_mask)
    }
}
