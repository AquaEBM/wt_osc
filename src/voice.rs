use super::*;
use core::{array, mem::transmute, num::NonZeroUsize};
use oscillator::Oscillator;

static UNISON_DETUNES: [[Float; NUM_VOICE_OSCILLATORS]; MAX_UNISON + 1] = {
    assert!(FLOATS_PER_VECTOR >= 2);

    let mut blocks = [[0.; NUM_VOICE_OSCILLATORS * FLOATS_PER_VECTOR]; MAX_UNISON + 1];

    /// sign_mask: 0. or -0.
    const fn const_copysign(x: f32, sign_mask: f32) -> f32 {
        f32::from_bits(x.to_bits() | sign_mask.to_bits())
    }

    let mut i = 2;
    while i < MAX_UNISON + 1 {
        let mut j = 0;
        let mut sign_mask = 0.;

        let step = 2. / (i - 1) as f32;
        let num_voices = i + i % 2; // next even number

        let remainder_voices = (num_voices - 1) % FLOATS_PER_VECTOR + 1;
        let empty_voices = FLOATS_PER_VECTOR - remainder_voices;

        while j < num_voices / 2 {
            let detune = const_copysign(1. - step * j as f32, sign_mask);

            let offset = if j + remainder_voices / 2 < num_voices / 2 {
                empty_voices
            } else {
                0
            };

            blocks[i][num_voices - j * 2 - 1 + offset] = detune;
            blocks[i][num_voices - j * 2 - 2 + offset] = -detune;

            j += 1;
            sign_mask = -sign_mask;
        }

        i += 1;
    }

    // SAFETY: we're transmuting f32s to Simd<f32, N>s so values are valid
    unsafe { transmute(blocks) }
};

pub struct WTOscVoice {
    oscs: [Oscillator; NUM_VOICE_OSCILLATORS],
    num_oscs: NonZeroUsize,
    remainder_mask: TMask,
}

impl Default for WTOscVoice {
    fn default() -> Self {
        Self {
            oscs: Default::default(),
            num_oscs: NonZeroUsize::new(1).unwrap(),
            remainder_mask: Default::default(),
        }
    }
}

impl WTOscVoice {
    fn get_voice_params<T: WTOscParams>(
        param_values: &T,
        cluster_idx: usize,
        voice_idx: usize,
    ) -> (usize, Float, Float, Float) {
        (
            param_values.get_num_unison_voices(cluster_idx)[voice_idx],
            splat_slot(&param_values.get_norm_frame(cluster_idx), voice_idx).unwrap(),
            splat_slot(&param_values.get_transpose(cluster_idx), voice_idx).unwrap(),
            splat_slot(&param_values.get_detune(cluster_idx), voice_idx).unwrap(),
        )
    }

    fn initialize(
        &mut self,
        base_phase_delta: Float,
        randomisation: Float,
        phases: &[Float; NUM_VOICE_OSCILLATORS],
    ) {
        self.oscs
            .iter_mut()
            .zip(phases.iter())
            .for_each(|(osc, &phase)| {
                osc.set_phase(flp_to_fxp(phase * randomisation));
                osc.base_phase_delta = base_phase_delta;
            });
    }

    pub fn deactivate(&mut self) {}

    pub fn activate<T: WTOscParams>(
        &mut self,
        param_values: &T,
        cluster_idx: usize,
        voice_idx: usize,
        note: u8,
        sr: f32,
    ) {
        let randomisation = splat_slot(&param_values.get_random(cluster_idx), voice_idx).unwrap();
        let base_phase_delta = Float::splat(440. * f32::exp2((note as i8 - 69) as f32 / 12.) / sr);
        let phases = param_values.get_starting_phases();

        self.initialize(base_phase_delta, randomisation, phases);

        self.set_params_instantly(param_values, cluster_idx, voice_idx);
    }

    pub fn set_params_instantly<T: WTOscParams>(
        &mut self,
        param_values: &T,
        cluster_idx: usize,
        voice_idx: usize,
    ) {
        let (num_unison_voices, frame, transpose, detune) =
            Self::get_voice_params(param_values, cluster_idx, voice_idx);

        let norm_detunes =
            self.set_num_unison_voices(NonZeroUsize::new(num_unison_voices).unwrap());

        self.oscs
            .iter_mut()
            .zip(norm_detunes.iter())
            .for_each(|(osc, norm_detune)| {
                osc.set_frame(frame);
                osc.set_detune_semitones(norm_detune.mul_add(detune, transpose));
            });
    }

    pub fn set_params_smoothed<T: WTOscParams>(
        &mut self,
        param_values: &T,
        cluster_idx: usize,
        voice_idx: usize,
        inc: Float,
    ) {
        let (num_unison_voices, frame, transpose, detune) =
            Self::get_voice_params(param_values, cluster_idx, voice_idx);

        let norm_detunes =
            self.set_num_unison_voices(NonZeroUsize::new(num_unison_voices).unwrap());

        self.oscs
            .iter_mut()
            .zip(norm_detunes.iter())
            .for_each(|(osc, norm_detune)| {
                osc.set_frame_smoothed(frame, inc);
                osc.set_detune_semitones_smoothed(
                    norm_detune.mul_add(detune, transpose),
                    inc,
                );
            });
    }

    #[inline]
    pub fn process(&mut self, table: &BandLimitedWaveTables) -> f32x2 {
        let mut samples = self.oscs[0].advance_and_resample_select(table, self.remainder_mask);

        self.oscs[1..]
            .iter_mut()
            .for_each(|osc| samples += osc.advance_and_resample(table));

        sum_to_stereo_sample(samples)
    }

    fn set_num_unison_voices(&mut self, num: NonZeroUsize) -> &'static [Float] {
        let num = num.get();
        let n = num + (num & 1);

        let num_vectors = enclosing_div(n, FLOATS_PER_VECTOR);

        self.num_oscs = NonZeroUsize::new(num_vectors).unwrap();

        let rem = (n - 1) % FLOATS_PER_VECTOR + 1;
        self.remainder_mask = TMask::from_array(array::from_fn(|i| i < rem));

        unsafe {
            UNISON_DETUNES
                .get_unchecked(num)
                .get_unchecked(..num_vectors)
        }
    }

    pub fn set_frame_instantly(&mut self, norm_frame: Float, num_frames: Float) {
        let frame = norm_frame * num_frames;
        for osc in self.oscs.iter_mut() {
            osc.set_frame(frame);
        }
    }

    pub fn reset(&mut self) {
        self.oscs.iter_mut().for_each(Oscillator::reset_phase)
    }
}
