use super::*;
use cell_project::cell_project as p;
use voice::{VoiceParams, WTOscVoice};

fn set_sample<T: SimdElement>(
    input: &Cell<Simd<T, FLOATS_PER_VECTOR>>,
    from: usize,
    output: &Cell<Simd<T, FLOATS_PER_VECTOR>>,
    to: usize,
) {
    let out_samples = split_stereo_cell(output).as_array_of_cells();
    let in_samples = split_stereo_cell(input).as_array_of_cells();
    out_samples[to].set(in_samples[from].get());
}

#[derive(Default)]
pub struct WTOscClusterParams {
    detune: LinearSmoother,
    detune_range: LinearSmoother,
    transpose: LinearSmoother,
    norm_frame: LinearSmoother,
    random: LinearSmoother,
    level: LinearSmoother,
    stereo: LinearSmoother,
    norm_pan: LinearSmoother,
    pub num_voices: UInt,
    pub phase_delta: Float,
}

impl WTOscClusterParams {
    const DETUNE_RANGE: f32 = 96.;

    fn tick_n(&mut self, n: NonZeroUsize) {
        let incs = Simd::splat(n.get() as f32);
        self.detune.tick(incs);
        self.detune_range.tick(incs);
        self.transpose.tick(incs);
        self.random.tick(incs);
        self.level.tick(incs);
        self.stereo.tick(incs);
        self.norm_pan.tick(incs);
    }

    fn move_state(this: &Cell<Self>, from: usize, other: &Cell<Self>, to: usize) {
        for (input, output) in [
            (p!(Self, this.detune), p!(Self, other.detune)),
            (p!(Self, this.detune_range), p!(Self, other.detune_range)),
            (p!(Self, this.transpose), p!(Self, other.transpose)),
            (p!(Self, this.norm_frame), p!(Self, other.norm_frame)),
            (p!(Self, this.random), p!(Self, other.random)),
            (p!(Self, this.level), p!(Self, other.level)),
            (p!(Self, this.stereo), p!(Self, other.stereo)),
            (p!(Self, this.norm_pan), p!(Self, other.norm_pan)),
        ] {
            set_sample(
                p!(LinearSmoother, input.value),
                from,
                p!(LinearSmoother, output.value),
                to,
            );
            set_sample(
                p!(LinearSmoother, input.increment),
                from,
                p!(LinearSmoother, output.increment),
                to,
            );
        }

        set_sample(
            p!(Self, this.phase_delta),
            from,
            p!(Self, other.phase_delta),
            to,
        );
        set_sample(
            p!(Self, this.num_voices),
            from,
            p!(Self, other.num_voices),
            to,
        );
    }

    pub fn set_param_smoothed(&mut self, param_id: u64, norm_val: Float, smooth_time: Float) {

        match param_id {
            1 => self.detune.set_target(norm_val, smooth_time),
            2 => self.detune_range.set_target((norm_val - Simd::splat(0.5)) * Simd::splat(Self::DETUNE_RANGE), smooth_time),
            3 => self.transpose.set_target(norm_val, smooth_time),
            4 => self.norm_frame.set_target(norm_val, smooth_time),
            5 => self.random.set_target(norm_val, smooth_time),
            6 => self.level.set_target(norm_val, smooth_time),
            7 => self.stereo.set_target(norm_val, smooth_time),
            8 => self.norm_pan.set_target(norm_val, smooth_time),
            9 => self.num_voices = norm_val.mul_add(Simd::splat(15.98), Simd::splat(1.)).cast(),
            _ => (),
        }
    }

    pub fn set_param(&mut self, param_id: u64, norm_val: Float) {
        

        let half = Simd::splat(0.5);

        match param_id {
            1 => self.detune.set_val_instantly(norm_val),
            2 => self.detune_range.set_val_instantly((norm_val - half) * Simd::splat(Self::DETUNE_RANGE)),
            3 => self.transpose.set_val_instantly(norm_val),
            4 => self.norm_frame.set_val_instantly(norm_val),
            5 => self.random.set_val_instantly(norm_val),
            6 => self.level.set_val_instantly(norm_val),
            7 => self.stereo.set_val_instantly(norm_val),
            8 => self.norm_pan.set_val_instantly(norm_val),
            9 => self.num_voices = norm_val.mul_add(Simd::splat(15.98), Simd::splat(1.)).cast(),
            _ => (),
        }
    }

    pub fn detune(&self) -> &Float {
        self.detune.get_current()
    }
    pub fn transpose(&self) -> &Float {
        self.transpose.get_current()
    }
    pub fn norm_frame(&self) -> &Float {
        self.norm_frame.get_current()
    }
    pub fn random(&self) -> &Float {
        self.random.get_current()
    }

    fn get_sample_weights(&self) -> (Float, Float) {
        let level = *self.level.get_current();

        let stereo = *self.stereo.get_current();
        let pan = *self.norm_pan.get_current();

        let unison_normalisation = self.num_voices.cast().recip();
        let pan_weights = triangular_pan_weights(pan) * unison_normalisation;

        (
            pan_weights.mul_add(stereo, pan_weights).sqrt() * level,
            pan_weights.mul_add(-stereo, pan_weights).sqrt() * level,
        )
    }
}

#[derive(Default)]
pub struct WTOscVoiceCluster {
    pub params: WTOscClusterParams,
    voices: [Option<WTOscVoice>; STEREO_VOICES_PER_VECTOR],
    normal_weights: LinearSmoother,
    flipped_weights: LinearSmoother,
}

impl WTOscVoiceCluster {
    pub fn move_state(this: &Cell<Self>, from: usize, other: &Cell<Self>, to: usize) {
        assert!(from < STEREO_VOICES_PER_VECTOR);
        assert!(to < STEREO_VOICES_PER_VECTOR);

        WTOscClusterParams::move_state(p!(Self, this.params), from, p!(Self, other.params), to);

        let sn = p!(Self, this.normal_weights);
        let sf = p!(Self, this.flipped_weights);
        let on = p!(Self, other.normal_weights);
        let of = p!(Self, other.flipped_weights);

        type L = LinearSmoother;

        set_sample(p!(L, sn.value), from, p!(L, on.value), to);
        set_sample(p!(L, sn.increment), from, p!(L, on.increment), to);
        set_sample(p!(L, sf.value), from, p!(L, of.value), to);
        set_sample(p!(L, sf.increment), from, p!(L, of.increment), to);

        let this_voices = p!(Self, this.voices).as_array_of_cells();
        let other_voices = p!(Self, other.voices).as_array_of_cells();

        other_voices[to].set(this_voices[from].get());
    }

    pub fn set_gains_instantly(&mut self) {
        let params = &self.params;

        let (normal_weights, flipped_weights) = params.get_sample_weights();

        self.normal_weights.set_val_instantly(normal_weights);
        self.flipped_weights.set_val_instantly(flipped_weights);
    }

    pub fn set_params_smoothed(
        &mut self,
        global_state: &WTOscGlobalState,
        num_samples: NonZeroUsize,
    ) {
        let inc = Simd::splat(1. / num_samples.get() as f32);

        self.params.tick_n(num_samples);

        let params = &self.params;

        let (normal_weights, flipped_weights) = params.get_sample_weights();

        self.normal_weights.set_target(normal_weights, inc);
        self.flipped_weights.set_target(flipped_weights, inc);

        self.voices
            .iter_mut()
            .enumerate()
            .filter_map(|(i, maybe_voice)| maybe_voice.as_mut().map(|voice| (i, voice)))
            .for_each(|(i, voice)| {
                voice.set_params_smoothed(VoiceParams::new(params, global_state, i).unwrap(), inc)
            });
    }

    #[inline]
    pub fn process(&mut self, table: &BandLimitedWaveTables) -> Float {
        let mut output = Simd::splat(0.);

        split_stereo_mut(&mut output)
            .iter_mut()
            .zip(&mut self.voices)
            .filter_map(|(sample, maybe_voice)| maybe_voice.as_mut().map(|voice| (sample, voice)))
            .for_each(|(sample, voice)| *sample = voice.process(table));

        let flipped = swap_stereo(output);

        self.normal_weights.tick1();
        self.flipped_weights.tick1();

        output = self.normal_weights.get_current() * output
            + self.flipped_weights.get_current() * flipped;

        output
    }

    pub fn reset(&mut self) {
        self.voices
            .iter_mut()
            .filter_map(Option::as_mut)
            .for_each(WTOscVoice::reset)
    }

    pub fn activate_voice(
        &mut self,
        voice_idx: usize,
        global_state: &WTOscGlobalState,
        note: u8,
    ) -> bool {
        let phase_deltas = split_stereo_mut(&mut self.params.phase_delta);

        phase_deltas[voice_idx] =
            Simd::splat(440. * f32::exp2((note as i8 - 69) as f32 / 12.) / global_state.sr);

        if self.voices.iter().all(Option::is_none) {
            self.activate();
        }

        if let Some(voice) = self.voices.get_mut(voice_idx) {
            let cluster_params = VoiceParams::new(&self.params, global_state, voice_idx).unwrap();
            voice.insert(Default::default()).activate(cluster_params);
            return true;
        }
        false
    }

    pub fn deactivate_voice(&mut self, index: usize) -> bool {
        if let Some(maybe_voice) = self.voices.get_mut(index) {
            return if let Some(voice) = maybe_voice {
                voice.deactivate();
                *maybe_voice = None;

                if self.voices.iter().all(Option::is_none) {
                    self.deactivate();
                }
                true
            } else {
                false
            };
        }

        false
    }

    pub fn activate(&mut self) {
        self.set_gains_instantly();
    }

    pub fn deactivate(&mut self) {}
}
