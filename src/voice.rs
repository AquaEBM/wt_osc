use super::*;
use cluster::WTOscClusterParams;
use oscillator::{Oscillator, OscillatorParams};

pub struct VoiceParams<'a> {
    global_state: &'a WTOscGlobalState,
    frame: Float,
    transpose: Float,
    random: Float,
    detune: Float,
    num_voices: UInt,
    phase_delta: Float,
}

impl<'a> VoiceParams<'a> {
    pub fn new(
        params: &WTOscClusterParams,
        global_state: &'a WTOscGlobalState,
        i: usize,
    ) -> Option<Self> {
        // SAFETY: i has just been bounds checked
        unsafe {
            (i < STEREO_VOICES_PER_VECTOR).then(|| Self::new_unchecked(params, global_state, i))
        }
    }

    pub unsafe fn new_unchecked(
        params: &WTOscClusterParams,
        global_state: &'a WTOscGlobalState,
        i: usize,
    ) -> Self {
        Self {
            global_state,
            frame: splat_stereo(*split_stereo(params.frame()).get_unchecked(i)),
            transpose: splat_stereo(*split_stereo(params.transpose()).get_unchecked(i)),
            random: splat_stereo(*split_stereo(params.random()).get_unchecked(i)),
            detune: splat_stereo(*split_stereo(params.detune()).get_unchecked(i)),
            num_voices: splat_stereo(*split_stereo(params.num_unison_voices()).get_unchecked(i)),
            phase_delta: splat_stereo(*split_stereo(params.base_phase_delta()).get_unchecked(i)),
        }
    }

    pub fn num_oscillators(&self) -> NonZeroUsize {
        unsafe {
            let n = split_stereo(self.num_unison_voices_raw()).get_unchecked(0);
            let [l, r] = (n / Simd::splat(FLOATS_PER_VECTOR as u32)).to_array();
            // SAFETY: l and r are guaranteed to be non-zero
            NonZeroUsize::new_unchecked(l.max(r) as usize)
        }
    }

    pub fn num_unison_voices_raw(&self) -> &UInt {
        &self.num_voices
    }

    pub fn detune(&self) -> &Float {
        &self.detune
    }
    pub fn transpose(&self) -> &Float {
        &self.transpose
    }
    pub fn frame(&self) -> &Float {
        &self.frame
    }
    pub fn random(&self) -> &Float {
        &self.random
    }
    pub fn starting_phases(&'a self) -> &'a [Float; NUM_VOICE_OSCILLATORS] {
        &self.global_state.starting_phases
    }
    pub fn base_phase_delta(&'a self) -> &'a Float {
        &self.phase_delta
    }
}

#[derive(Clone, Copy)]
pub struct WTOscVoice {
    num_oscs: NonZeroUsize,
    oscs: [Oscillator; NUM_VOICE_OSCILLATORS],
}

impl Default for WTOscVoice {
    fn default() -> Self {
        Self {
            oscs: Default::default(),
            num_oscs: NonZeroUsize::MIN,
        }
    }
}

impl WTOscVoice {
    #[allow(dead_code)]
    fn set_num_oscs(&mut self, num_oscs: NonZeroUsize) -> bool {
        let valid_index = num_oscs.get() <= NUM_VOICE_OSCILLATORS;
        if valid_index {
            self.num_oscs = num_oscs
        }
        valid_index
    }

    unsafe fn set_num_oscs_unchecked(&mut self, num_oscs: NonZeroUsize) {
        self.num_oscs = num_oscs;
    }

    fn active_oscs_mut(&mut self) -> &mut [Oscillator] {
        unsafe { self.oscs.get_unchecked_mut(..self.num_oscs.get()) }
    }

    pub fn deactivate(&mut self) {}

    pub fn activate(&mut self, params: VoiceParams) {
        self.set_params_instantly(params);
    }

    pub fn set_params_instantly(&mut self, params: VoiceParams) {
        unsafe { self.set_num_oscs_unchecked(params.num_oscillators()) };

        self.active_oscs_mut()
            .iter_mut()
            .enumerate()
            .for_each(|(i, osc)| osc.set_params_instantly(OscillatorParams::new(i, &params)));
    }

    pub fn set_params_smoothed(&mut self, params: VoiceParams, inc: Float) {
        unsafe { self.set_num_oscs_unchecked(params.num_oscillators()) };

        self.active_oscs_mut()
            .iter_mut()
            .enumerate()
            .for_each(|(i, osc)| osc.set_params_smoothed(OscillatorParams::new(i, &params), inc));
    }

    #[inline]
    pub fn process(&mut self, table: &BandLimitedWaveTables) -> f32x2 {
        let mut samples = self.oscs[0].advance_and_resample(table);

        self.oscs[1..]
            .iter_mut()
            .for_each(|osc| samples += osc.advance_and_resample(table));

        sum_to_stereo_sample(samples)
    }

    pub fn reset(&mut self) {
        self.oscs.iter_mut().for_each(Oscillator::reset)
    }
}
