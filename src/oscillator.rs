use super::*;

#[derive(Default, Clone, Copy)]
pub struct Oscillator {
    /// phase delta before unison detuning, transposition
    pub base_phase_delta: Float,
    phase_delta: LogSmoother,
    phase: UInt,
    frame: LinearSmoother,
}

impl Oscillator {
    #[inline]
    pub fn advance_phase(&mut self) -> UInt {
        let phase_delta_fixed_point = flp_to_fxp(self.phase_delta.get_current());

        self.set_phase(self.phase + phase_delta_fixed_point);

        phase_delta_fixed_point
    }

    #[inline]
    pub fn set_phase(&mut self, phase: UInt) {
        self.phase = phase;
    }

    #[inline]
    unsafe fn get_frame_index(&self) -> UInt {
        self.frame.get_current().to_int_unchecked()
    }

    #[inline]
    pub fn update_phase_delta_smoother(&mut self) {
        self.phase_delta.tick()
    }

    #[inline]
    pub fn reset_phase(&mut self) {
        self.phase = Simd::splat(0);
    }

    pub fn set_detune_semitones_smoothed(&mut self, semitones: Float, inc: Float) {
        let detune_ratio = semitones_to_ratio(semitones);
        self.phase_delta
            .set_increment(self.base_phase_delta * detune_ratio, inc);
    }

    pub fn set_detune_semitones(&mut self, semitones: Float) {
        self.phase_delta
            .set_instantly(self.base_phase_delta * semitones_to_ratio(semitones));
    }

    pub fn set_frame_smoothed(&mut self, frame: Float, inc: Float) {
        self.frame.set_increment(frame, inc);
    }

    pub fn set_frame(&mut self, frame: Float) {
        self.frame.set_instantly(frame);
    }

    #[inline]
    pub fn advance_and_resample_select(
        &mut self,
        table: &BandLimitedWaveTables,
        mask: TMask,
    ) -> Float {
        self.update_phase_delta_smoother();
        let phase_delta = self.advance_phase();
        let frame_idx = unsafe { self.get_frame_index() };
        table.resample_select(phase_delta, frame_idx, self.phase, mask)
    }

    #[inline]
    pub fn advance_and_resample(&mut self, table: &BandLimitedWaveTables) -> Float {
        self.update_phase_delta_smoother();
        let phase_delta = self.advance_phase();
        let frame_idx = unsafe { self.get_frame_index() };
        table.resample(phase_delta, frame_idx, self.phase)
    }
}
