# wt_osc

Basic wavetable oscillator plugin written in Rust

Note that this repository is no longer maintained, and the plugin itself is no longer installable, as this has been moved to a bigger
project ([Mythril](https://github.com/AquaEBM/Mythril)) with the intention of being usied as an individual node as part of the modular synth.
The demos hereafter are old recordings I have found from the time when this was it's own plugin.

Bandlimited resampling techniques used are described in [this talk by Matt Tytel](https://www.youtube.com/watch?v=qlinVx60778)

## Features

- All processing is SIMD-optimised, several voices are processed at once, depending on the architecture of the compiled-for target (up to 8 stereo voices at a time on CPUs supporting AVX-512)
- Real-time wavetable switching, wavetables can be loaded from WAV files with exactly 2048 * 256 samples
- Basic stereo unison, with up to 16 voices, all SIMD-optimised
- Real-time automation of transpose, volume level, wavetable position, unison detune, and number of unison voices

## Demos (old)

No effects, just two instances.

https://github.com/AquaEBM/wt_osc/assets/79016373/867d9887-1b7f-429e-b0cb-40ee3cc7f5f5

4 instances with different wavetables and transpose values + LFOTool (volume envelope because we don't have one) + Chorus + Delay + Reverb

https://github.com/AquaEBM/wt_osc/assets/79016373/d7420983-d805-4d0f-bb44-3c80d166ca58
