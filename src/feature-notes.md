EEG Feature extraction sources:

https://bits-pilani-neurotech-lab.github.io/posts/bandpower/


#### I have never tried to analyse EEG signals before. These features seem to be regarded as good indications of different brain activities. But they are a shot in the dark.....

##### Some notes
- EEG signals are typically decomposed into different frequency bands, each associated with distinct neural processes. 
- PSD describes how the power of a time-varying signal is distributed across different frequencies. Welch's method estimates the PSD
- We can get the power of a certain frequency by integrating the PSD in that frequency band 
- Want to normalise this because the amplitude of different freqs varies a lot?
- Anyways, the root-mean-square value already captures overall amplitude


