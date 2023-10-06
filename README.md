# SiTraNo (SInes+TRAnsients+NOise)
A MATLAB app for sines-transients-noise decomposition of audio signals. Developed using App Designer in Matlab 2020b.

**SiTraNo<sup>+</sup> - A Real-Time JUCE-based version of SiTraNo is available** [here](https://github.com/tantepSjnk/SiTraNoPlus)!

* L. Fierro, and V. Välimäki. _"**SiTraNo: a MATLAB app for sines-transient-noise decomposition of audio signals**"_.  Proceedings of the 24th International Conference on Digital Audio Effects (DAFx20in21), Vienna, Austria.

![](GUIFinal.png)

## Abstract

Decomposition of sounds into their sinusoidal, transient, and noise components is an active research topic and a widely-used tool in audio processing. Multiple solutions have been proposed in recent years, using time-frequency representations to identify either horizontal and vertical structures or orientations and anisotropy in the spectrogram of the sound. 
This is SiTraNo: an easy-to-use MATLAB application with a graphic user interface for audio decomposition that enables visualization and access to the sinusoidal, transient, and noise classes, individually. This application allows the user to choose between different well-known separation methods to analyze an input sound file, to instantaneously control and remix its spectral components, and to visually check the quality of the decomposition, before producing the desired output file. The visualization of common artifacts, such as birdies and dropouts, is easy to get in SiTraNo. 

This app wants to promote experimenting with the sound decomposition process by observing the effect of variations for each spectral component on the original sound and by comparing different methods against each other, evaluating the separation quality both audibly and visually.

## Dependencies

* [Audio Toolbox](https://www.mathworks.com/products/audio.html)
* [Image Processing Toolbox](https://www.mathworks.com/products/image.html)

## Installation and use
* If your version of MATLAB is 2020b or later, download the latest [release](https://github.com/himynameisfuego/SiTraNo/releases/latest). If your version is 2020a or previous, refer to this [hotfix](https://github.com/himynameisfuego/SiTraNo/files/6351972/SiTraNo_HotFix_1.0.0.1.zip) until the next release.
* In MATLAB, navigate to the SiTraNo folder, open **SiTraNo.mlappinstall** and install. You will find SiTraNo in the "Apps" tab, in the "My apps" group. Click on it to execute the app.
* Upon launching SiTraNo, a navigation folder should pop up, asking you to choose the input audio file.

## Featured decomposition methods

* **STN**: Sines-Transient-Noise decomposition. Default option. [1]
* **HP**: Harmonic-Percussive separation [2]. Modes: hard mask, soft mask.
* **HPR**: Harmonic-Percussive-Residual separation [3]. Modes: single decomposition, two-round decomposition. 
* **ST**: Structure-Tensor-based separation [4].
* **Fuzzy**: Fuzzy logic decomposition [5].

## Future updates
* Add Standalone installer featuring MATLAB Runtime.
* Adding new separation methods.
* Polish spectrogram presentation.
* Improve code efficiency.
* JUCE version of the app --> soon available on [Tantep Sinjanakhom](https://github.com/60010454)'s GitHub!

Expected work period: Summer 2023

## Contributing
Suggestions and contributions to the code are both welcomed and encouraged. Please open an issue to discuss your changes and submit a pull request.

## License
SiTraNo is distributed under the MIT Licence. Please refer to [**LICENCE.md**](LICENSE.md) for further information.

## References
* [1] L. Fierro, V. Välimäki. _"Enhanced Fuzzy Decomposition of Sound Into Sines, Transients and Noise"_. In Journal of Audio Engineering Society, July 2023.
* [2] D. Fitzgerald. _“Harmonic/percussive separation using median filtering”_. In Proc. Digital Audio Effects (DAFx), Graz, Austria, Sept. 2010, vol. 13.
* [3] J. Driedger, M. Müller, and S. Disch. _“Extending harmonic-percussive  separation  of  audio  signals”_. In Proc. ISMIR, Taipei, Taiwan, Oct. 2014, pp. 611–616.
* [4]  R. Füg, A. Niedermeier, J. Driedger, S. Disch, and M. Müller. _“Harmonic-percussive-residual  sound  separation  using  the structure tensor on spectrograms”_. In Proc. IEEE Int. Conf.Acoust. Speech Signal Process. (ICASSP), Shanghai, China,Mar. 2016, pp. 445–449.
* [5] E. Moliner,  J. Rämö,  and V. Välimäki. _“Virtual bass system with fuzzy separation of tones and transients”_. In Proc. Digital Audio Effects (DAFx), Vienna, Austria, Sept. 2020.
