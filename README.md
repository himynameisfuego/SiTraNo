# SiTraNo
A MATLAB app for tonal-transient-noise decomposition of audio signals. Developed using App Designer in Matlab 2020b.

![](GUIFinal.png)

* [1] Fierro, L. and Välimäki, B. 2021. [**SiTraNo: a MATLAB app for tonal-transient-noise decomposition of audio signals**]. Submitted to Digital Audio Effects (DAFx) Conference 2021, Vienna, Austria.

## Abstract

Decomposition of sounds into their tonal, transient, and noise components is an active research topic and a widely-used tool in audio processing. Multiple solutions have been proposed in recent years, using time-frequency representations to identify either horizontal and vertical structures or orientations and anisotropy in the spectrogram of the sound. In this paper, we present SiTraNo: an easy-to-use MATLAB application with a graphic user interface for audio decomposition that enables visualization and access to the tonal, transient, and noise classes, individually. This application allows the user to choose between different well-known separation methods to analyze an input sound file, to instantaneously control and remix its spectral components, and to visually check the quality of the separation, before producing the desired output file. The visualization of common artifacts, such as birdies and dropouts, is demonstrated. This application promotes experimenting with the sound decomposition process by observing the effect of variations for each spectral component on the original sound and by comparing different methods against each other, evaluating the separation quality both audibly and visually.

## Dependencies

* Audio Toolbox
* Image Processing Toolbox

## Installation and use
* If your version of MATLAB is 2020b or later, download the latest [release](https://github.com/himynameisfuego/SiTraNo/releases/latest). If your version is 2020a or previous, refer to this [hotfix](https://github.com/himynameisfuego/SiTraNo/files/6351972/SiTraNo_HotFix_1.0.0.1.zip) until the next release.
* In MATLAB, navigate to the SiTraNo folder, open **SiTraNo.mlappinstall** and install. You will find SiTraNo in the "Apps" tab, in the "My apps" group. Click on it to execute the app.
* Upon launching SiTraNo, a navigation folder should pop up, asking you to choose the input audio file.

## Featured decomposition methods

* **HP** (Harmonic-Percussive separation).
* **HPR** (Harmonic-Percussive-Residual separation).
* **ST** (Structure Tensor)
* **Fuzzy** (Fuzzy)

## App description

Three panels.

## Future updates
* Add Standalone installer featuring MATLAB Runtime.
* Adding new separation methods.
* Polish spectrogram presentation.
* Improve code efficiency.
* JUCE version of the app.

## Contributing
Suggestions and contributions to the code are both welcomed and encouraged. Please open an issue to discuss your changes and submit a pull request.

## License
SiTraNo is distributed under the MIT Licence. Please refer to [**LICENCE.md**](LICENSE.md) for further information.
