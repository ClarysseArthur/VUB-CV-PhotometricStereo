# Group Work Computer Vision
## Authors
> **Arthur Clarysse**<br>
> Vrije Universiteit Brussel<br>
> arthur.clarysse@vub.be

> **Jens Dumortier**<br>
> Vrije Universiteit Brussel<br>
> jens.dumortier@vub.be

> **Guillaume Tibergyn**<br>
> Vrije Universiteit Brussel<br>
> guillaume.tibergyn@vub.be

## Original paper
> Hashimoto, Shuhei, Daisuke Miyazaki, and Shinsaku Hiura. 2019. “Uncalibrated Photometric Stereo Constrained by Intrinsic Reflectance Image and Shape from Silhoutte.” 2019 16th International Conference on Machine Vision Applications (MVA), May, 1–6. https://doi.org/10.23919/MVA.2019.8758025.<br><br>
> Link to the paper: [https://doi.org/10.23919/MVA.2019.8758025](https://doi.org/10.23919/MVA.2019.8758025)

## Running the Code
After installing the dependencies from [requirements.txt](./requirements.txt), you can run [the Jupyter Notebook](./Code/PhotometricStereo.ipynb)

## Hardware
- The code is tested on:
    - Windows 11 (±2 seconds to run, including Plotly 3D visualization)
    - Mac OS 26 Tahoe (±2 seconds to run, including Plotly 3D visualization)

## Results
- The expected results for the cat dataset are the following:
    - Mean error: 0.4
    - Max error: 2.15
    - Min error: 0.0

    ![Expected Results](./Report/IMG/Error/error-cat.png)