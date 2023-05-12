# Perceptual Analysis of Speaker Embeddings for Voice Discrimination 

This repository includes the source code that can be used to reproduce the speaker discrimination listening test 
from the paper:

I. Thoidis, C. Gaultier, and T. Goehring, "Perceptual Analysis of Speaker Embeddings for Voice Discrimination 
between Machine And Human Listening," ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and 
Signal Processing (ICASSP), Rhodes Island, Greece, 2023, pp. 1-5, doi: [10.1109/ICASSP49357.2023.10094782](https://doi.org/10.1109/ICASSP49357.2023.10094782).

## Authors
* **Iordanis Thoidis**, 
School of Electrical and Computer Engineering, 
Aristotle University of Thessaloniki, Thessaloniki, Greece
* **Clément Gaultier**,  Cambridge Hearing Group,
MRC Cognition and Brain Sciences Unit, 
University of Cambridge, UK
* **Tobias Goehring**, Cambridge Hearing Group,
MRC Cognition and Brain Sciences Unit, 
University of Cambridge, UK

## Abstract 
This study investigates the information captured by speaker embeddings with relevance to human speech perception. 
A Convolutional Neural Network was trained to perform one-shot speaker verification under clean and noisy conditions, 
such that high-level abstractions of speaker-specific features were encoded in a latent embedding vector. We demonstrate
that robust and discriminative speaker embeddings can be obtained by using a training loss function that optimizes the 
embeddings for similarity scoring during inference. Computational analysis showed that such speaker embeddings predicted 
various hand-crafted acoustic features, while no single feature explained substantial variance of the embeddings. 
Moreover, the relative distances in the speaker embedding space moderately coincided with voice similarity, as inferred 
by human listeners. These findings confirm the overlap between machine and human listening when discriminating voices 
and motivate further research on the remaining disparities for improving model performance.
  
![plot](./Audio/abstract.png)

# Getting Started

We provide an environment file that can be used to install the project dependencies. Open your terminal, 
navigate to the directory you saved the file to, and run:
```
conda env create -f environment.yml
```

# Run the listening test

Running the `speaker.py` file will check and download (if needed) all the necessary data before starting the listening 
test procedure. 

## Results

After running the listening test, the result file can be found in the `Results/speaker` directory.

### Reference
* If using this code in your study, please consider citing the above paper.

```
@INPROCEEDINGS{10094782,
  author={Thoidis, Iordanis and Gaultier, Clément and Goehring, Tobias},
  booktitle={ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Perceptual Analysis of Speaker Embeddings for Voice Discrimination between Machine And Human Listening}, 
  year={2023},
  volume={},
  number={},
  pages={1-5},
  doi={10.1109/ICASSP49357.2023.10094782}}
```

