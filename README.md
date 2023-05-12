# Perceptual Analysis of Speaker Embeddings for Voice Discrimination 

This repository includes the source code that can be used to reproduce the speaker discrimination listening test 
from the paper:

I. Thoidis, C. Gaultier and T. Goehring, "Perceptual Analysis of Speaker Embeddings for Voice Discrimination 
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

  
# Getting Started

We provide an environment file that can be used to install the project dependencies. Open your terminal, 
navigate to the directory you saved the file to, and run:
```
conda create -f environment.yml
```

# Run the listening test

Running the `speaker.py` file will check, download (if needed) all the necessary data and export the stimuli.

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

