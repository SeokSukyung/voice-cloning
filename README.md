# Voice Cloning
## 1. Deepvoice3
### 1.1. English
#### 1. Data
- LJSpeech (en): https://keithito.com/LJ-Speech-Dataset/
- VCTK (en): http://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html 
> VCTK URL is unavailable.

LJSpeech
- public domain
- 13,100 clips / approximately 24 hours / 225,715 words / 1,308,678 characters
- clip duration (mean: 6.57s / min: 1.11s / max: 10.10s)
- mean words per clip: 17.23 / distinct words: 13,821
- clips were segmented automatically based on silences (generally align with sentence or clause boundaries, but not always)
- a single speaker (recorded 2016-2017)
- 7 non-fiction books (published between 1884-1964)

- 19 transcriptions 
- The text was matched to the audio manually
- a QA pass was done to ensure that the text accurately matched
- non-ASCII characters 

- recorded by LibriVox project
- 128 kbps MP3 files

#### 2. Requirements
- CUDA >= 8.0 # https://whereisend.tistory.com/227
- PyTorch >= v1.0.0 # https://pytorch.org/get-started/locally/
- Python >= 3.5
- nnmnkwii >= v0.0.11 # https://github.com/r9y9/nnmnkwii
- Mecab (Japanese only)

```python
nvcc -V; nvidia-smi
PyTorch -V
python -V
pip install nnmnkwii
```

3. 
https://github.com/r9y9/deepvoice3_pytorch
https://uos-deep-learning.tistory.com/20

### 1.2. Korean
## 2. LPCNet
## 3. Tensoflow TTS
