# Voice Cloning
## 1. Deepvoice3
### 1.1. English
#### 1. Data

||LJSpeech|VCTK|
|:--:|:--:|:--:|
|The Number of Speaker|1|110|
|Capacity|2.6GB|10.9GB|
|Number|13,100 clips||
|Hour|about 24H||
|word/charactor/sentence|225,715 words/1,308,678 characters|44,000 sentences
|Text|7 non-fiction books|400 sentences from newspaper|

|Transcription|

##### LJSpeech (en)
- https://keithito.com/LJ-Speech-Dataset/
- clip
  - segmented automatically based on silences (generally align with sentence or clause boundaries, but not always)
  - duration (mean: 6.57s / min: 1.11s / max: 10.10s), mean words per clip: 17.23

##### VCTK (en)
- https://datashare.ed.ac.uk/handle/10283/3443
  > Original VCTK URL (http://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html) is unavailable.
  > cf) TensorFlow Datasets > Audio https://www.tensorflow.org/datasets/catalog/vctkv

- clips were 
- a single speaker (recorded 2016-2017)
- 7 non-fiction books (published between 1884-1964)

- 19 transcriptions 
- The text was matched to the audio manually
- a QA pass was done to ensure that the text accurately matched
- non-ASCII characters 

- recorded by LibriVox project
- 128 kbps MP3 files


VCTK
The rainbow passage: http://web.ku.edu/~idea/readings/rainbow.htm
the text selection algorithms that increases the contextual and phonetic coverage
(C. Veaux, J. Yamagishi and S. King, "The voice bank corpus: Design, collection and data analysis of a large regional accent speech database," https://doi.org/10.1109/ICSDA.2013.6709856)

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
