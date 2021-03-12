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
|Text|7 non-fiction books|400 sentences from a newspaper, the rainbow passage, and  an elicitation paragraph|

|Transcription|19

##### LJSpeech (en)
- https://keithito.com/LJ-Speech-Dataset/
- recorded 2016-2017 by LibriVox project; texts published between 1884-1964 128 kbps MP3 files
- clip
  - segmented automatically based on silences (generally align with sentence or clause boundaries, but not always)
  - duration (mean: 6.57s / min: 1.11s / max: 10.10s), mean words per clip: 17.23
- Transcription
  - The text was matched to the audio manually
  - a QA pass was done to ensure that the text accurately matched
  - non-ASCII characters 
  
##### VCTK (en)
- https://datashare.ed.ac.uk/handle/10283/3443
  > Original VCTK URL (http://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html) is unavailable.
  > cf) TensorFlow Datasets > Audio https://www.tensorflow.org/datasets/catalog/vctkv

- Text
  - the text selection algorithms that increases the contextual and phonetic coverage (C. Veaux, J. Yamagishi and S. King, "The voice bank corpus: Design, collection and data analysis of a large regional accent speech database," https://doi.org/10.1109/ICSDA.2013.6709856)
  - The rainbow passage: http://web.ku.edu/~idea/readings/rainbow.htm
    > When the sunlight strikes raindrops in the air, they act as a prism and form a rainbow.
The rainbow is a division of white light into many beautiful colors. These take the shape
of a long round arch, with its path high above, and its two ends apparently beyond the
horizon. There is, according to legend, a boiling pot of gold at one end. People look, but
no one ever finds it. When a man looks for something beyond his reach, his friends say
he is looking for the pot of gold at the end of the rainbow. Throughout the centuries
people have explained the rainbow in various ways. Some have accepted it as a miracle
without physical explanation. To the Hebrews it was a token that there would be no more
universal floods. The Greeks used to imagine that it was a sign from the gods to foretell
war or heavy rain. The Norsemen considered the rainbow as a bridge over which the gods
passed from earth to their home in the sky. Others have tried to explain the phenomenon
physically. Aristotle thought that the rainbow was caused by reflection of the sun's rays
by the rain. Since then physicists have found that it is not reflection, but refraction by the
raindrops which causes the rainbows. Many complicated ideas about the rainbow have
been formed. The difference in the rainbow depends considerably upon the size of the
drops, and the width of the colored band increases as the size of the drops increases. The
actual primary rainbow observed is said to be the effect of super-imposition of a number
of bows. If the red of the second bow falls upon the green of the first, the result is to give
a bow with an abnormally wide yellow band, since red and green light when mixed form
yellow. This is a very common type of bow, one showing mainly red and yellow, with
little or no green or blue.

  - an elicitation paragraph: http://accent.gmu.edu; https://sites.ualberta.ca/~aacl2009/PDFs/WeinbergerKunath2009AACL.pdf
    > Please call Stella. Ask her to bring these
things with her from the store: six spoons
of fresh snow peas, five thick slabs of blue
cheese, and maybe a snack for her
brother Bob. We also need a small plastic
snake and a big toy frog for the kids. She
can scoop these things into three red
bags, and we will go meet her Wednesday
at the train station. 


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
