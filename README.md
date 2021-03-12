# Voice Cloning
## 1. Deepvoice3_English
https://github.com/r9y9/deepvoice3_pytorch

https://uos-deep-learning.tistory.com/20
### 1.1. Data
||LJSpeech|VCTK|
|:--:|:--:|:--:|
|The Number of Speaker|1|110|
|Capacity|2.6GB|10.9GB|
|Number|13,100 clips|-|
|Hour|about 24H|-|
|Word/Charactor/Sentence|225,715 words/1,308,678 characters|44,000 sentences
|Text|7 non-fiction books|400 sentences from a newspaper, the rainbow passage, and  an elicitation paragraph|
 
#### LJSpeech (en)
- https://keithito.com/LJ-Speech-Dataset/
- recorded 2016-2017 by LibriVox project
  > cf) texts published 1884-1964
- clip
  - segmented automatically based on silences (generally align with sentence or clause boundaries, but not always)
  - duration (mean: 6.57s / min: 1.11s / max: 10.10s), mean words per clip: 17.23
- Transcription
  - The text was matched to the audio manually
  - a QA pass was done to ensure that the text accurately matched
  - non-ASCII characters 
  
#### VCTK (en)
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
    - The paragraph contains practically all of the sounds of English.
    
      ![](http://accent.gmu.edu/images/sounds.GIF)
      
### 1.2. Requirements
- CUDA >= 8.0 # https://whereisend.tistory.com/227
- PyTorch >= v1.0.0 # https://pytorch.org/get-started/locally/
- Python >= 3.5
- nnmnkwii >= v0.0.11 # https://github.com/r9y9/nnmnkwii
- Mecab (Japanese only)

```python
## Version check
nvcc -V; nvidia-smi
PyTorch -V
python -V

## Installation
# PyTorch
conda install pytorch==1.2.0 torchvision==0.4.0 -c pytorch # windows
# nnmnkwii
pip install nnmnkwii
```

### 1.3. Installation
```python
git clone https://github.com/r9y9/deepvoice3_pytorch && cd deepvoice3_pytorch
pip install -e ".[bin]"
```

### 1.4. Getting started
#### 1. Download dataset
- 여기서는 우선 LJSpeech data만 대상으로 코드를 실행할 예정임.
- Download URL: https://keithito.com/LJ-Speech-Dataset/ (2.6GB)
- 3에서 clone한 [deepvoice3_pytorch] 폴더 안에 [data] 폴더를 만들고, 그 안에 미리 다운 받은 LJSpeech data를 넣어 놓음.

#### 2. Preprocessing
```python
# 자꾸 모듈 에러가 나서 conda 가상 환경을 만들어서 진행함.
conda create -n pyenv36 python=3.6
activate pyenv36
pip install docopt
pip install tqdm
pip install numpy
# 1.3. Installation의 line 2 재실행
pip install -e ".[bin]" # 꽤 오래 걸림
```

```python
python preprocess.py ${dataset_name} ${dataset_path} ${out_dir} --preset=<json>
# LJSpeech data
python preprocess.py --preset=presets/deepvoice3_ljspeech.json ljspeech ./data/LJSpeech-1.1/ ./data/ljspeech 
```

![Preprocessing](https://user-images.githubusercontent.com/42113942/110910613-f2448380-8354-11eb-8667-f7d32041e85a.JPG)


#### 3. Training
```python
python train.py --data-root=${data-root} --preset=<json> --hparams="parameters you may want to override"
# LJSpeech data
python train.py --preset=presets/deepvoice3_ljspeech.json --data-root=./data/ljspeech/
```
- Lookup error
  
  ![Training](https://user-images.githubusercontent.com/42113942/110910893-49e2ef00-8355-11eb-9749-0854fd34d1e4.JPG)
  - https://uos-deep-learning.tistory.com/20 참고
    - 1) http://www.nltk.org/nltk_data/ 에서 cmudict.zip 다운로드
      ![cmudict](https://user-images.githubusercontent.com/42113942/110911278-d392bc80-8355-11eb-9b59-c81eb12bd526.JPG)
    - 2) error 메시지 내 Searched in 에 있는 디렉토리에 nltk_data 폴더를 새로 생성하고 그 아래 corpora 라는 이름의 폴더 생성후 아래 링크에서 cmudict.zip 을 다운 후 압축해제
      ![folder](https://user-images.githubusercontent.com/42113942/110911875-97139080-8356-11eb-81f3-0bafc400980c.JPG)

#### 4. Monitor with Tensorboard
```python
pip install tensorboard
tensorboard --logdir=log
```

#### 5. Synthesize from a checkpoint

```python
python synthesis.py ${checkpoint_path} ${text_list.txt} ${output_dir} --preset=<json>
# LJSpeech data
python synthesis.py ./checkpoint_step000020000.pth test_list.txt ./output_dir --preset=presets/deepvoice3_ljspeech.json
```

## 2. Deepvoice3_Korean
## 3. LPCNet
## 4. Tensoflow TTS
