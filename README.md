# LAFITE 프로젝트

본 저장소는 [LAFITE: Towards Language-Free Training for Text-to-Image Generation](https://arxiv.org/abs/2111.13792) 논문을 바탕으로 MS-COCO 2014 데이터셋을 사용하여 Language-free 방식으로 학습을 수행하는 과정을 안내합니다.

## 사전 요구사항

다음과 같은 패키지를 설치해야 합니다.

* torch
* torchvision
* click
* clip
* styleGAN

```bash
pip install torch torchvision click
# CLIP 및 StyleGAN은 각 공식 깃허브에서 설치 안내를 참조
```

## 데이터셋 준비

MS-COCO 2014 데이터셋을 다운받아야 합니다.

```bash
curl -O http://images.cocodataset.org/zips/train2014.zip
curl -O http://images.cocodataset.org/zips/val2014.zip
```

압축을 풀어줍니다. **폴더 구조를 반드시 확인하세요.**

## 데이터 전처리

`dataset_tool.py`를 이용해 데이터를 전처리합니다.

```bash
python dataset_tool.py --source=./train2014 --dest=./train2014.zip --width=256 --height=256
python dataset_tool.py --source=./val2014 --dest=./val2014.zip --width=256 --height=256
```

## 학습 실행

학습 스크립트는 다음과 같이 실행합니다.

```bash
python train.py --gpus=1 --outdir=./outputs/ \
--data=./datasets/train2014.zip --test_data=./datasets/val2014.zip \
--temp=0.5 --itd=10 --itc=10 --gamma=10 \
--mixing_prob=1.0 --mirror=1 --kimg=100 --batch=32 --workers=4 \
--metrics=fid50k_full --snap=2
```

## mini 데이터셋 제작

학습 속도를 높이기 위해 mini 데이터셋을 구성할 수도 있습니다.

```bash
python dataset_tool.py --source=./train2014_mini --dest=./train2014_mini.zip
python dataset_tool.py --source=./val2014_mini --dest=./val2014_mini.zip
```

이 데이터셋으로 빠른 테스트 학습이 가능합니다.

## 환경 설정 관련 이슈

학습 중 PyTorch CUDA 관련 플러그인 설치 오류 (`bias_act_plugin`, `upfirdn2d_plugin`)가 발생할 경우 다음을 수행하세요.

### 문제 해결 방법

1. Visual Studio 2019 버전 설치 (2022 버전은 사용하지 않음)
2. 시스템 환경 변수에 Visual Studio 2019의 경로를 가장 상단에 설정
3. Ninja 패키지 설치

```bash
pip install ninja
```

위의 방법으로 해결되지 않을 경우, 다음을 수행하세요:

* 이전 학습 시 생성된 캐시 파일 삭제 (`*.pkl`):

```
C:\Users\사용자\.cache\dnnlib\gan-metric\*.pkl
```

삭제 후 재실행하면 오류가 해결됩니다.

## 학습 과정 로그 설명

학습 로그에서 각 항목의 의미는 다음과 같습니다.

* `tick`: 체크포인트 주기
* `kimg`: 누적 학습 이미지 수 (천 단위)
* `sec/tick`: 체크포인트 주기당 소요 시간 (초)
* `maintenance`: 이미지 저장, 평가 등 비학습 작업 시간
* `cpumem`: CPU 메모리 사용량
* `gpumem`: GPU 메모리 사용량
* `augment`: 데이터 증강 확률 (보통 0.000)

## 추가 정보

### step 0 \~ step 5 설명

학습 시 저장되는 이미지의 `step`은 CLIP 텍스트 조건 문장 리스트의 각 항목을 의미합니다.

```python
text = [
    'A living area with a television and a table',
    'A child eating a birthday cake near some balloons',
    'A small kitchen with low ceiling',
    'A group of skiers are preparing to ski down a mountain',
    'A school bus in the forest',
    'A green train is coming down the tracks'
]
```

각 문장이 step=0\~5 이미지와 매칭됩니다.

## 학습 결과

학습 완료된 모델과 결과는 `./outputs` 디렉토리에 저장됩니다. 필요에 따라 Tensorboard를 통해 학습 과정을 시각화할 수 있습니다.

```bash
pip install tensorboard
```

```bash
tensorboard --logdir=./outputs
```

이제 준비된 환경과 데이터로 성공적으로 LAFITE 학습을 진행할 수 있습니다.
