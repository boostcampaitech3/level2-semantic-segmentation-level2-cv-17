# MMSegmentation

## Setting for MMSegmentation

- Virtual Environment

```
conda create -n mmseg python=3.8 # must
conda activate mmseg
```
- Install Packages ( Pytorch, MMCV )
제공되는 서버의 cuda가 11.0이기에 이를 기준으로 구성했습니다. 각자의 환경에 맞춰서 구성해주셔도 될 것 같습니다.
```
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7/index.html
```

- Install MMSegmentation
```
git clone https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
pip install -e .  # or "python setup.py develop"
```

## MMSegmentation Data Format
mmsegmentation의 'CustomDataset' 혹은 이를상속받은 'COCOStuffDataset"을 사용할 수 있는데, 여기서는 전자를 사용하려 합니다. 이 때 dataset의 경우 mmsegementation docs를 참고하면 아래와 같은 file structure를 따라야 합니다.
```
├── data
│   ├── my_dataset
│   │   ├── img_dir
│   │   │   ├── train
│   │   │   │   ├── xxx{img_suffix}
│   │   │   │   ├── yyy{img_suffix}
│   │   │   │   ├── zzz{img_suffix}
│   │   │   ├── val
│   │   ├── ann_dir
│   │   │   ├── train
│   │   │   │   ├── xxx{seg_map_suffix}
│   │   │   │   ├── yyy{seg_map_suffix}
│   │   │   │   ├── zzz{seg_map_suffix}
│   │   │   ├── val
```

이러한 file structure를 따르기 위해서 `input/data/convert_mmseg.py`를 실행시켜 아래와 같은 file structure를 만들고자 합니다.
핵심은 두가지 입니다
- image들을 img_dir에 넣어서 구성해주기
- image에 해당하는 annotation된 mask image를 ann_dir에 넣어서 구성해주기

```
|-- mmseg
    |-- ann_dir
        |-- train
        |-- val
    |-- img_dir
        |-- test
        |-- train
        |-- val
```
위와 같이 구성했다면 mmsegmentation config를 구성해주고, train.py를 통해 실행시켜주시면 됩니다.( 아래부터는 추후 업데이트할 예정입니다.)
