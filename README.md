## Recycle Trash Semantic Segmentation Competition
&nbsp;&nbsp;쓰레기 대란, 매립지 부족 등의 사회문제 해결에 기여하고자 대회 형식으로 진행되었던 프로젝트이며, 이미지에서 10종류의 쓰레기(일반 쓰레기, 종이, 금속, 유리, 플라스틱 등)의 영역을 정확히 분할하는 모델을 설계하였습니다. 이는 쓰레기장에 설치되어 정확한 분리수거를 돕거나, 어린아이들의 분리수거 교육 등에 사용될 수 있을 것입니다.      
- Input: 이미지, 쓰레기 종류, 쓰레기 영역 등 Segmentation Annotation (coco format)    
- Output: 쓰레기 종류와 영역( pixel 좌표들)    

![image](https://user-images.githubusercontent.com/39791467/173186400-4571dc1d-05b2-4195-b172-b38ee62b56bf.png)

## 💁TEAM
### CV 17조 MG세대
|민선아|백경륜|이도연|이효석|임동우|
| :--------: | :--------: | :--------: | :--------: | :--------: |
|<img src="https://user-images.githubusercontent.com/78402615/172766340-8439701d-e9ac-4c33-a587-9b8895c7ed07.png" width="120" height="120"/>|<img src="https://user-images.githubusercontent.com/78402615/172766371-7d6c6fa3-a7cd-4c21-92f2-12d7726cc6fc.png" width="120" height="120"/>|<img src="https://user-images.githubusercontent.com/78402615/172784450-628b30a3-567f-489a-b3da-26a7837167af.png" width="120" height="120"/>|<img src="https://user-images.githubusercontent.com/78402615/172766321-3b3a4dd4-7428-4c9f-9f91-f69a14c9f8cc.png" width="120" height="120"/>|<img src="https://user-images.githubusercontent.com/78402615/172766404-7de4a05a-d193-496f-9b6b-5e5bdd916193.png" width="120" height="120"/>|
|[@seonahmin](https://github.com/seonahmin)|[@baekkr95](https://github.com/baekkr95)|[@omocomo](https://github.com/omocomo)|[@hyoseok1223](https://github.com/hyoseok1223)|[@Dongwoo-Im](https://github.com/Dongwoo-Im)|
|mmsegm기반</br>모델 실험|모델 다양성,</br>Augmentation 실험|데이터 Visualization</br>모델 실험|mmsemg baseline작성</br>다양한 실험|torch baseline 작성,</br> Data Annotation 수정|

## 🏆Leaderboard Ranking
- Public: 4등, mIoU 0.8209
- Private: 2등, mIoU 0.7702

## 🧪Experiment
1. Data : 시각화를 통해 실험방향성을 설계하고, unique case 및 label nosie 등을 공유하며 문제를 정의하고 해결해나갔다.
2. Validation : miou의 산식을 생각해볼 때 클래스의 분포에 민감해 Leader board와 상관성이 높은 validatoin set의 선정을 중요하게 생각했다.
3. Augmentation : 다양한 augmentaiton 및 앞서 정의된 문제 해결을 위한 적절한 기법을 활용했다.
4. Loss 및 학습  :  mIoU와 클래스 불균형을 고려해 잘 measure할 수 있는 loss를 고민했고, 사전학습된 backbone의 사용을 통한 인코더 와 디코더의 학습 불균형 해결을 위한 lr trick 등을 고안했다.
5. Ensemble : voting 기법에서 비효율적임을 느껴 Pseudo Labeling 등을 통해 일종의 앙상블 효과를 얻고자 했다.

## ✨Model
Framework, CNN base or Transformer base 등을 고려해 모델의 다양성을 높이고, , Receptive Field를 고려해 local한 정보를 최대한 활용해 기존의 모델이 가진 global한 정보만으로 판단하는 문제를 해결했다. 또한, mIoU산식과 대회 데이터가 가진 불균형한 분포의 문제점을 고려해 정량적 지표 뿐 아니라 결과를 시각화해봄으로써 부족한 클래스를 잘 맞추는 등의 적절한 실험 방향을 설계할 수 있었다.
![image](https://user-images.githubusercontent.com/90104418/173222314-8392d3aa-5d6a-4ec1-9e78-8723e59bb8ba.png)


## 🌈 Pseudo Labeling
### Pseudo Labeling 전략
다양한 Pseudo Labeling전략을 통해 일종의 앙상블 효과를 내도록 함으로써, 다양한 모델이 가진 특성을 살릴 수 있었다.     
1. Train data와 Pseudo Labeling된 Test data를 함께 학습
2. 기학습된 모델에 Pseudo Labeling된 Test data를 포함하여 추가 학습 / 의도 : 기학습된 모델이 잘 예측하지 못하던 케이스에 대한 가이드를 제공함으로써 일종의 앙상블처럼 각 모델의 장점을 가져오고자 했습니다.
3. Train data와 Pseudo Labeling된 Test data를 함께 학습한 후, Train data만 활용하여 추가 학습 / 의도 : Pseudo Labeling 학습 시, 전체적인 경향이 비슷해지면서 Test data에 대한 틀린 예측까지도 학습하는 현상을 보완하고자 했습니다.
![image](https://user-images.githubusercontent.com/90104418/173222191-5576d535-5433-4a77-a1d2-e93395bb8d03.png)


## 📖 Reference
* mmsegmentatoin
    * github : https://github.com/open-mmlab/mmsegmentation
    * documentation : https://mmsegmentation.readthedocs.io
    * 구체적인 실행 방법및 세팅은 mmseg 폴더 안의 README.md를 참고해주세요.

* torch
    * 신규범 캠퍼님이 공유해주신 baseline을 참고했습니다.
