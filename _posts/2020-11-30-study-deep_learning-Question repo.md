---
published: true
layout: post
title: "딥러닝 관련 궁금한 내용 정리"
subtitle: Question
categories: study
tags: deep_learning
comments: true
---

#  딥러닝관련 궁금한 질문 모음

#### 목차

1. 
2. 





### 1. COCO 데이터셋와 PASCAL VOC 데이터셋, ImageNet 데이터셋의 차이가 뭐길래 객체 인식시 여러가지 도메인으로 performance를 평가하는 걸까?

VOC 데이터셋은 20개의 class를 가지고 있음. 클래스별 AP를  제시함.

COCO는 class가 많아 거의 mAP만 제시







### 2. 딥러닝 알고리즘(YOLO, SSD, Faster-RCNN 등)과 모델 backbone(AlexNet, GoogleNet 등) 간의 관계는 어떤 관계인지..?

예를 들면 YOLO에서 모델 architecture를 ResNet 이런 것으로 바꿔도 되는건지?











### <u>3. YOLO와 성능지표에 대해서</u>

 

> Detector의 성능을 COCO 데이터셋에 대해 평가했을 때,

- MS COCO mAP 기준으로 YOLOv3는 33.0 ≪ RetinaNet은 40.8
- Pascal VOC mAP (AP50) 기준으로 YOLOv3는 57.0 ≒ RetinaNet 61.1 (속도는 YOLOv3가 4배 빠름)

 

똑같은 mAP 인데 왜 이렇게 차이가 나는 걸까?

 ☞ 과거 Pascal VOC 기준으로는 YOLO가 단연 최고의 detector인데, 최근 논문들이 사용하는 COCO 챌린지 기준으로는 속도만 빠른 detector로 전락해 버린다.

 

도대체 뭐가 다른지.. 이번 기회에 mAP 계산 방식에 대해 살펴봤는데 이게 여간 복잡하지 않다.

\- [Evaluation metrics for object detection and segmentation: mAP](https://kharshit.github.io/blog/2019/09/20/evaluation-metrics-for-object-detection-and-segmentation)

 

Pascal VOC는 IoU(Intersection over Union) > 0.5 인 detection은 true, 그 이하는 false로 평가하는 방식이고 COCO는 IoU>0.5, IoU>0.55, IoU>0.6, …, IoU>0.95 각각을 기준으로 AP를 계산한 후 이들의 평균을 취하는 방식이다.

 

COCO 방식은 기존 Pascal VOC 방식이 정확도가 높은 (ground truth와 일치하는 box를 검출하는) detector들을 구분하지 못하는 단점을 보완하기 위해서 도입한 것이라고 한다. 나름 일리가 있다.

 

그런데, yolo 저자는 이에 대해 다음과 같이 반박한다.

1. > ground truth는 사람이 만든 것으로서 오차가 있을 수 있으며 그러한 오차를 감안하여 Pascal VOC에서는 IoU 0.5라는 기준을 적용하였다. (사실, 무엇이 정답인지에 대한 기준 자체도 모호하다. 예를 들어 팔을 벌리고 있는 사람의 가장 이상적인 bounding box는 어디인가? 저자는 box를 사용하면서 정확도를 평가한다는 것 자체가 멍청한 짓이라고 말한다)

2. > IoU 기준을 높이다 보면 상대적으로 classification 정확도는 무시되게 된다. box의 위치 정확도가 물체의 클래스 인식율보다 중요한 것인가? (예를 들어 IoU>0.7 기준을 사용하면, 그 이하의 검출은 물체를 사람으로 인식했는지 자동차로 인식했는지 여부와 무관하게 모두 false로 간주된다)

3. > 물체 class별로 각각 AP를 계산하는 방식도 문제이다. 사람 class의 경우, 현재의 평가방식은 detector의 출력들 중 사람에 대한 출력값만 가지고 precision, recall을 계산한다. 그래서, detector가 실제 사람을 치타라고 분류하거나 강아지라고 분류하더라도 이것들은 성능 평가에 아무런 영향을 미치지 않는다. 지금처럼 개별 class별로 AP를 계산한 후 평균하는 방식이 아니라 모든 class를 한꺼번에 놓고 평가하는 방식으로 바꿔야 한다. (이건 조금 이해하기 힘들 수 있는데, multi-class classification 문제에서는 class 개수만큼 ouput 노드를 만들고 그중 가장 출력이 큰 값을 해당 객체의 class로 분류한다. 그런데, 해당 classifier로 사람만 평가한다고 하면 사람 노드의 출력값만 이용해서 출력값>threshold인 경우에 대해 precision-recall 그래프를 그리게 되고, 다른 class 노드에서 사람 노드보다 더 높은 출력값이 나오더라도 이 값은 무시되게 된다)

 

개인적으로는 yolo 저자의 주장에 대부분 공감이 간다. IoU>0.5라는 기준이 사실 그렇게 낮은 기준이 아니다. 크기가 동일한 box라 하더라도 ground truth와 66.7% 이상 overlap이 되어야 IoU = 0.5가 나오고 사람의 눈으로 봤을 땐 대부분 잘 찾았네 하고 느껴지는 수준이다.

 COCO 기준이 IoU>0.5 AP와 IoU>0.75 AP의 평균을 사용한다면 어느정도 수긍할 수 있다. 그런데, IoU>0.95까지 동일한 weight로 평균한 것은 수긍이 어렵다. 

 기준이 어찌되었든 모든 detector들에게 동일하게 적용되는 것이니 공평한 것 아니냐 할 수도 있다. 하지만, COCO 방식이 찾은 물체의 사람/자동차/물건 구분은 종종 틀리더라도 COCO가 정한 ground truth 박스와 최대한 일치하게 물체의 bounding box를 찾아주는 detector들에게 유리한 방식임은 틀림없다.

 

그런데, Joseph Redmon이 yolo를 중단한 후, 그동안 yolo에 대한 다양한 플랫폼 빌드([github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet))를 제공하던 Alexey Bochkovskiy이 대대적인 optimization을 통해 올해 2020년 4월에 YOLOv4를 release한다. 그리고 YOLOv4는 MS COCO 지표로 mAP 43.5% (Pascal VOC mAP로는 65.7%), 속도 65 fps를 발표한다.

 

by 다크 프로그래머



### <u>4. CNN에서 Fully connected layer의 역할?</u>

CNN에서 

Convolutional layer는 이미지의 특징점을 효과적으로 찾기 위해 사용.

Fully connected layer는 발견한 특징점을 기반으로 이미지를 분류하는데 활용.



1. Convolutional layer

   - Convoluational layer를 사용하는 목적은 입력된 이미지에서 테두리, 선, 색 등 이미지의 시각적인 특징이나 특성을 감지하기 위해서임.

   - **입력계층, Window, Stride, Padding, Filter(Kernel)**

   - 좌표를 겹쳐가면서 윈도우를 이동한다면 이미지의 가장자리 부분은 내부의 영겨보다 탐색 기회가 적어진다.

     ->가장 외곽의 1픽셀씩은 1회만 탐색, 두번째 픽셀은 2회만 탐색.

   - 따라서 패딩을 도입하여 윈도우를 이미지 영역 밖으로 확장하고 전체 이미지를 공평하게 탐색하도록 한다.

   - 필터는 입력계층의 window를 은닉 계층의 뉴런 하나로 압축할 때 window 크기 만큼의 가중치와 1개의 편향값(bias)를 적용하게 되는데 이를 필터라 칭함.

   - 필터를 사용하게 되면 입력된 이미지를 인식처리하기 위해서 필요한 가중치의 수를 줄일 수 있고, 가중치의 수가 줄어든다면 전체 연산량을 대폭으로 감소시킬 수 있음.

2. Pooling layer

   - CNN 모델에서는 이미지 데이터, 즉 2차원의 평면행렬에서 지정한 영역의 값들을 하나의 값으로 압축을 하게됨.
   - 여기서 압축할 때, Convolutional layer는 필터, 즉 가중치와 편향 값을 적용하는 역할을 하고, Pooling layer는 해당 범위의 값들 중에서 하나를 선택해서 가져오는 역할을 함. 
   - Pooling의 목적은 1. input size를 줄임, 2. Overfitting을 조절, 3. Feature를 잘 뽑아냄.

3. LCN layer(Local Contrast Normalization)
   
   - 우리가 입력하는 이미지 중에서자연물 이미지와 같이 주변의 조명, 카메라의 노출 등 환경의 변화에 따라서 이미지 전체의 밝기, 대비가 크게 변하는 경우 LCN layer를 사용함.
4. Fully Connected Layer
   - FCN은 기존의 신경망에서 각 층별 연결에 사용되는 방식으로 전결합층임.
   - 기존의 신경망 모델은 보유하고 있는 모든 노드가 서로 연결이 되어있음.
   - CNN의 특징은 모든 노드를 결합하지 않음으로써 연산량을 줄이고 효율성을 높이는 방식임.
   - CNN 모델을 통해서 지금까지 학습해온 데이터는 입력된 이미지를 분류한다는 최종 목적을 가지고 있음.
   - 최종 분류를 하기 위해서는 결국 분류하기 위한 목록에서 어떤 Label을 선택해야하는지가 핵심임.
   - 최종 결과를 분류하기 위한 기반 정보를 모두 가지고 있어야 분류를 위한 Softmax 함수를 사용할 수 있음.
   - 따라서 FCN은 지금까지 처리된 결과데이터를 가진 모든 노드를 연결시켜 1차원 배열로 풀어서 표시함.
5. Softmax layer
   
   - 2개 이상의 categories를 분류할 때 사용하는 함수로, 각각의 카테고리 별로 계산된 확률값을 이용하여 분류를 수행함.



### 5. 신경망 설계시 고려해야 하는 하이퍼 파라미터들은 어떠한 것이 있는가?

1. 신경망의 깊이
   - Number of layer
2. Dropout
   - 목적 : 
3. 가중치 초기화
4. Regularlization
   - Batch normalization
5. 활성화함수
6. Batch size
7. 

### <u>6. Regularization이란 무엇인가?</u>

- 위키백과에서 정의된 사전 정의는 아래와 같음.
  "Regularization is the process of adding information in order to solve an ill-posed problem or to prevent overfitting."

- Overfitting을 피한다는 것은 DNN의 일반화(Generalization) 성능을 높인다는 말임.
- 모델은 가중치를 업데이트해서 무언가를 분류하는데 가중치가 업데이트 될 때, 학습데이터에 편중되지 않도록 해주면 모델의 일반화 성능을 높일 수 있지 않을까?
- 좀더  풀이하면 업데이트 되는 가중치에 규제를 가해야 한다는 것이다. Regularization의 종류에는 L1과 L2가 있음.
- 내용을 요약하면 아래와 같음.
  1. 가중치가 큰 값은 overfitting을 일으킬 요인이 크기 때문에 학습시 페널티를 주어야한다는 관점에서 나온 것이 weight decay regularization이다.
  2. Weight decay regularization 종류에는 L1-regularization, L2-regularization이 있음
  3. 가중치가 큰 정도를 판단하기 위해 가중치를 하나의 벡터라고 가정했고, 그 벡터의 크기를 가중치의 크기로 보고자 L1-norm, L2-norm이라는 개념을 도입해 L1-regularization, L2-regularization을 고안함.
  4. L1-regularization은 L2-regularization보다 낮은 가중치 값을 0으로 만들어줘 입력차원을 줄여주는 경향이 있고, 이것은 입력차원을 낮춰주는 효과가 있음.
  5. 보통의 weight decay regularization에서는 L2-regularization이 사용되고 있음.



### <u>7. 차원의 저주란? (Curse of dimension)</u>

1. 주어진 데이터 샘플에 대한 세밀하고 밀도 있는 데이터(=고차원 데이터)는 많은 정보를 담고 있지만, 딥러닝 모델 관점에서 고차원 입력데이터는 부작용을 일으킬 가능성이 많다.

2. 즉, 고차원 입력데이터일수록 훨씬 더 많은 데이터량이 필요하게 되는데, 이것을 차원의 저주라고 한다.

3. 하지만 현실적으로 방대한 양의 데이터를 모으는데 힘든 부분이 많음.

4. 또한 입력차원의 데이터가 고차원이어도 실제분석에서는 저차원으로 구성된 데이터가 더 분류성능을 높일 수 있음. 
   왜냐하면 가끔씩 우리가 품질이 좋고 세말하며 밀도있는 데이터를 구성했다고 하지만, 그건 언제나 우리 관점에서이다.

   다시 말해, 없어도 되는 변수들을 데이터에 추가했을 가능성이 있음.(오히려 분류에 방해되는 변수를 넣을 가능성이 있음.)

5. 이러한 이유들 때문에 어떻게든 입력데이터의 차원을 줄여주려는 노력을 하고 있음.
   EX) Feature extraction : PCA
         Feature selection : Correlation Analysis, VIF(Variance Inflation Factor), Random Foreast



### <u>8. Batch normalization은 무엇인가?</u>

- 배치 정규화는 학습의 효율을 높이기 위해 도입됨. 배치 정규화는 Regularization을 해준다고 볼 수 있음.
  - 학습 속도가 개선됨.(학습률을 높게 설정할 수 있기 때문)
  - 가중치 초깃값 선택의 의존성이 적어짐.(학습을 할 때마다 출력값을 정규화하기 때문)
  - 과적합(Overfitting) 위험을 줄일 수 있음.(Dropout 같은 기법 대체 가능)
  - Gradient vanishing 문제 해결
- 배치 정규화는 활성화함수의 활성화값 또는 출력값을 정규화(정규분포로 만든다)하는 작업을 말한다.
- 신경망의 각 layer에서 데이터(배치)의 분포를 정규화하는 작업임.
- 일종의 노이즈를 추가하는 방법으로 (bias와 유사) 이는 배치마다 정규화를 함으로써 전체 데이터에 대한 평균의 분산과 값이 달라질 수 있음.
- 학습을 할 때마다 활성화값/출력값을 정규화하기 때문에 초기화(가중치 초깃값) 문제에서 비교적 자유로워짐.



### 9. AutoML이란 무엇인가?





### 10. Sampling, subsampling, upsampling, downsampling 용어가 뭘까?





### 11. Semantic information이 뭘까?

https://velog.io/@haejoo/Feature-Pyramid-Networks-for-Object-Detection-%EB%85%BC%EB%AC%B8-%EC%A0%95%EB%A6%AC

에 따르면

> CNN에서 resolution을 줄여가면서 feature를 extracting하는데 이전에 Extract된 high resolution의 feature은 나중에 extract된 low resolution의 feature를 전혀 반영하지 못한다는거다. 나중에 뽑힌게 더 rich semantic information을 가지고 있을텐데도 말이다. ㅠㅠ 논문에서는 이걸 *large segmantic gaps caused by different depth*라고 표현한다.

> 그렇다면 pyramidal feature hierarchy 처럼 연산량도 적고, 또 sematically weak feature인 high resolution의 feature와 semantically strong feature인 low resolution의 feature를 어떻게 잘 연결해서 장점을 모두 취하는 아키텍처가 없을까 !? 그게 바로 이 논문에서 소개하는 **Feature Pyramid Network** 이ㄷㅏ ^**___**^

여기서 언급하는 sematically weak feature가 어떠한 의미를 가지는지 궁금함.



#### 12. Ensemble이란 뭘까?



