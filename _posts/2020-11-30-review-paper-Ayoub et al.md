---
published: true
layout: post
title: "[Nature] Planning chemical syntheses with deep neural networks and symbolic AI"
subtitle: Nature
categories: review
tags: plant_engineering
comments: true
---

# ?번째 논문 리뷰

- 목차
  1. [Paper info.](#paper-info)
  2. [Introduction](#introduction)
  4. [Background](#background)
  5. Methodology
  6. Result
  7. Contribution
  
- [생소한 영어 단어](#생소한-영어-단어)

## Paper info

​	Journal : [International Conference on Image and Signal Processing](https://link.springer.com/conference/icisp) 2020

​	Title : Convolutional Neural Networks Backbones for Object Detection

​	Author : Ayoub Benali Amjoud, Mustapha Amrouch 등

​	Published date : 2020/07



## Abstract

  본 논문에서 특히 object detection task에서 딥러닝과 convolutional neural network의 중요한 역할을 나타내고자 한다.

  우리는 객체 인식 모델에서 backbone으로서 역할을 하는 다양한 최신의 CNN에 집중하여 분석한다.

  우리는 그것들을 common 데이터셋에 테스트 및 평가해본다.

  각 architecture의 주요 특징을 살펴보고 몇몇의 CNN 구조의 적용이 <u>이미지 분류</u>와 <u>객체 인식</u> task에서 SOTA 결과를 낸것을 증명한다.



## Introduction

AlexNet, VGGNet, ResNet과 같은 several CNN들이 feature 추출에 사용되었다.

이 네트워크들은 주로 object classification task에 사용됌.

ImageNet 데이터는 image classification 목적임(A single object in the image, outputs a single category per image).

COCO, PASCAL VOC 데이터는 object detecion 목적임(Several objects in a single image and provides the coordinates).



초기 object detection은 사람의 두눈은 명암이 어둡고 코는 명암이 밝은 점을 이용해 패턴을 구하는 Haar-Like features방법, HOG, Scale-Invariant Feature Transform 방법 등을 사용했었다. 이 접근들은 우리가 manually 우리의 사고에 따라서 model feature를 추출할 수 있다는 가정을 기반으로 한다.

이러한 image의 feature를 machine이 대신 수행하게 하는게 더 효율적으로 증명되었고 그것이 CNN임.



## Convolutional Neural Network Backbones

#### 2.0 LeNet-5

- 1998년, Image classification에 보편적으로 사용되는 CNN을 최초로 제안한 Yann LeCun의 LeNet-5.
- 기존의 FC multi-layer network가 갖는, 즉 MLP가 가지는 한계점인 
  1. input pixel 수가 많아지면 parameter가 기하급수적으로 증가하는 문제
  2. Local한 distortion(ex, image를 1pixel shift)에 취약한 문제.
- **Input을 1차원적으로 바라보던 관점에서 2차원으로 확장하였고**, **parameter sharing을 통해** input의 pixel 수가 증가해도 **parameter 수가 변하지 않는다.**

- Input image 크기는 32X32.
- Conv layer 2개, FC layer 3개로 구성됨.



#### 2.1 AlexNet, 2012

- 2012년 개발된 Krizhevsky et al. CNN으로 5개의 Convolutional layer와 3개의 fully connected layer로 구성된 네트워크임.

- LeNet-5와 비교해서 더많은 layer와 6천만개의 parameter가 있음. 

- 비선형성을 추가해주기 위해서 sigmoid와 tanh 활성화 함수 대신 ReLu 함수를 사용함.
- Input image는 224X224, RGB 3 channel image.
- Filter size는 11X11, 5X5, 3X3 등 다양함. Max pooling 적용.



#### 2.2 ZFNET, 2013

- 2013년 ILSVRC 대회에서 우승한 Clarifai 팀의 ZFNET.
- AlexNet을 기반으로 첫 Conv layer의 filter size를 11에서 7로, stride를 4에서 2로 바꾸고, 그 뒤의 Conv layer들의 filter 개수를 키워주는 등(Conv3,4,5: 384, 384, 256 –> 512, 1024, 512) 약간의 튜닝을 거쳤음.
- 이 논문은 architecture에 집중하기 보다는, 학습이 진행됨에 따라 feature map을 시각화하는 방법과, 모델이 어느 영역을 보고 예측을 하는지 관찰하기 위한 Occlusion 기반의 attribution 기법 등 시각화 측면에 집중한 논문이라고 할 수 있습니다.



#### 2.3 VGG-16, 2014

- 2014년 ILSVRC 대회에서 2위의 성적을 거둔 Visual Geometry Groupdml VGG-16.
- 기존 방식들과 다르게 비교적 작은 크기인 3X3 conv filter를 깊게 쌓음.
- AlexNet과 ZFNET은 8개의 layer을 사용하였다면 VGG는 11개, 13개, 16개, 19개 등 더 많은 수의 layer를 사용.
- GPU의 연산속도가 빨라지면서 신경망 층이 깊어짐에도 불구하고 빠른 연산이 가능해짐.
- 이렇게 3x3 filter를 중첩하여 쌓는 이유는, 3개의 3x3 conv layer를 중첩하면 1개의 7x7 conv layer와 receptive field가 같아지지만, activation function을 더 많이 사용할 수 있어서 더 많은 비선형성을 얻을 수 있으며, parameter 수도 줄어드는 효과를 얻을 수 있음. (3x3x3 = 27 < 7x7 = 49)
- 하지만 신경망의 깊이가 깊어져 vanishing gradient 문제가 발생.



#### 2.4 GoogLeNet, 2014

- 2014년 ILSVRC 대회에서 우승을한 Inception architecture라는 예명을 갖는 GoogLeNet.
- 총 22개의 layer로 구성됨.
- GoogLeNet의 주요 특징은 아래 3가지임.
  1. Inception module이라 불리는 block 구조를 제안함.
     - 기존에는 각 layer 간에 하나의 convolution 연산, 하나의 pooling 연산으로 연결을 하였다면, inception module은 총 4가지 서로 다른 연산을 거친 뒤 feature map을 channel 방향으로 합치는 concatenation을 이용하고 있음.
     - 다양한 receptive field를 표현하기 위해 1x1, 3x3, 5x5 convolution 연산을 섞어서 사용을 하였습니다.  이를 **Naïve Inception module** (Multi-scale convolutional transformations)이라 부름.
     - 여기에 추가로, 3x3 conv, 5x5 conv 연산이 많은 연산량을 차지하기 때문에 두 conv 연산 앞에 1x1 conv 연산을 추가하여서 feature map 개수를 줄인 다음, 다시 3x3 conv 연산과 5x5 conv 연산을 수행하여 feature map 개수를 키워주는 **bottleneck** 구조를 추가한 **Inception module with dimension reduction** 방식을 제안함.
     - 이 덕에 Inception module의 연산량을 절반이상 줄일 수 있음.
  2. GoogLeNet은 Inception module을 총 9번 쌓아서 구성이 되며, 3번째와 6번째 Inception module 뒤에 classifier를 추가로 붙여서 총 3개의 classifier를 사용하였고, 이를 **Auxiliary Classifier** 라 부름.
     -  가장 뒷 부분에 Classifier 하나가 존재하면 input과 가까운 쪽(앞 쪽)에는 gradient가 잘 전파되지 않을 수 있는데, Network의 중간 부분, 앞 부분에 추가로 softmax Classifier를 붙여주어 vanishing gradient를 완화시킬 수 있다고 주장하고 있음.
     - 다만 Auxiliary Classifier로 구한 loss는 보조적인 역할을 맡고 있으므로, 기존 가장 뒷 부분에 존재하던 Classifier보단 전체적으로 적은 영향을 주기 위해 0.3을 곱하여 total loss에 더하는 식으로 사용을 하였다고 합니다. 
     - 학습 단계에만 사용이 되고 inference 단계에선 사용이 되지 않으며, 이유론 inference 시에 사용하면 성능 향상이 미미하기 때문입니다.
  3. 대부분의 CNN의 대부분의 parameter를 차지하고 있는 Fully-Connected Layer를 NIN 논문에서 제안된 방식인 **Global Average Pooling(GAP)** 으로 대체하여 parameter 수를 크게 줄이는 효과를 얻었음.
     - GAP란 각 feature map의 모든 element의 평균을 구하여 하나의 node로 바꿔주는 연산을 뜻하며, feature map의 개수만큼의 node를 output으로 출력하게 됩니다. 
     - GoogLeNet에서는 GAP를 거쳐 총 1024개의 node를 만든 뒤 class 개수(ImageNet=1000)의 output을 출력하도록 하나의 Fully-Connected layer만 사용하여 classifier를 구성하였습니다. 
     - 그 덕에 AlexNet, ZFNet, VGG 등에 비해 훨씬 적은 수의 parameter를 갖게 되었습니다.

#### 2.5 ResNets, 2015

- 2015년, ILSVRC 대회에서 우승한 Microsoft Research에서 제안한 구조임.
- Architecture의 이름은 본 논문에서 제안한 핵심 아이디어인 **Residual Block**에서 유래하였으며, 실제로도 이 Residual Block 하나만 알면 architecture를 이해할 수 있을 정도로 단순하면서 효과적인 구조를 제안함.

- ResNet은 3x3 conv가 반복된다는 점에서 VGG와 유사한 구조를 가지고 있음.
- Layer의 개수에 따라 ResNet-18, ResNet-34, ResNet-50, ResNet-101, ResNet-152 등 5가지 버전으로 나타낼 수 있으며, ILSVRC 2015 대회에선 ResNet-152로 1위를 차지하였습니다.
- Layer 개수를 많이 사용할수록 연산량과 parameter 개수는 커지지만 정확도도 좋아지는 효과를 얻을 수 있습니다.
- Shortcut으로는 **identity shortcut**, 즉 input feature map x를 그대로 output에 더해주는 방식을 사용합니다.
- **projection shortcut** 을 이용합니다. 이러한 shortcut 구조를 통해 vanishing gradient에 강인한 학습을 수행할 수 있게됩니다.
-  ResNet-50 이상의 모델에서는 feature map의 개수가 많다 보니 연산량도 많아지게 되는데, Inception module에서 보았던 bottleneck 구조를 차용하여 **bottleneck residual block** 을 중첩하여 사용하는 점이 특징입니다.
- 마지막으론 같은 2015년에 제안이 되었고, 지금도 굉장히 자주 사용되는 방식인 **Batch Normalization(BN)** 을 Residual block에 사용을 하였으며, **Conv-BN-ReLU** 순으로 배치를 하였습니다.



#### 2.6 Pre-Act ResNet, 2016

- 2016년 CVPR에 발표된 “Identity Mappings in Deep Residual Networks” 라는 논문에서 제안한 **Pre-Act ResNet** 입니다.
- Pre-Act는 Pre-Activation의 약자로, Residual Unit을 구성하고 있는 Conv-BN-ReLU 연산에서 Activation function인 ReLU를 Conv 연산 앞에 배치한다고 해서 붙여진 이름입니다. 
  - ResNet의 성능을 개선하기 위해 여러 종류의 shortcut 방법에 대해 실험을 해봄. 실험 결과 original, 즉 identity sortcut일 때 가장 성능이 좋았음.
  - Activation function의 위치에 따른 test error 결과를 보임. 실험 결과 **full pre-activation** 구조일때 가장 test error가 낮았고, 전반적인 학습 안정성도 좋아지는 결과를 보임.
- 제안된 full pre-activation 구조는 모든 Conv 연산에 normalized input이 전달되기 때문에 좋은 성능이 관찰되는 것이라고 분석함.



#### 2.7 Inception-v2, 2016

- 2016년 CVPR에  “Rethinking the Inception Architecture for Computer Vision” 라는 제목으로 발표가 된 논문입니다. GoogLeNet의 후속연구임.
- Inception-v2의 핵심 요소는 크게 3가지로 나눌 수 있습니다.
  - Conv Filter Factorization
  - Rethinking Auxiliary Classifier
  - Avoid representational bottleneck
- Inception-v1(GoogLeNet)은 VGG, AlexNet에 비해 parameter수가 굉장히 적지만, 여전히 많은 연산량을 필요로 합니다. Inception-v2에서는 연산의 복잡도를 줄이기 위한 여러 **Conv Filter Factorization** 방법을 제안하고 있습니다. 우선 VGG에서 했던 것처럼 5x5 conv를 3x3 conv 2개로 대체하는 방법을 적용합니다. 여기서 나아가 연산량은 줄어들지만 receptive field는 동일한 점을 이용하여 n x n conv를 1 x n + n x 1 conv로 쪼개는 방법을 제안합니다.
-  Inception-v1(GoogLeNet)에서 적용했던 **auxiliary classifier**에 대한 재조명을 하는 부분입니다. 여러 실험과 분석을 통해 auxiliary classifier가 학습 초기에는 수렴성을 개선시키지 않음을 보였고, 학습 후기에 약간의 정확도 향상을 얻을 수 있음을 보였습니다. 또한 기존엔 2개의 auxiliary classifier를 사용하였으나, 실제론 초기 단계(lower)의 auxiliary classifier는 있으나 없으나 큰 차이가 없어서 제거를 하였다고 합니다.
- 마지막으론 representational bottleneck을 피하기 위한 효과적인 **Grid Size Reduction** 방법을 제안하였습니다. representational bottleneck이란 CNN에서 주로 사용되는 pooling으로 인해 feature map의 size가 줄어들면서 정보량이 줄어드는 것을 의미합니다. 이해를 돕기 위해 위의 그림으로 설명을 드리면, 왼쪽 사진과 같이 pooling을 먼저 하면 Representational bottleneck이 발생하고, 오른쪽과 같이 pooling을 뒤에 하면 연산량이 많아집니다. 그래서 연산량도 줄이면서 Representational bottleneck도 피하기 위해 가운데와 같은 방식을 제안하였고, 최종적으론 맨 오른쪽과 같은 방식을 이용하였다고 합니다.



#### 2.8 Inception-v3, 2016

- 2016년에 개발된 모델로 Inception-v2의 architecture는 그대로 가져가고, 여러 학습 방법을 적용한 버전입니다. 한 논문에서 Inception-v2와 Inception-v3를 동시에 설명하고 있습니다.
  - Model Regularization via Label Smoothing
    - one-hot vector label 대신 smoothed label을 생성하는 방식이며 자세한 설명은 [**제가 작성했던 글** ](https://hoya012.github.io/blog/Bag-of-Tricks-for-Image-Classification-with-Convolutional-Neural-Networks-Review/)의 3-B를 참고하시기 바랍니다.
  - Training Methodology
    - Momentum optimizer –> RMSProp optimizer / gradient clipping with threshold 2.0 / evaluation using a running average of the parameters computed over time
  - BN-auxiliary
    - Auxiliary classifier의 FC layer에 BN을 추가



#### 2.9 Inception-v4, 2017

- Inception 시리즈의 최종 진화형인 Inception-v4는 2017년 AAAI에 “Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning” 라는 제목으로 발표가 되었으며, 이 논문에서 Inception-v4와 Inception-ResNet 구조를 제안하였습니다. 
- Input과 바로 연결되는 Stem block과 3개의 Inception Block(Inception-A, Inception-B, Inception-C)과, feature map의 size가 절반으로 줄어드는 2개의 Reduction Block(Reduction-A, Reduction-B)으로 구성이 되어있음
- Inception-v2, Inception-v3에서 확장하여 architecture를 좀 더 균일하고 단순화한 버전이 Inception-v4라 할 수 있습니다. 구조의 단순화로 인해 backpropagation 단계에서의 memory가 최적화되는 효과를 얻을 수 있었다고 합니다.

#### 2.10 Inception-ResNet, 2017

- 지금까지 Inception-v2, v3, v4를 설명드렸는데, Inception도 결국 ResNet의 아이디어를 가져오기 시작합니다. Inception-v4과 같은 논문에서 제안된 방법이며 Inception block에 ResNet의 Residual block을 합친 **Inception-ResNet-v1** 과 **Inception-ResNet-v2** 를 제안하였습니다.
- 전자는 Inception-v3과 연산량이 거의 유사한 모델이고, 후자는 Inception-v4와 연산량이 거의 유사하면서 정확도가 더 좋은 모델이라고 정리할 수 있습니다.
- Inception-ResNet-v2의 전반적인 틀은 Inception-ResNet-v1과 거의 유사하고 각 block의 filter 개수가 늘어나는 정도의 차이만 있습니다. 또한 Stem block은 Inception-v4에서 사용한 Stem block을 사용하였습니다. 다만 Inception-ResNet이 Inception-v3에 비해 학습이 빨리 수렴하는 효과를 얻을 수 있고, Inception-v4보다 높은 정확도를 얻을 수 있다고 합니다. 



#### 2.11 Stochastic Depth ResNet, 2016

-  2016년 ECCV에 발표된 “Deep Networks with Stochastic Depth”라는 논문이며, vanishing gradient로 인해 학습이 느리게 되는 문제를 완화시키고자 **stochastic depth** 라는 randomness에 기반한 학습 방법을 제안합니다. 
- 이 방법은 2019년 말 ImageNet 데이터셋에 대해 State-of-the-art 성능을 달성한 [**“Self-training with Noisy Student improves ImageNet classification”** ](https://hoya012.github.io/blog/Self-training-with-Noisy-Student-improves-ImageNet-classification-Review/)논문 리뷰 글에서 noise 기법으로 사용된 기법입니다. 사실 이 논문은 새로운 architecture를 제안했다고 보기는 어렵습니다. 기존 ResNet에 새로운 학습 방법을 추가했다고 보는게 맞지만, ResNet의 layer 개수를 overfitting 없이 크게 늘릴 수 있는 방법을 제안하였다는 점에서 오늘 소개를 드리고 싶습니다.
- 비슷한 아이디어로는 여러분들이 잘 아시는 Dropout이 있습니다. Dropout은 network의 hidden unit을 일정 확률로 0으로 만드는 regularization 기법이며, 후속 연구론 아예 connection(weight)을 끊어버리는 DropConnect(2013 ICML) 기법, MaxOut(2013 ICML), MaxDrop(2016 ACCV) 등의 후속 연구가 존재합니다. 위의 방법들은 weight나 hidden unit(feature map)에 집중했다면, Stochastic depth란 network의 depth를 학습 단계에 random하게 줄이는 것을 의미합니다.
- ResNet으로 치면 확률적으로 일정 block을 inactive하게 만들어서, 해당 block은 shortcut만 수행하는, 즉 input과 output이 같아져서 아무런 연산도 수행하지 않는 block으로 처리하여 network의 depth를 조절하는 것입니다. 이 방법은 학습시에만 사용하고 test 시에는 모든 block을 active하게 만든 full-length network를 사용합니다.
- Stochastic Depth ResNet은 CIFAR-10, SVHN 등에 대해선 test error가 줄어드는 효과가 있지만, ImageNet과 같이 복잡하고 큰 데이터 셋에서는 별다른 효과를 보지 못했습니다. 다만 CIFAR-10과 같이 비교적 작은 데이터셋에서는 ResNet을 1202 layer를 쌓았을 때 기존 ResNet은 오히려 정확도가 떨어지는 반면 Stochastic Depth ResNet은 정확도가 향상되는 결과를 보여주고 있습니다.

#### 2.12 Wide ResNet, 2016

- 2016년 BMVC에 발표된 “Wide Residual Networks” 논문입니다. 
- 처음 소개드렸던 Pre-Act ResNet, 방금 소개드린 Stochastic Depth ResNet과 같이 ResNet의 성능을 높이기 위한 여러 실험들을 수행한 뒤, 이를 정리한 논문입니다.
- 나아가 정확도를 높이기 위해 Layer(Depth)만 더 많이 쌓으려고 해왔는데, Conv filter 개수(Width)도 늘리는 시도를 하였고, 여러 실험을 통해 **Wide ResNet** 구조를 제안하였습니다. 마지막으로, BN의 등장 이후 잘 사용되지 않던 dropout을 ResNet의 학습 안정성을 높이기 위해 적용할 수 있음을 보였습니다.
- Conv layer의 filter 개수를 늘리고, dropout을 사용하는 방법이 효과적임을 실험을 통해 보여주고 있습니다. 
- 병렬처리 관점에서 봤을 때, layer의 개수(depth)를 늘리는 것보다 Conv filter 개수(width)를 늘리는 것이 더 효율적이기 때문에 본인들이 제안한 WRN-40-4 구조가 ResNet-1001과 test error는 유사하지만 forward + backward propagation에 소요되는 시간이 8배 빠름을 보여주고 있습니다.
- 이 논문에서는 depth 외에 width도 고려해야 한다는 점이 핵심인데, 2019년에 학계를 뜨겁게 달궜던 [**EfficientNet**](https://hoya012.github.io/blog/EfficientNet-review/) 에서는 한발 더 나아가 width와 depth 뿐만 아니라 input resolution을 동시에 고려하여서 키워주는 **compound scaling** 방법을 통해 굉장히 효율적으로 정확도를 높이는 방법을 제안했습니다. WRN과 같은 연구가 있었기 때문에 EfficientNet도 등장할 수 있지 않았나 생각해봅니다.



#### 2.13 SqueezeNet, 2017

- 2016년에 arXiv에 공개되고, 2017년 ICLR에 아쉽게 reject 되었지만, 많이 사용되는 “SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size” 라는 논문에서 제안한 **SqueezeNet** 입니다. 
- Squeeze라는 단어는 쥐어짜내는 것을 뜻하며 제가 좋아하는 야구에서도 점수를 짜내기 위한 스퀴즈 번트라는 작전이 존재합니다. 이처럼 network를 쥐어 짜내는 것을 의미하며, 제목에서 알 수 있듯이 AlexNet의 parameter를 50배 이상 줄여서 0.5MB 이하의 model size를 가질 수 있는 architecture 구조를 제안하고 있습니다.
- 논문에서는 총 3가지 종류의 SqueezeNet architecture를 제안하고 있으며, 모든 architecture는 그림 왼쪽에 있는 **Fire module** 로 구성이 되어있습니다. Fire module은 1x1 convolution으로 filter 개수를 줄인 뒤(squeeze) 1x1 conv와 3x3 conv를 통해 filter 개수를 늘려주는(expand) 연산을 수행합니다. 3개의 conv layer의 filter 개수는 hyper parameter이며 자세한 구조는 다음과 같습니다.
- NIN, GoogLeNet 등에서 사용했던 것처럼 FC layer 대신 GAP를 이용하였고, 실험에는 architecture 구조를 제안한 것에 추가로 pruning 기법과 compression 기법(Deep Compression) 등을 같이 적용하여 최종적으로 AlexNet 대비 ImageNet Accuracy는 비슷하거나 약간 더 높은 수치를 얻으면서 Model Size는 적게는 50배에서 많게는 510배(6bit compression)까지 줄일 수 있음을 보이고 있습니다. 추가로 첫번째 그림에서 보여드렸던 3가지 구조 중 ImageNet에 대한 정확도는 **Simple Bypass SqueezeNet > Complex Bypass SqueezeNet > SqueezeNet** 순으로 좋았다고 합니다.
- Pruning, Compression 등 모델 경량화 기법들을 많이 사용하였지만 architecture 관점에서도 연산량을 줄이기 위한 시도를 논문에서 보여주고 있습니다. 다만, fire module의 아이디어는 이미 지난 번 소개 드린 Inception v2의 Conv Filter Factorization과 비슷한 방법이고, Inception v2에 비해 정확도가 많이 낮아서 좋은 평가를 받지 못한 것으로 생각됩니다.



#### 2.14 Xception, 2017

-  2017 CVPR에 “Xception: Deep Learning with Depthwise Separable Convolutions”라는 제목으로 발표된 **Xception** 입니다. 
- 본 논문은 Inception 구조에 대한 고찰로 연구를 시작하였으며, 추후 많은 연구들에서 사용이 되는 연산인 **depthwise-separable convolution** 을 제안하고 있습니다
- Inception v1, 즉 GoogLeNet에서는 여러 갈래로 연산을 쪼갠 뒤 합치는 방식을 이용함으로써 **cross-channel correlation**과 **spatial correlation**을 적절히 분리할 수 있다고 주장을 하였습니다. 쉽게 설명하자면, channel간의 상관 관계와 image의 지역적인 상관 관계를 분리해서 학습하도록 가이드를 주는 Inception module을 제안한 셈이죠.



#### 2.15 MobileNet, 2017

- 2017년 4월 arXiv에 “MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications”라는 제목으로 발표된 **MobileNet** 입니다.
- MobileNet도 핵심은 Depthwise-Separable Convolution 연산을 적절히 사용하는 것이며, 이는 직전에 소개드린 Xception 논문에서 제안한 아이디어입니다. 약간의 차이가 있다면, architecture 구조를 새롭게 제안을 하였고, Depthwise Convolution과 Point Convolution 사이에 BN과 ReLU를 넣은 점이 차이점입니다.
- 또한 Xception은 Inception보다 높은 정확도를 내기 위해 Depthwise-Separable Convolution을 적절히 사용하는 데 집중한 반면, MobileNet은 Depthwise-Separable Convolution을 통해 Mobile Device에서 돌아갈 수 있을 만큼 경량 architecture 구조를 제안하는데 집중을 했다는 점에서도 차이가 있습니다. 즉, 같은 연산을 사용하였지만 바라보고 있는 곳이 다른 셈이죠.



#### 2.16 ResNet, 2017

- 2016년 말 arXiv에 공개되고 2017년 CVPR에 “Aggregated Residual Transformations for Deep Neural Networks” 라는 이름으로 발표된 **ResNext** 입니다. 이름에서 유추하실 수 있듯이 ResNet을 기반으로 새로운 구조를 제안한 논문입니다. 2016년 ILSVRC 대회에서 2등을 하였으며 2015년 우승팀의 ResNet보다 높은 정확도를 달성하였습니다.
- 2016년 ILSVRC 대회의 1등은 Trimps-Soushen 팀이며 이 팀은 Inception-v3, Inception-v3, Inception-v4, Inception-ResNet-v2, ResNet-200, WRN-68-3 5가지 model을 적절히 앙상블하여 1위를 달성하였다고 합니다. 그동안 ILSVRC 대회에서는 단일 모델로 참가하는 경우가 많았는데 2016년에는 잘 알려진 모델들을 앙상블한 팀이 1위를 했다는 점이 특징이며, 논문을 위한 접근이 아니라 Competition을 위한 접근을 한 방법이라고 할 수 있습니다.
- 기존 ResNet은 Res Block의 반복 구조로 이루어져 있고, 지난 2편에서 소개드렸던 여러 ResNet의 변형들도 ResNet의 width(filter 개수)와 depth(layer 개수)를 조절하는 시도를 하였는데, 본 논문에서는 width와 depth 외에 **cardinality** 라는 새로운 차원의 개념을 도입합니다.
- Cardinality는 한글로 번역하면 집합의 크기 또는 집합의 원소 개수를 의미하는데, CNN에서는 하나의 block 안의 transformation 개수 혹은 path, branch의 개수 혹은 group의 개수 정도로 정의할 수 있습니다. 위의 그림에서는 64개의 filter 개수를 32개의 path로 쪼개서 각각 path마다 4개씩 filter를 사용하는 것을 보여주고 있으며, 이는 AlexNet에서 사용했던 **Grouped Convolution** 과 유사한 방식입니다. 사실 AlexNet은 GPU Memory의 부족으로 눈물을 머금고 Grouped Convolution을 이용하였는데 ResNext에서는 이러한 연산이 정확도에도 좋은 영향을 줄 수 있음을 거의 최초로 밝힌 논문입니다.



#### 2.17 PolyNet, 2017

- 2016년 arXiv에 공개되고 2017년 CVPR에 “PolyNet: A Pursuit of Structural Diversity in Very Deep Networks” 란 이름으로 발표된 논문이며, **PolyNet** 이라는 architecture를 제안하였습니다. 참고로 이 방법론은 2016년 ILSVRC 대회에서 0.01%라는 근소한 차이로 ResNext에게 밀려서 3위를 하였습니다. (Top-5 Error Rate: 3.031% vs 3.042%)
- 이제 본론으로 들어가서, Inception, ResNet 등 좋은 성능을 내는 구조들이 제안이 되었는데, 한가지 문제점이 network를 굉장히 깊게 쌓으면 정확도는 미미하게 향상되거나 오히려 떨어지고, 학습 난이도만 높이는 부작용이 있습니다. 본 논문에서는 이러한 어려움을 해결하기 위한 **PolyInception module** 을 제안하였으며, 이러한 구조를 사용하였을 때, Inception-ResNet-v2보다 Convolution Layer를 많이 쌓을수록 더 높은 정확도를 달성하는 결과를 보였습니다.



#### 2.18 PyramidNet, 2017

- 2017년 CVPR에서 발표된 “Deep Pyramidal Residual Networks” 논문의 **PyramidNet** 입니다. ResNet을 기반으로 성능을 개선한 논문이며 제목에서 알 수 있듯이 Pyramid의 모양에서 이름을 따왔습니다.
-  ResNet은 feature map의 가로, 세로 size가 같은 layer에서는 feature map의 개수가 동일한 구조를 가지고 있고, down sampling이 되는 순간 feature map의 개수가 2배씩 증가하는 구조로 설계가 되었습니다. 이 **down sampling + doubled feature map layer** 가 전체 성능에 관여하는 비중이 크다는 실험 결과가 있었는데, 이에서 착안해서 down sampling layer에서만 이뤄지던 feature map의 개수를 늘리는 과정을 전체 layer에 녹여내는 구조를 제안하였습니다. 

- Feature map의 개수를 늘려주는 방법은 linear하게 늘려주는 방법과 exponential하게 늘려주는 방법을 생각할 수 있습니다. 위의 그림의 (a)와 같이 linear하게 늘려주는 방법을 additive라 부르고, (b)와 같이 exponential하게 늘려주는 방법을 multiplicative라 부르는데 본 논문에선 성능이 더 좋았던 additive 방법을 사용하였습니다. 위의 그림 (c)이 Additive와 multiplicative의 차이를 보여주며, additive 방식이 input과 가까운 초기의 layer의 feature map 개수가 더 많아서 성능이 더 좋다고 설명하고 있습니다.
- Pre-Act ResNet에서 제안한 Res Block(BN-ReLU-Conv) 대신 약간 다른 Res Block(BN-Conv-BN-ReLU-Conv-BN)을 이용하여 약간의 성능 향상을 얻었습니다.
- Pyramidal Block을 이용하면 input과 output의 feature map의 개수가 달라지기 때문에 기존의 identity mapping을 Shortcut connection으로 이용할 수 없는데, 여러 실험을 통해 zero padding을 이용한 Identity mapping with zero-padded shortcut이 가장 효율적임을 보여주고 있습니다.



#### 2.19 Residual Attention Network, 2017

- 2017년 CVPR에 “Residual Attention Network for Image Classification” 라는 제목으로 발표된 **Residual Attention Network** 입니다. 제목에서 알 수 있듯이 ResNet에 Attention 아이디어를 접목시킨 논문입니다. Attention 아이디어는 자연어 처리에서 굉장히 잘 사용이 되어왔으며, 이를 Computer Vision 문제에 접목시켰습니다.
- Attention을 적용하기 전에는 feature map이 분류하고자 하는 물체의 영역에 집중하지 못하는 경향이 있는데, attention을 적용하면 feature map을 시각화 했을 때 물체의 영역에 잘 집중을 하고 있는 것을 확인할 수 있습니다.
- Residual Attention Network의 architecture는 위의 그림과 같이 Attention을 뽑아내는 Soft Mask Branch와 일반적인 Conv 연산이 수행되는 Trunk Branch로 구성이 되어있으며, Soft Mask Branch에서는 receptive field를 늘려 주기 위해 down sampling과 up sampling을 수행하는 식으로 구성이 되어있습니다.
-  Attention을 주는 방식에도 Spatial Attention과 Channel Attention이 있는데, 실험 결과 두가지 방법을 섞은 Mixed Attention이 가장 성능이 좋았다고 합니다.
- 본 논문에서는 다른 task에 대한 실험은 수행하지 않았지만 비단 Classification 뿐만 아니라 Object Detection, Segmentation 등 다른 Computer Vision task에 대해서도 적용할 수 있는 아이디어라 생각됩니다.



#### 2.20 DenseNet, 2017

- 2017 CVPR의 Best Paper인 “Densely Connected Convolutional Networks”의 **DenseNet** 입니다. ResNet의 shortcut 방식이랑 비슷한 아이디어인데, ResNet은 feature map끼리 더하기를 이용하였다면, DenseNet은 feature map끼리의 Concatenation을 이용하였다는 것이 가장 큰 차이이며 핵심 내용입니다.



#### 2.21 Dual Path Network(DPN), 2017

- 2017년 NIPS (지금은 NeurIPS)에 “Dual Path Networks” 란 이름으로 발표된 **Dual Path Networks(이하 DPN)** 입니다. 2017년 ILSVRC Object Localization task에서 1위를 차지하였고, Classification task에서는 3위를 차지한 방법론입니다. 
- Dual Path Network는 2개의 path를 가지는 network라는 뜻으로, ResNet과 DenseNet의 각각의 아이디어에서 장점만 가져오는 데 집중하였으며, 논문에서는 ResNet과 DenseNet이 Higher Order Recurrent Neural Network(HORNN)의 형태로 표현이 가능하다는 점에서 착안해서 각각 Network의 성공 요인을 분석하고 있습니다. 
-  ResNet은 **Feature refinement** 혹은 **Feature re-use** 효과를 얻게 해주고 DenseNet은 **Feature re-exploration** 효과를 얻게 해준다고 합니다. 저는 HORNN을 잘 몰라서 이 부분이 잘 와 닿지 않았는데, 혹시 더 깊이 알고 싶으신 분은 2016년 발표된 [**“Higher Order Recurrent Neural Networks”** ](https://arxiv.org/pdf/1605.00064.pdf)논문을 확인하시기 바랍니다. 중요한 건 ResNet과 DenseNet의 장점만을 가져온다는 점입니다!

- 실제로 architecture도 기존 ResNet에서의 add 기반의 shortcut 방식과, DenseNet에서의 concatenate 해주는 shortcut 방식을 적절하게 섞어서 사용하고 있으며, ResNext에서의 Grouped Convolution layer 또한 사용하고 있습니다. 논문에서는 DPN-92와 DPN-98을 DenseNet-161, RexNext-101과 비교하고 있으며, 적절히 효율적인 parameter 수와 FLOPs를 보이고 있습니다.
- DPN은 “ResNet과 DenseNet의 장점들을 잘 합쳐서 높은 정확도, 작은 모델 크기, 적은 계산량, 적은 GPU memory 점유량 등을 얻을 수 있으며, Object Detection, Semantic Segmentation task에서도 좋은 성능을 보인다!” 정도로 요약할 수 있을 것 같습니다.



#### 2.22 Squeeze-and-Excitation Network (SENet), 2018

- 2018년 CVPR에서 발표된 “Squeeze-and-Excitation Networks” 라는 논문이며 **SENet** 이라는 이름으로 불립니다. 
- **Squeeze 연산** 에서는 feature map을 spatial dimension (H x W)을 따라서 합쳐주는 과정이 진행되며 이 때 저희가 지난 포스팅들을 통해서 봤었던 Global Average Pooling이 사용됩니다.
- **Excitation 연산** 이 뒤따르며, input vector를 같은 모양을 갖는 output vector로 embedding 시키는 간단한 self-gating 메커니즘을 통해 구현이 됩니다. 즉 여기선 channel 마다 weights를 주는 느낌을 받을 수 있으며 이를 channel 방향으로 attention을 준다고도 표현을 합니다. 그림을 보시면 excitation 연산을 통해 vector에 색이 입혀진 것을 확인할 수 있으며, 이렇게 색이 매겨진 vector와 input feature map U를 element-wise로 곱하면 output feature map이 생성되며, output feature map에 각 channel마다 아까 구한 weight(색)들이 반영이 되어있는 것을 확인할 수 있습니다.
- Squeeze-Excitation block의 가장 큰 장점으로는 이미 존재하는 CNN architecture에 붙여서 사용할 수 있다는 점입니다. 즉, ResNet 등 잘 알려진 network 구조에서 squeeze-excitation block을 추가하면 SE-ResNet이 되는 셈이죠. 위의 그림과 같이 Inception module, ResNet module에도 부착이 가능하며 실제로 저자들이 ResNet, ResNext, Inception 등에 SE block을 추가한 결과 미미한 수치의 연산량은 증가하였지만 정확도가 많이 향상되는 결과를 제시하였습니다.
- 이 외에도 ablation study를 여러 개 수행하였는데, 간단히 요약을 드리겠습니다. Inception과 ResNet에 SE block을 추가한 위의 그림을 자세히 보시면 FC layer를 거쳐서 channel이 C에서 C/r로 줄어들었다가 다시 C로 돌아오는 형태를 확인하실 수 있습니다. 이 r을 reduction ratio라 하는데, 여러 reduction ratio에 대해 실험을 한 결과 대부분 비슷했으나 정확도와 complexity를 동시에 고려하면 16이 최적임을 실험적으로 밝혔습니다. 또한 Squeeze 연산에서 Max와 Average 연산 중 Average 연산이 더 효과적이고, Excitation operation에서는 ReLU, Tanh보다 Sigmoid를 쓰는 것이 효과적임을 실험적으로 보이고 있습니다.



#### 2.23 ShuffleNet, 2018

- 2018년 CVPR에 “ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile” 라는 제목으로 발표된 **ShuffleNet** 입니다. 
- ShuffleNet은 지난 포스팅에서 소개 드렸던 MobileNet과 같이 경량화된 CNN architecture를 제안하였으며 AlexNet과 ResNext에서 봐서 이제는 친숙할 법한 **Group Convolution** 과 본인들이 제안한 **Channel Shuffle** 이 ShuffleNet의 핵심입니다. Depthwise Separable Convolution 연산이 제안된 이후 경량화된 CNN에는 필수로 사용이 되고 있었는데, 예전에는 연산량을 줄이기 위해 제안되었던 1x1 convolution 연산이 이제는 전체 연산 중에 많은 비율을 차지하게 되었습니다. 이 점에서 출발해서 1x1 convolution 연산에 Group Convolution을 적용하여 MobileNet보다 더 효율적인 구조를 제안했습니다. MobileNet과 ShuffleNet의 이러한 경쟁 구도가 선순환으로 이어지게 됩니다. MobileNet V2가 ShuffleNet에 이어서 공개가 되었으며 자세한 설명은 뒤에서 드리도록 하겠습니다.



#### 2.24 CondenseNet, 2018

- 2018년 CVPR에 “CondenseNet: An Efficient DenseNet using Learned Group Convolutions” 라는 제목으로 발표된 **CondenseNet** 입니다.
- 이 논문도 MobileNet, ShuffleNet과 같이 Mobile Device 등 computational resource가 제한된 기기에서 CNN을 돌리기 위해 경량화 된 architecture를 제안한 논문이며, 제목에서 알 수 있듯이 Learned Group Convolution이라는 방법을 DenseNet에 접목시키는 방법을 제안하고 있습니다. 여기에 Network Pruning의 아이디어도 접목이 됩니다. Network Pruning은 layer 간의 중요하지 않은 연결을 제거하는 방식이며, 이러한 아이디어가 CondenseNet에 들어가 있습니다.
- CondenseNet은 MobileNet, ShuffleNet, NASNet 등과 비교하였을 때 더 적은 연산량과 parameter를 가지고 더 높은 정확도를 보이고 있음을 확인할 수 있으며, 실제 ARM processor에서 inference time을 측정하였을 때도 MobileNet보다 거의 2배 빠른 처리 속도를 보였습니다. 이 외에도 여러 ablation study가 수행이 되었는데 관심있으신 분들은 논문을 참고하시기 바랍니다.



#### 2.25 MobileNetV2, 2018

- 이 논문도 2018년 CVPR에 “MobileNetV2: Inverted Residuals and Linear Bottlenecks” 제목으로 발표가 되었으며 지난 포스팅에서 소개 드렸던 MobileNet의 두번째 버전이다.

-  MobileNetV1의 핵심이었던 Depthwise Separable Convolution이 역시 MobileNetV2에도 사용이 되며, **Linear Bottlenecks** 아이디어와 **Inverted Residual** 아이디어가 MobileNetV2의 가장 큰 변화입니다.



#### 2.26 ShuffleNetV2, 2018

- ShuffleNet의 2번째 버전인 **ShuffleNetV2** 이며 ECCV 2018에 “ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design”라는 제목으로 발표가 되었습니다.
- 논문의 제목에 있는 Practical Guideline에서 알 수 있듯이, 실제로 사용하는 입장에서 network architecture를 설계해야 함을 강조하고 있습니다. FLOPs(Floating Point Operations Per Second)는 기존 연구 들에서 주로 목표로 삼아오던 지표인데 이는 실제로 mobile device에서 CNN을 사용하는데 있어서 가장 중요한 지표가 아닙니다. 본 논문에서는 실제 동작 시간인 takt time 혹은 throughput 지표, memory access cost 지표 등 실제로 중요한 지표를 고려하여 CNN architecture를 설계하는 방법을 제안합니다. 위의 그림에 논문에서 제안한 4가지 가이드라인이 정리가 되어있습니다.



#### 2.27 NASNet, 2018

- 다음은 2018년 CVPR에 “Learning Transferable Architectures for Scalable Image Recognition” 라는 제목으로 발표된 **NASNet** 입니다. 이 논문은 AutoML을 이용하여 architecture를 찾는 Neural Architecture Search(NAS)를 통해 ImageNet과 같은 비교적 규모가 큰 데이터셋에 적용시키기 위한 방법을 제안하였습니다.
- 2017년 ICLR에 발표된 NAS의 시초격인 [**“Neural Architecture Search with reinforcement learning”** ](https://arxiv.org/pdf/1611.01578.pdf)논문은 CIFAR-10과 같은 작은 데이터셋에 대한 최적의 architecture를 찾기 위해 800대의 GPU로 약 한달정도 search를 해야하기 때문에, ImageNet에 대해선 거의 수백년이 걸린다는 한계가 있습니다. NASNet은 이를 극복하기 위해 작은 데이터셋에서 찾은 연산들의 집합인 Convolution Cell을 큰 데이터셋에 적절하게 재활용을 하는 방법을 제안하였고, 그 결과 ImageNet 데이터셋에 대해서 기존 SOTA 논문이었던 SENet 보다 더 적은 파라미터수와 연산량으로 동등한 정확도를 달성하는 결과를 보여주고있습니다.



#### 2.28 AmoebaNet, 2018

- NAS 논문이며, 2018년 2월 arXiv에 공개되고 2019년 AAAI에 “Regularized Evolution for Image Classifier Architecture Search” 제목으로 발표된 **AmoebaNet** 입니다.
- 기존 NAS, NASNet 등의 연구에서는 강화학습을 기반으로 architecture search를 수행하였는데, AmoebaNet에서는 Evolutionary Algorithms(진화 알고리즘)을 기반으로 architecture를 찾는 방법을 제안하였습니다. 다만 architecture search에서 중요한 역할을 하는 search space는 NASNet의 search space를 그대로 사용하였고, 실제로 두 방식의 최종 test accuracy도 거의 비슷하게 측정이 됩니다. 위의 그림과 같은 search space와 aging 기반 토너먼트 selection 진화 알고리즘을 바탕으로 architecture search를 수행합니다.
- 진화 알고리즘에서 다양한 가능성을 만들어주는 돌연변이(mutation) 생성 방법에는 2가지 방법이 있는데 각 연산의 output을 연결해주는 곳(Hidden state)에 mutation을 가하는 방법과, 각 연산자를 바꾸는 방법이 사용됩니다.
- 진화 알고리즘을 바탕으로 찾은 AmoebaNet-A의 architecture는 다음과 같으며, NASNet과 마찬가지로 Normal Cell과 Reduction Cell을 여러 번 반복하는 구조로 되어있습니다.
- AmoebaNet의 architecture의 크기를 결정하는 요소는 stack 당 normal cell을 쌓는 개수인 N과 convolution 연산의 output filter의 개수인 F에 의해 결정되며, F과 F를 키워주면 parameter 개수, 연산량이 증가하지만 정확도도 증가하는 경향을 보이며, ImageNet 데이터셋에 대해서 F를 448로 크게 키워주는 경우 기존 NASNet, 바로 다음 설명드릴 PNASNet 보다 약간 더 높은 정확도를 달성하는 결과를 보여주고 있습니다. 다만 NASNet, PNASNet 자체도 성능이 꽤 높은 편이라 AmoebaNet이 압도적인 성능을 보여주진 못합니다. 그래도 진화 알고리즘을 architecture search에 접목시켰다는 점에서 많이 인용되는 논문인 것 같습니다.



#### 2.29 PNASNet, 2018

- 2018년 ECCV에 “Progressive Neural Architecture Search” 제목으로 발표된 **PNASNet** 입니다. NAS에 “점진적인” 이라는 뜻을 갖는 Progressive를 붙인 것에서 알 수 있듯이, Cell 구조에 들어가는 연산들을 하나하나 추가하는 과정을 통해 Cell을 형성하며, 이 때 Sequential Model-Based Optimization(SMBO) 방식이 사용됩니다. 
- Search space는 NASNet에서 사용한 것을 거의 참고하였으며, NASNet의 search space에 있던 13개의 연산 중 거의 사용되지 않은 5개의 연산을 제거하고 8개의 연산만 사용하였습니다.
- 큰 kernel을 갖는 max pooling과 1x3 + 3x1 convolution도 제거가 된 점은 저는 수긍이 갔지만, 사람이 자주 사용하던 연산인 1x1 convolution과 3x3 convolution이 제외된 점은 다소 놀라웠습니다. 마치 사람이 제안한 방법이 인공지능에게 선택 받지 못한 느낌이 들었네요.. 아니면 3x3 convolution 대신 3x3 dilated convolution, depthwise-separable convolution이 더 효과적이기 때문에 선택받지 못한 것 같기도 합니다.

#### 2.30 MnasNet, 2018

- 2018년 arXiv에 공개되고 2019년 CVPR에 “MnasNet: Platform-Aware Neural Architecture Search for Mobile” 제목으로 발표된 **MnasNet** 입니다. 
- 이전 NAS 논문들은 정확도에 초점을 두어서 굉장히 오랜 시간 GPU를 태워가면서 거대한 CNN architecture를 찾는데 집중했다면, 이 논문에서는 Mobile Device에서 돌릴 수 있는 가벼운 CNN architecture를 NAS를 통해 찾는 것을 목표로 연구를 시작합니다.
- NASNet, PNASNet 등에서는 Cell을 구성하는 여러 연산들을 옵션으로 두고 Cell을 찾은 뒤 그 Cell을 반복해서 쌓는 구조를 사용하였다면, MnasNet에서는 각 Cell 마다 다른 구조를 갖지만 하나의 Cell에는 같은 연산을 반복시키는 구조로 search space를 가져갑니다. 이렇게 search space를 가져감으로써 다양성과 search space의 크기의 적절한 균형을 얻을 수 있었다고 합니다.



> 위에서 NAS, NASNet, AmoebaNet, PNASNet, MnasNet 5가지 방법에 대해 소개를 드렸는데, 이 방법들 모두 사람이 만든 CNN architecture보다 더 높은 정확도를 달성하는 데 성공하였지만, search에 수백, 수천 gpu days가 소요되는 점은 여전히 풀어야할 숙제였습니다. 
>
> 이 논문들 이후에는 search에 소요되는 시간을 줄이려는 연구들이 많이 제안이 되었고, 대표적인 방법이 RL 기반의 [**Efficient Neural Architecture Search via Parameter Sharing(ENAS)**](https://arxiv.org/pdf/1802.03268.pdf) 와, Gradient Descent 기반의 [**DARTS: Differentiable Architecture Search (DARTS)**](https://arxiv.org/pdf/1806.09055.pdf) 등이 있으며, 해당 논문들 덕분에 거의 single gpu로 하루만에 search가 가능한 수준까지 빠르게 올라오게 됩니다. 



#### 2.31 RetinaNet

- 기존 ResNet에 FPN 방법을 결합함.

- 지금까지 one-stage Network(YOLO,SSD 등) 의 Dense Object Detection 는 two-stage Network(R-CNN 계열) 에 비해 속도는 빠르지만 성능은 낮았다.

  우리는 그 이유가 극단적인 클래스 불균형 문제 때문이라는 것을 발견했다.

  이 문제를 해결하기 위해서 클래스 분류에 일반적으로 사용되는 크로스 엔트로피 로스 함수를 조금 수정한 Focal Loss 를 제안한다.





## Result







## 생소한 영어 단어

1. canonical / 표준의
2. recursively / 재귀적으로
3. analogous / 유사한
4. intuitively / 직관적으로
5. prioritise / 우선순위를 정하다.
6. steric / 입체의
7. executable / 실행 가능한
8. constitute / 구성되다.
9. tactically / 요령있게, 전략적으로
10. comprise / 구성되다.
11. encompass / 포함하다, 에워싸다.
12. associate / 연상하다, 결부짓다
13. metrics / 측정기준
14. marginally / 아주 조금, 미미하게
15. drastically / 과감히
16. exhaustively / 철저하게
17. owing to / ~ 때문에
18. particular / 특정의
19. hypothetical / 가상의, 가설의
20. respective / 각각의

