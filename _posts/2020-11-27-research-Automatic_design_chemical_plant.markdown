---
layout: post
title: 화학 공정 자동화 설계
subtitle: 화학 공정 자동화 설계
categories: research
tags: automatic design
comments: true
published: true
---
# 개요

1. 문제 정의
2. 이론적 배경
3. 방법론
4. 실험결과
5. 결론

## 1. 문제 정의

기존의 공정 설계 방법

- 계층적 설계
  - 특징

    ① 계층적인 순서로 설계함

  - 한계점

    ① 후단 설계 변수까지 고려 X

- Superstructure 설계

  - 특징

    ① 모든 가능한 공정구조를 고려하려 함

  - 한계점

    ① 계산량이 많고 연산 시간이 많이 소요

    ② Search space를 user가 상정해야 하므로 모든 structure에 대해 고려했는지에 대해 확신하지 못함.

    ​	->우리가 정의한 boundary 안에서만 최적임을 증명할 수 있음.
    
    

위와 같은 한계점을 보완하기 위해서 공정설계에 대한 새로운 접근법을 제시

후단 설계 변수를 고려하면서 경제성, 안전성, 환경성을 고려할 수 있는 방법론.

> MCTS for automatic process design



## 2. 이론적 배경

MCTS(Monte Carlo Tree Search)

MCTS 활용 역합성 경로 자동 선택 연구에서 아이디어 획득.

![image](https://user-images.githubusercontent.com/55575547/101718773-41c0ac80-3ae5-11eb-9ba2-ae49587598f2.png)

- 루트노드에서 출발하여 모든 수 탐색 대신, 가장 가능성이 있는 수를 샘플링하여 확률적 탐색
- 몇 수 앞을 내다보며, 내가 택할 수 있는 행동이 어느 정도의 승률을 불러올지 시뮬레이션
- 4단계 반복 : Selection-Expansion-Rollout-Update
- 가장 점수가 높은 경로 선택



각 Tree는 Node와 Edge로 구성됨.

Node : Target molecule Precursor

Edge : Transformation rule


## 3. 방법론



아래는 가상의 오류가 있는 공정 데이터를 생성하는데 제시하는 방법

1. **GAN 알고리즘**





2. **Training Deep Networks with Synthetic Data: Bridging the Reality Gap by Domain Randomization**

Deep neural network를 학습시키는 방법론에 대한 연구는 과거에도, 최근에도 굉장히 많이 다뤄지고 있음.

이러한 논문들은 주로 MNIST, CIFAR-10, LSUN, COCO, VOC 등 공개된 데이터셋에 대해서 성능을 검증함.

하지만 저희가 실제 현업에서 Deep neural network를 적용하기 위해서는 직접 데이터를 취득하고 가공해야 하는 과정이 필수로 들어가게 됌.



직접 데이터셋을 구축해야 하는 경우, 데이터를 취득하고 Labeling을 하는데 많은 시간과 비용이 드는 문제가 있습니다. 이러한 문제를 해결하기 위한 선행 연구로 graphic simulator를 이용하여 실제 이미지와 비슷하게 생긴 이미지를 생성하는 연구들이 많이 존재합니다. 하지만 이렇게 graphic simulator로 이미지를 생성하는 과정도 simulator를 제작하는 시간과 비용, 인력 등이 필요한 것은 마찬가지이며 이 또한 한계라고 주장하고 있습니다.



본 논문에서는 이러한 점에 주목해서 실제 데이터를 많이 모으기 힘든 상황에서 OpenAI의“Domain Randomization” 이라는 기법을 Object Detection 문제에 적용하여, 저비용으로 대량의 이미지를 합성하여 데이터셋을 만들고, 정확도를 향상시키는 방법에 대해 제안하고 있습니다.



Source item을 바탕으로

Random Background

Random Texture

Random Geometric shape를 통해 Target item을 생성해냄.



## 4. 실험결과





## 5. 결론

