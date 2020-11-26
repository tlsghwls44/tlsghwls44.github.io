---
published: true
layout: post
title: "[Nature] Planning chemical syntheses with deep neural networks and symbolic AI"
subtitle: Nature
categories: review
tags: plant_engineering
comments: true
---

# 1번째 논문 리뷰

- 목차
  1. [Paper info.](#paper-info)
  2. [Introduction](#introduction)
  3. [Problem definition](#problem-definition)
  4. [Background](#background)
  5. Methodology
  6. Result
  7. Contribution
  
- [생소한 영어 단어](#생소한-영어-단어)

## Paper info

​	Journal : Nature

​	Title : Planning chemical sytheses with deep neural networks and symbolic AI

​	Author : Marwin H.S. Segler, Mark P. Waller 등

​	Published date : 2018/03



## Introduction

​	기존의 Computer-aided retrosynthesis는 <u>①느리고</u>, <u>②성능이 만족스럽지 못하다</u>.

​	본 논문에서는 retrosynthetic route를 발견하기 위해 **Monte Carlo tree search** 방법과 **symbolic AI** 기술을 사용한다.

​	**Monte Carlo tree search**에 search를 가이드하기 위해 **expansion policy network**와 유망한 역합성 단계를 선택하기 위해 **filter network**를 결합했다.

​	DNN 신경망은 유기화학 분야에서 출판된 모든 반응에 대해 학습을 수행했다.

​	우리의 시스템은 기존의 Computer-aided search method(추출된 규칙과 hand-designed heuristics를 기반으로 하는)보다 <u>2배 많은 분자</u>들과 <u>3배 더 빠르게</u> 문	제를 해결하였다.



## Problem definition

​	역합성은 분자를 더 간단한 precursor(전구체)로 변환하는 방법이다. 이 때 전구체는 우리가 알고있는 set이거나 상업적으로 구매가능한 building-block 분자들	을 말한다.

​	변환은 유사한 시작 물질을 가지고 성공적으로 수행된 일련의 비슷한 반응들로부터 유래된다.

​	패턴 인식 단계에서 화학자들은 직관적으로 그들이 생각했을 때 덜 유망한 케이스를 제외한 가장 유망한 변환 메카니즘에 대해 우선순위를 매긴다.

​	하지만 변환이 새로운 분자에 적용되었을 때, 해당하는 반응이 기대한 방향으로 진행될 것이라는 보장이 없다.

​	예측한 대로 반응에 실패하는 분자는 **'out of scope'**이라고 부른다.

​	이것은 반응 메커니즘에 대해 불완전한 이해나 분자 문맥에서 대립되는 반응성으로 인한 입체적 효과나 전기적 효과 때문일 수도 있다.

​	'In scope'에 있는 분자를 예측하는 것은 가장 위대한 화학자들에게도 challenging한 일일 수 있다.



​	Computer-assisted synthesis planning(CASP)는 화학자들이 더 빠르고 나은 route를 찾는데 도움을 준다.

​	CASP를 수행하기 위해서는 human knowledge가 실행 가능한 프로그램에 전달되어야 한다.

​	60년의 연구에도 불구하고 manual하게 chemistry를 formalize하는 일은 너무 복잡하고 어려운 일이다.



​	우리는 DNN이 추출된 symbolic 변환을 랭크매기고, 반응성 충돌을 피하며 전문가의 직관적인 의사판단을 흉내낼 수 있는 게 학습을 할 수 있다는 것을 보였다.

​	유망한 방향으로 search를 위해 hand-designed heuristic function이 position value을 결정하는 **heuristic best first search(BFS)**가 적용되었다.

​	불행하게 체스와 달리 화학에서 strong heuristics를 규정하기는 다음의 3가지로 이유로 어렵다.

​			① 화학자들은 좋은 position을 구성하는 것에 동의하지 않는 경향이 있다.

​			② 비록 분자를 간소화하는 것이 일반적으로 권장되나 protecting이나 directing group을 사용하여 복잡성을 일시적으로 증가시키는 것이 전략적으로 유리				할 수 있다.

​			③ position value는 적합한 선구체의 이용가능성에 높게 의존하고 있다. 

​				복잡한 분자도 만약 선구체가 가능하다면 몇 개의 step만으로 만들 수 있다.



​	Monte Carlo tree search(MCTS)는 강한 heuristics가 없이 큰 branching factor를 갖는 sequential decision 문제에 대한 일반적인 serach 기술로 등장했다.

​	MCTS는 position value를 결정하기 위해 `rollouts`를 사용한다. Rollout은 solution이 찾아지거나, maximum depth에 도달하기 전까지 branching 없이 			random search step가 수행되는 Monte Carlo simulation이다.

​	이 random step은 machine-learned policies p(t|s)로부터 샘플링 된다.

​	이 policy는 position s에서 move t를 취하는 확률을 예측하고, 게임에서 승리하는 move를 예측하도록 학습된다.



​	본 연구에서는 화학 합성 계획을 수행하기 위해 3가지의 DNN을 MCTS와 조합한다.

​	1N(the expansion policy)는 자동으로 추출된 변환의 숫자를 제한하면서 유망한 방향으로 serach를 진행한다.

​	2N은 제안된 반응이 실제 feasible한지를 예측한다.

​	3N은 position value를 예측하기 위해 변환이 rollout phase 동안 3N으로부터 샘플링 된다. 

​	신경망은 유기화학 분야에서 출판된 모든 반응에 대해 학습된다.



## Background

heuristic best first search(BFS)

Monte Carlo tree search(MCTS)

강화학습

공부할만한 사이트 : https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html

Policy : Agent가 행동ㅇㄹ 선택하는데 사용하는 규칙을 policy(정책)이라고 칭한다.

Rollout : Rollout이란 initial state부터 terminal state까지 에이전트가 거친 (상태, 행동, 보상)의 sequence를 의미함. 자세한 내용은 강화학습 분야 학습 필요.

Expansion policy랑 rollout policy가 뭐야..?







## Methodology

+ 순서
  + [Training the expansion and rollout policies](#training-the-expansion-and-rollout-policies)
  + [Prediction with the in-scope filter network](#prediction-with-the-in-scope-filter-network)
  + [Integrating neural networks and MCTS](integrating-neural-networks-and-MCTS)





### Training the expansion and rollout policies

​	Reaxys chemistry database의 12.4 million개의 단일 단계 반응로부터 2가지의 transformation rule을 추출했다.

​	**Rollout set**은 반응과정에서 변한 원자와 결합(reaction centre)을 포함하는 규칙과 first-degree neighbouring atom들로 구성된다.

​	2015년 전에 출판된 반응들 중 적어도 50번 이상 출현한 규칙들만 보존된다.

​	**Expansion rule**에 대해서는 더 general한 rule 정의가 적용되었다.

​	적어도 3번이상 등장한 규칙들만 보존되었다.

​	2가지 set은 17,134개와 301,671개의 규칙, 2015년 이후 출판된 문헌의 52%, 79%의 화학반응을 포함한다.

​	규칙 추출은 각 반응, 변환 규칙을 갖는 각 product와 연결된다. 이것은 우리에게 신경망을 product에 대해 가장 좋은 transformation과 다시 말해, product를 	만드는데 가장 좋은 반응을 예측하도록 policy로서 학습을 하도록 한다.

​	중요한 점은 그러한 신경망들은 반응이 일어날 수 있는(functional group tolerance) 문맥에 대해서 학습을 한다는 것이다.

​	**Expansion policy**에 대해 우리는 `exponential linear unit nonlinearities`을 갖는 deep highway network를 적용하였다.

​	일반화 능력에 대해 평가하기위해, 우리는 `time-split strategy`를 수행하였다.

> 여기서 exponential linear unit nonlinearities가 무슨 말인지 모르겠다.
>
> 활성화함수는 말 그대로, 출력값을 활성화를 일으키게 할 것이냐를 결정하고, 그 값을 부여하는 함수다.
>
> 활성화 함수를 사용하는 이유는 데이터를 비선형으로 바꾸기 위해서다. 선형시스템을 신경망에 적용시, 망이 깊어지지 않고 1층의 hidden layer로 구현이 가능하다.
>
> 모든 a,b,x,y(a,b는 상수, x,y는 변수) 에 대하여 f(ax+by)=af(x)+bf(y)의 성질을 가졌기 때문에, 망이 아무리 깊어진들 1개의 은닉층으로 구현이 가능하다.
>
> 망이 깊어지지 않는 것이 왜 문제가 될까?
>
> 망이 깊어질 때 여러가지 장점을 갖는데 대표적으로 2가지가 있다.
>
> 1. 매개변수가 줄어든다.
>
>    -> 망이 깊어지면 같은 수준의 정확도의 망을 구현하더라도 매개변수가 더 적게 필요하다.
>
> 2. 필요한 연산의 수가 줄어든다.
>
>    -> CNN에서 필터의 크기를 줄이고, 망을 깊게 만들면 연산 횟수가 줄어들면서도 정확도를 유지하는 결과를 확인할 수 있다.

> **ELU**
>
> ELU는 비교적 가장 최근에 나온 비선형 함수이다.
>
> ![image-20201124160158770](C:\tlsghwls44_git\tlsghwls44.github.io\assets\img\post_img\image-20201124160158770.png)
>
> - ELU의 특징은 다음과 같다.
>   - ReLU의 모든 장점을 포함한다.
>   - "Dying ReLU" 문제를 해결했다.
>   - 출력값이 거의 zero-centered에 가깝다.
>   - 일반적인 ReLU와 달리 exp함수를 계산하는 비용이 발생한다.
>
> 

Training에 대해서 2015년 이전에 출판된 모든 반응들에 대해 사용되었고 반면에 validation and testing에 대해 2015년 이후에 데이터들이 선택되었다.



​	*Extended Data Table 1부분 해석이 안됨...*

​	Search tree expansion에서 possible한 transformation 수를 최대 50으로 제한함.

​	게다가 예측된 행동의 확률을 high-ranked transformation에서부터 시작해서 합산하고, 만약 축적 확률이 0.995에 도달하면 further expansion을 멈춘다. 비록 	50개보다 더 적은 행동이 expanded 되더라도.

​	이것은 시스템이 더 적은 좋은 option이 존재할 때 transformation 하는 경향을 더 높게 하도록 하는 것에 초점을 맞춘다. 



​	하나의 hidden layer를 갖는 신경망인 **Rollout policy network**도 expansion policy와 같은 방식으로 학습된다.

​	이것은 expansion policy보다 더 낮은 coverage를 유지하는 17,134개의 규칙을 갖는 set을 사용한다. 하지만 그것은 예측에 90ms가 소요되는 expansion 			policy와 달리 더 작은 output layer 때문에 10ms 밖에 소요되지 않는다. 



### Prediction with the in-scope filter network

​	Expansion policy로 search space가 가장 유망한 transformations로 좁혀진 이후, 해당 반응이 특정 분자에 실제로 효과가 있을 것인지 예측해 볼 필요가 있다.

​	우리는 policy network에 의해 선택된 transformation에 대한 반응이 실제로 feasible한지 안한지를 예측하기 위해 **DNN을 binary classifier**로 학습했다.

​	Classifier는 성공 반응과 실패 반응에 대해 학습되어야 한다.

​	불행히도 실패 반응들은 거의 기록되지 않고, reaction database에 포함되지 않는다. 하지만 출판된 반응들은 발생하지 않는 반응들에 대한 내재적인 정보를 담	고 있다.

​	예를 들면, 높은 수율을 갖는 반응인 A+B->C에서 우리는 가상의 product인 D,E가 형성되지 않을 것이라고 가정할 수 있다.

​	기록된 반응들의 반응물의 정반응에 대한 반응 규칙을 적용함으로써 잘못된 regio 및 화학적 반응성을 갖는 부정적인 반응이 생성될 수 있다.

​	Expansion policy에 대해서도 같은 규칙 set을 사용했다. 추가적으로 product와 해당 반응의 쌍을 셔플하여 negative examples를 만들었따. 이러한 data 			augmentation 전략을 통해 training set으로 2015년 이전 출판된 반응의 100 million negative reactions과 test set으로 2015년 이후 출판된 반응의 10 		million 반응들을 생성했다.



### Integrating neural networks and MCTS

Exapnsion policy network와 in-scope filter network는 하나의 pipeline으로 조합되었다.

![image-20201124202117921](C:\tlsghwls44_git\tlsghwls44.github.io\assets\img\post_img\image-20201124202117921.png)

Figure 2  Schematic of MCTS methodology

**a, MCTS searches byiterating over four phases.** 

In the selection phase (1), the most urgent node for analysis is chosen on the basis of the current position values.

> Postion value는 어떻게 결정되는거지?

In phase (2) this node may be expanded by processing the molecules of the position A with the expansion procedure (b), which leads to new
positions B and C, which are added to the tree. 

Then, the most promisingnew position is chosen, and a rollout phase (3) is performed by randomly sampling transformations from the rollout policy until all moleculesare solved or a certain depth is exceeded. 

In the update phase (4), theposition values are updated in the current branch to reflect the result of the rollout. 

**b, Expansion procedure.** 

First, the molecule (A) to retroanalyse is converted to a fingerprint and fed into the policy network, which returns a probability distribution over all possible transformations (T1 to Tn).

Then, only the k most probable transformations are applied to molecule A. This yields the reactants necessary to make A, and thus complete reactions R1
to Rk.

For each reaction, the reaction prediction is performed using the in-scope filter, returning a probablity score. Improbable reactions are then
filtered out, which leads to the list of admissible actions and corresponding precursor positions B and C.



​	Position s<sub>i<sub>가 분석될 때, 각 위치의 분자는 policy network로 들어간다. 가장 높은 score를 갖는 transformation이 분자에 적용된다. 이는 가능한 	precursor와 full reaction을 산출해낸다. 이 반응들은 in-scope filter로 들어간다. 이 in-scope filter에서는 positively classified 반응들에 해당하는transformation과 precursor들만 보존된다.

​	그것들은 position s<sub>i<sub>에서 가능한 'legal moves'를 대표한다.

​	Expansion procedure와 rollout policy는 3N-MCTS를 형성하기 위해 MCTS 알고리즘의 각각의 phase로 결합된다.

​	4개의 MCTS phase는 search tree를 형성하기 위해 반복된다.

​			(1) Selection

​			(2) Expansion

​			(3) Rollout

​			(4) Update



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

