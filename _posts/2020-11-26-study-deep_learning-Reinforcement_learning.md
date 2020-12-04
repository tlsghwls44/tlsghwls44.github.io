---
published: true
layout: post
title: "강화학습 개념"
subtitle: 강화학습
categories: study
tags: deep_learning
comments: true
---

#  Monte Carlo Tree Search(MCTS)

# 강화학습

- 본 작성 문서는 [혁펜하임님의 Youtube 강화학습 강의 1강](https://www.youtube.com/watch?v=cvctS4xWSaU&list=PL_iJu012NOxehE8fdF9me4TLfbdv3ZW8g&index=1)을 매개체로 작성되었음.

### 강화학습 적용 사례

1. AlphaGo, AlphaStar
2. Cart-Pole Reinforcement Learning
   - 테니스채가 스스로 수직으로 서 있을 수 있게 학습 수행.
3. Learning to walk via Deep Reinforcement Learning
   - 로봇이 스스로 걷는 방법을 학습 수행.



### 강화학습의 목표

Action의 Sequential한 정보를 얻어내는 것. 어떤 action이 가장 이득이 될까?

-> Reward를 Maximize 하는 것

![2020-11-26-RL_1](https://user-images.githubusercontent.com/55575547/100296404-730e8800-2fcf-11eb-82af-f7d13e5a4ab2.JPG)



우리가 하고있는 연구에 접목을 하고싶다 했을 때

이것을 주의깊게 생각해봐야 한다.

> 내가 하는 연구가 연속된 action을 찾는 연구인가?
>
> 확실한 목표를 지정해줄 수 있는가?
>
> > 예를 들면 알파고는 한 수, 한 수 둘 때마다 각 바둑돌의 연관된 sequence가 형성되고, `상대를 이겨라`라는 확실한 목표가 있다.
> >
> > 또한 위에서 언급한 Cart-Pole RL같은 경우에도 테니스채가 움직이는 행동반경에 대해 sequence가 형성되고, `수직으로 서있어라`라는 확실한 목표가 있음.



Reinforcement Learning과 Deep learning은 뿌리가 다른 학문임.

최근에 RL에 DL을 접목해서 더 좋은 성능을 나타내고 있음. (2014년도 구글 논문)



### Q-learning

![2020-11-26-RL_2](C:\tlsghwls44_git\tlsghwls44.github.io\assets\img\post_img\2020-11-26-RL_2.jpg)





**Greedy action**

Action을 취하면서 점수를 매기는데 그 점수가 가장 큰 action을 Greedy action이라고 칭한다.

Episode 1에서는 모든 방향이 다 점수가 0이므로 action을 random하게 취한다. (한 번도 목표점에 도달하지 못했기 때문에)

Q값(랭크를 매긴 점수)는 이동을 하면서 가장 큰 값을 가지는 랭크로 update 된다. 

![2020-11-26-RL_3](C:\tlsghwls44_git\tlsghwls44.github.io\assets\img\post_img\2020-11-26-RL_3.jpg)



**Exploration**

더 좋은 경로가 없을까 하고 탐험을 하는 것.

탐험을 수행하기 위해서 `Epsilon-Greedy`라는 방법을 사용한다.

![수식](C:\tlsghwls44_git\tlsghwls44.github.io\assets\img\post_img\CodeCogsEqn (1).gif) : 0~1 사이의 값. 이 입실론값만큼은 random하게 움직이고 나머지는 Greedy하게 움직인다. 

&epsilon; 값이 0이면 너무 Greedy하게 움직여 최적의 경로인지 아닌지 모르고, 입실론값이 1이면 너무 random하게 움직여서 목표값 도달이 어렵다.

![image](https://user-images.githubusercontent.com/55575547/100296781-77877080-2fd0-11eb-92ed-d0638b4661c5.png)

- **Exploration**의 장점

  ① 새로운 path를 찾을 수 있다.

  ② 새로운 reward가 더 큰 state를 찾을 수 있다.

![image](https://user-images.githubusercontent.com/55575547/100297561-490a9500-2fd2-11eb-9049-d4bf7c13b5f6.png)

> Exploration(Random) & Exploitation(Greedy)
>
> - 서로 trade-off 관계가 있음. 



그래서 `Decaying epsilon Greedy` 방법론이 적용이 됨.

-> 처음에는 exploration을 많이 하다가 점점 greedy한 방법으로 수행(e : 0.9 -> 0.1)



현재 path의 랭크간에 점수가 1로 동일하기 때문에 보다 더 효율적인 경로를 탐색하기 위한 점수 배정이 필요함.

이것을 **Discount factor**, (\gamma) 표시로 칭함. 

![수식](C:\tlsghwls44_git\tlsghwls44.github.io\assets\img\post_img\CodeCogsEqn.gif) : 0~1 사이의 값. Action에 대해서 평가할 때 이전 state의 가장 큰 값에 대해 gamma를 곱해서 update를 해라.

![image](https://user-images.githubusercontent.com/55575547/100297941-2927a100-2fd3-11eb-9eb4-c41f9a36580e.png)



- **Discount factor**의 장점

  ① 효율적인 Path를 찾을 수 있음.

  ② 현재 vs 미래 reward를 비교할 수 있음.

  ​	ex) &gamma; 값이 1에 가까우면 현재에 받을 reward에 충실하고 , 0에 가까우면 미래에 받을 reward에 충실하다.



**Q-update**

![{\displaystyle Q(s_{t},a_{t})\leftarrow (1-\alpha )\cdot \underbrace {Q(s_{t},a_{t})} _{\rm {old~value}}+\underbrace {\alpha } _{\rm {learning~rate}}\cdot \left(\overbrace {\underbrace {r_{t}} _{\rm {reward}}+\underbrace {\gamma } _{\rm {discount~factor}}\cdot \underbrace {\max _{a}Q(s_{t+1},a)} _{\rm {estimate~of~optimal~future~value}}} ^{\rm {learned~value}}\right)}](https://wikimedia.org/api/rest_v1/media/math/render/svg/8158847df27c65c1ecb2fde471c62f197f3d6738)

<center>From wiki</center>



알고리즘이 시작되기 전에 Q 함수는 고정된 임의의 값을 가진다.

각 시간 t에 에이전트는 어떠한 상태 s<sub>t<sub>에서 행동 a<sub>t<sub>를 취하고 새로운 상태 s<sub>t+1<sub>로 전이한다. 이 때 보상 r<sub>t<sub>가 얻어지며, Q 함수가 갱신된다.

알고리즘의 핵심은 이전의 값과 새 정보의 가중합(weighted sum)을 이용하는 간단한 값 반복법이다.

도달한 상태 s<sub>t+1<sub>이 종결 상태일 경우 알고리즘 의 episode 하나가 끝난다. 그러나 Q러닝은 작업이 에피소드로 구성되지 않더라도 학습이 가능하다. Discount factor가 1보다 작을 경우 무한히 반복하더라도 Discount 총계는 유한하기 때문이다.



### Markov Decision Process

![image](https://user-images.githubusercontent.com/55575547/100299539-4bbbb900-2fd7-11eb-8553-53cff6062da6.png)

  **Markov Decision Process 속성**

 1. ![2020-11-26-RL_8](C:\tlsghwls44_git\tlsghwls44.github.io\assets\img\post_img\2020-11-26-RL_8.gif) 

    s<sub>a<sub>와 a<sub>0<sub>에 대한 정보는 이미 s<sub>1<sub>에 다 포함되므로 a<sub>1<sub>의 확률 예측시 포함하지 않는다.

    -> 한 다리로 연결되어있는 거는 연관성이 있지만 두 다리 건너서는 필요없음. 단 a_1과 s_1은 세트임(한 뭉탱이)

	2.  ![2020-11-26-RL_9](C:\tlsghwls44_git\tlsghwls44.github.io\assets\img\post_img\2020-11-26-RL_9.gif)

    -> 요거는 transition probability라고 한다.

- s_1이라는 state에서 a_1이라는 action을 취할 확률분포(1은 time)를 **Policy, 정책**이라고 한다.



강화학습에서 목표

Goal = maximize **Reward**

다시말하면

Goal = maximize **Expected Return**



**Return의 정의**

 ![2020-11-26-RL_10](C:\tlsghwls44_git\tlsghwls44.github.io\assets\img\post_img\2020-11-26-RL_10.gif)

- G_t는 Discounted reward의 sum
- a_t라는 행동을 했을 때 받는 reward를 r_t+1라고 취한다.