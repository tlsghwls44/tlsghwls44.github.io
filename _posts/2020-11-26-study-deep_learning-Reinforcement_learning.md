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

 ![2020-11-26-RL_1](C:\tlsghwls44_git\tlsghwls44.github.io\assets\img\post_img\2020-11-26-RL_1.jpg)



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





Greedy action()

Action을 취하면서 점수를 매기는데 그 점수가 가장 큰 action을 Greedy action이라고 칭한다.

Episode 1에서는 모든 방향이 다 점수가 0이므로 action을 random하게 취한다. (한 번도 목표점에 도달하지 못했기 때문에)

Q값(랭크를 매긴 점수)는 이동을 하면서 update 된다. 

![2020-11-26-RL_3](C:\tlsghwls44_git\tlsghwls44.github.io\assets\img\post_img\2020-11-26-RL_3.jpg)