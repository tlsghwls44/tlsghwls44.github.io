---
layout: post
title: 설계구조 기반 이상진단 방법론 정의
subtitle: 설계구조 기반 이상진단 방법론 정의
categories: research
tags: P&ID_abnormal_analysis
comments: true
published: true
---
# 개요

1. 문제 정의
2. 방법론
3. 데이터 정의
4. 결과물
5. 결론

## 1. 문제 정의

PFD와 P&ID내 심볼과 텍스트를 인식하여 설계오류를 진단할 수 있는 알고리즘은 현재까지 개발된 적이 없다.

기존에는 QC(Quailty Control) 단계에서 manually하게 공정도면에 존재하는 오류 점검을 수행하였고 이는 `작업 소요시간` 및 `오류 발생률` 측면에서 개선을 필요로 한다.

이 논문에서는 각 공정별로 typical한 **공정구조**가 존재한다는 점을 바탕으로 기존 엔지니어링 분야에서 문서화된 정보를 활용하여 도면 이상진단을 수행하는 방법론을 제안한다.

## 2. 방법론

1) Sequence 모델 구축 단계  
- 도면 이상진단을 위해 각 공정 도면들로부터 심볼과 텍스트를 추출하고, 이를 이용하여 공정 unit간 연결성을 부여한 sequence 데이터를 형성하는 모델을 구축한다.

2) General process structure 추출 단계
 - 기존 도면 이미지로부터 구축한 sequence 모델을 이용해 general process structure를 추출한다.  
데이터 추출방법은 sequence 모델을 통해 추출한 모든 데이터를 pool에 저장한 후 기 정의된 threshold값을 초과하는 경우에만 general process structure로 목록화하는 절차로 진행된다. 
 
3) 도면 이상진단 수행 단계
 - 테스트 도면으로 인식한 sequence 데이터와 2번째 단계에서 추출한 general process structure간 비교를 통해 이상진단 여부를 판단한다.
 

## 3. 데이터 정의
```
[펌프]
Case study 1
 - 모든 원심 펌프의 suction 배관에는 영구 strainer를 설치한다.  
 	* suction 배관 크기가 3inch 이상 : "T" type 또는 "Basket" type  
 	* suction 배고나 크기가 2inch 이하 : "y" type

Case study 2
 - 모든 펌프, 컴프레서 등의 압축장비는 discharge쪽에 check valve를 설치한다.  
 역류가 형성될 가능성이 없는 경우(예 : 위치나 압력이 낮은 곳으로 유체를 이송하는 단일 펌프인 경우)를 제외한다.
 
Case study 3
 - Discharge측 압력계기는 check valve 전단에 설치한다.

참고문헌 : 화공도면실무 / 이주영, 박근수 지음
```


[asdf](https://naver.com)  
`메롱`

유형1(`설명어`를 클릭하면 URL로 이동) : [TheoryDB 블로그](https://theorydb.github.io "마우스를 올려놓으면 말풍선이 나옵니다.")  
유형2(URL 보여주고 `자동연결`) : <https://theorydb.github.io>  
유형3(동일 파일 내 `문단 이동`) : [동일파일 내 문단 이동](#markdown의-반드시-알아야-하는-문법) 


![이미지](https://github.com/tlsghwls44/tlsghwls44.github.io/blob/master/assets/img/main_column2.jpg)
