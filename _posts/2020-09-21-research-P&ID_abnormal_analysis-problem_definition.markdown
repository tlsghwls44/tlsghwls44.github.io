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
2. 이론적 배경
3. 방법론
4. 실험결과
5. 결론

## 1. 문제 정의

PFD와 P&ID내 심볼과 텍스트를 인식하여 설계오류를 진단할 수 있는 알고리즘은 현재까지 개발된 적이 없다.

기존에는 QC(Quailty Control) 단계에서 manually하게 공정도면에 존재하는 오류 점검을 수행하였고 이는 `작업 소요시간` 및 `오류 발생률` 측면에서 개선을 필요로 한다.

이 논문에서는 각 단위공정별로 typical한 **공정구조**가 존재한다는 점을 바탕으로 기존 엔지니어링 분야에서 문서화된 정보를 활용하여 도면 이상진단을 수행하는 방법론을 제안한다.

## 2. 이론적 배경





## 3. 방법론

1) 도면 데이터 추출 모델 구축 단계

- 각 공정 도면들로부터 심볼과 텍스트를 추출한다.
- Large symbol과 Small symbol을 동시에 검출하는 모델 개발



2) Sequence 모델 구축 단계  

- 1)에서 추출된 데이터를 이용하여 공정 unit간 연결성을 부여한 sequence 데이터를 형성하는 모델을 구축한다.



3) General process structure 추출 단계

 - 기존 도면 이미지로부터 구축한 sequence 모델을 이용해 general process structure를 추출한다.  
데이터 추출방법은 sequence 모델을 통해 추출한 모든 데이터를 pool에 저장한 후 기 정의된 threshold값을 초과하는 경우에만 general process structure로 목록화하는 절차로 진행된다. 



4) 도면 이상진단 수행 단계

 - 테스트 도면으로 인식한 sequence 데이터와 2번째 단계에서 추출한 general process structure간 비교를 통해 이상진단 여부를 판단한다.



#### 데이터 정의

 - 심볼인식을 통해 인식한 object간에 연결성 및 순서를 부여한다.

**[펌프]**  

```
Case study 1
 - 모든 원심 펌프의 suction 배관에는 영구 strainer를 설치한다.  
 	* suction 배관 크기가 3inch 이상 : "T" type 또는 "Basket" type  
 	* suction 배고나 크기가 2inch 이하 : "y" type
```
![이미지1](https://raw.githubusercontent.com/tlsghwls44/tlsghwls44.github.io/master/assets/img/post_img/2020-09-22-research-P%26ID_abnormal_recognition-casestudy1.JPG)
```
Case study 2
 - 모든 펌프, 컴프레서 등의 압축장비는 discharge쪽에 check valve를 설치한다.  
 역류가 형성될 가능성이 없는 경우(예 : 위치나 압력이 낮은 곳으로 유체를 이송하는 단일 펌프인 경우)를 제외한다.
```
![이미지2](https://raw.githubusercontent.com/tlsghwls44/tlsghwls44.github.io/master/assets/img/post_img/2020-09-22-research-P%26ID_abnormal_recognition-casestudy2.JPG)
```
Case study 3
 - Discharge측 압력계기는 check valve 전단에 설치한다.
```
![이미지3](https://raw.githubusercontent.com/tlsghwls44/tlsghwls44.github.io/master/assets/img/post_img/2020-09-22-research-P%26ID_abnormal_recognition-casestudy3.JPG)


참고문헌 : 화공도면실무 / 이주영, 박근수 지음



#### 도면 데이터 추출 모델 구축 단계

















==========================================================

- P&ID를 smart P&ID 소프트웨어 데이터 형식으로 변환하는 기술

->여기서 새로운 방법론을 제시할 수 있는게 있을까?

->이상진단은 이 변환과정에서 제안할 수 있는 기반기술 중의 하나

