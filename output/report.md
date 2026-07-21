# End2End 데이터 분석 리포트 - Adult Census Income

## 1. 데이터 준비
- 원본 shape: (32561, 15)
- 정제 후 shape: (30139, 15) (결측치·중복 제거)
- 선택된 수치형 변수: ['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
- 선택된 범주형 변수: ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country']

## 2. 시각화
- [상관관계 히트맵](correlation_heatmap.png)
- [소득 그룹 비교](income_group_comparison.html)

## 3. 통계 분석
### 기술통계 (평균·표준편차·분위수)
| index | age | education_num | capital_gain | capital_loss | hours_per_week |
| --- | --- | --- | --- | --- | --- |
| mean | 38.44 | 10.12 | 1092.84 | 88.44 | 40.93 |
| std | 13.13 | 2.55 | 7409.11 | 404.45 | 11.98 |
| median | 37.00 | 10.00 | 0.00 | 0.00 | 40.00 |
| q1(25%) | 28.00 | 9.00 | 0.00 | 0.00 | 40.00 |
| q3(75%) | 47.00 | 13.00 | 0.00 | 0.00 | 45.00 |

### 상관계수
| index | age | education_num | capital_gain | capital_loss | hours_per_week |
| --- | --- | --- | --- | --- | --- |
| age | 1.00 | 0.04 | 0.08 | 0.06 | 0.10 |
| education_num | 0.04 | 1.00 | 0.12 | 0.08 | 0.15 |
| capital_gain | 0.08 | 0.12 | 1.00 | -0.03 | 0.08 |
| capital_loss | 0.06 | 0.08 | -0.03 | 1.00 | 0.05 |
| hours_per_week | 0.10 | 0.15 | 0.08 | 0.05 | 1.00 |

### t-test
- 대상: `education_num` (<=50K vs >50K)
- t-statistic=-62.3740, p-value=0.000000
- 해석: 유의미한 차이 있음 (p<0.05 기준)

## 4. ML Pipeline
- accuracy: 0.8550
- precision: 0.7972
- recall: 0.5603
- f1: 0.6581
- 저장된 모델 파일: income_classifier.joblib
