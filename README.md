# MT_project
## Intro
1. 이미지 전처리는 컨투어를 이용해 사각형으로 잘라내 사용
2. 비슷하게 생긴 대장내시경 데이터 세트로 Pretrain 시켜 MT 데이터세트에 적용해볼 계획임
3. Hyper kvasir의 라벨링 된 이미지 데이터세트를 이용해 Pretrain 진행 => 데이터 n수가 작아 LUMIC 데이터세트로 교체해서 학습중(07.23)
4. 학습은 Xception 모델로 Adam Optimizer를 사용함
5.  optimizer는 AdamW and AdaBelief 도 사용하여 테스트 예임
