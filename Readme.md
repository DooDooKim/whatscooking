# What's Cooking #

---
**요약 / 소스코드 설명**

<https://www.kaggle.com/c/whats-cooking>

train data를 8:2로 random split

1. csv_make.py로 train dataset을 Fasttext, skip-gram 방식으로 Embedding 해주고, csv 파일로 저장한다. (path 변경 필요)
2. ensemble_clfs.py로 트레이닝을 하고 accuracy를 확인한다. (총 5개의 CNN 모델 ensemble)

---
**결과**

![default](https://user-images.githubusercontent.com/32383404/50070475-7577dd00-0211-11e9-892d-bc2748e63eba.PNG)

---


 