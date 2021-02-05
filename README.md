# ResNet-18 in Pytorch

## Introduction

## Result

Total test accuracy: 85.08%

Training result visualised:
![alt text](https://github.com/jason2468087/Pytorch-ResNet/blob/main/img/ResNet%20Result.png?raw=true)

Accuracy per class:
| Airplanes | Automobile | Birds | Cats | Deer | Dogs | Frogs | Horses | Ships | Trucks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 90.19% | 92.00% | 80.12% | 71.41% | 81.70% | 73.19% | 93.51% | 87.80% | 91.82% | 90.97% |

Confusion Matrix:

| Predict\Actual | Airplanes | Automobile | Birds | Cats | Deer | Dogs | Frogs | Horses | Ships | Trucks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Airplanes | 855 | 7 | 33 | 9 | 17 | 4 | 1 | 13 | 41 | 20 |
| Automobile | 2 | 954 | 0 | 4 | 1 | 0 | 0 | 3 | 5 | 31 |
| Birds | 34 | 2 | 794 | 40 | 51 | 32 | 19 | 14 | 7 | 7 |
| Cats | 10 | 6 | 38 | 677 | 33 | 172 | 21 | 29 | 5 | 9 |
| Deer | 5 | 2 | 39 | 36 | 844 | 26 | 11 | 30 | 6 | 1 |
| Dogs | 5 | 3 | 20 | 90 | 25 | 830 | 2 | 24 | 1 | 0 |
| Frogs | 4 | 6 | 40 | 64 | 29 | 28 | 821 | 3 | 1 | 4 |
| Horses | 8 | 1 | 17 | 21 | 28 | 35 | 0 | 885 | 0 | 5 |
| Ships | 17 | 11 | 7 | 5 | 4 | 5 | 2 | 2 | 932 | 15 |
| Trucks | 8 | 45 | 3 | 2 | 1 | 2 | 1 | 5 | 17 | 916 |
