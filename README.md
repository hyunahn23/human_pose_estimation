# human_pose_estimation
mmpose를 활용하여 human pose estimation 기능을 구현. 

---

## 환경 설정

- Python 3.8  
- PyTorch 2.0.1  
- CUDA 11.8  

필수 패키지 설치:

```bash
penmim과 mmpose 설치에 필요한 패키지들을 설치
pip install -U openmim
mim install mmengine
mim install "mmcv==2.1.0"
mim install "mmdet==3.3.0"
mim install "mmpose==1.3.2"
```

---

##  Config 파일
```bash
# YOLOXPose-M 모델의 config 코드
[gdown 1r3B1xhkyKYcQ9SR7o9hw9zhNJinRiHD-](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/yolox_pose/yoloxpose_m_8xb32-300e_coco-640-84e9a538_20230829.pth)
```

---

## Result (해변가에서 제가 직접 저의 모습을 촬영한 모습입니다.)
[![Watch the Video](https://img.shields.io/badge/🎬%20Result-blue?style=for-the-badge)](https://raw.githubusercontent.com/hyunahn23/human_pose_estimation/main/result_2.mp4)
<img src="[result.gif](https://github.com/hyunahn23/human_pose_estimation/main/result_2.mp4)" width="100%">


