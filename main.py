import model
import torch

if __name__== '__main__':
    # input_path = 'COCO-128-2/train/000000000634_jpg.rf.2feb5ee0e764217c6796acd17da1b7fa.jpg'
    # output_path = 'result_2.png'
    input_path = 'test_video_2.mp4'
    output_path = 'result_2.mp4'    

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    cfg_path = 'configs/yoloxpose_l_8xb32-300e_coco-640.py'
    ckpt_path = 'weights/yoloxpose_l_8xb32-300e_coco-640-de0f8dee_20230829.pth'

    estimator = model.initialize(cfg_path, ckpt_path, device)

    # results = model.inference(input_path, estimator)
    # model.visualize(input_path, output_path, estimator, results)

    results = model.inference_and_save_video(input_path, output_path, estimator)
