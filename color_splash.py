import numpy as np
import skimage
import cv2


def color_splash(image, mask):

    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255

    # 传入mask为(1,n,H,W),变换为(H,W,1),便于后续np.where()使用
    mask = (np.sum(mask, 1) >= 1) # 将多个bbox的mask合并
    mask = mask.transpose(1,2,0)

    splash = np.where(mask, image, gray).astype(np.uint8) #np.where(cond,x,y)：满足条件cond,输出x，不满足输出y
    return splash

def detect_and_color_splash(model, video_path=None):
    assert video_path

    vcapture = cv2.VideoCapture(video_path)
    width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vcapture.get(cv2.CAP_PROP_FPS)

    # Define codec and create video writer
    file_name = "/share12/home/zhenni/PythonProject/openmmlab-camp/mmdet-1/output.avi"
    vwriter = cv2.VideoWriter(file_name,
                                cv2.VideoWriter_fourcc(*'MJPG'),
                                fps, (width, height))

    count = 0
    success = True
    while success:
        print("frame: ", count)
        # Read next image
        success, image = vcapture.read()
        if success:
            # OpenCV returns images as BGR, convert to RGB
            image = image[..., ::-1]
            # Detect objects
            result = inference_detector(model, image)
            # Color splash
            splash = color_splash(image, result[1])
            # RGB -> BGR to save image to video
            splash = splash[..., ::-1]
            # Add image to video writer
            vwriter.write(splash)
            count += 1
        vwriter.release()
    print("Saved to ", file_name)

if __name__ == '__main__':

    from mmdet.apis import init_detector, inference_detector, show_result_pyplot

    config_file = '/share12/home/zhenni/PythonProject/openmmlab-camp/mmdet-1/train.py'
    checkpoint_file = '/share12/home/zhenni/PythonProject/openmmlab-camp/mmdet-1/work_dirs/train/latest.pth'

    model = init_detector(config_file, checkpoint_file) # 构建模型

    detect_and_color_splash(model,video_path='/share12/home/zhenni/PythonProject/openmmlab-camp/mmdet-1/test_video.mp4')