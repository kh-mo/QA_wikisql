## horizontal flip(좌우대칭)
## vertical filp(상하대칭)
## noise
## rotatie
## translate
## zoom/stretch
## blur

import os
import cv2
import random
import numpy as np

def main():
    train_path = 'C:/Users/user/Desktop/Deep_Learning/Image_Classification/data/train'
    aug_img_count = 10

    for folder in os.listdir(train_path):
        subtrain_path = train_path + "/" + folder
        for idx, file in enumerate(os.listdir(subtrain_path)):
            subfile = subtrain_path + "/" + file

            img = cv2.imread(subfile, cv2.IMREAD_UNCHANGED)
            output = cv2.resize(img, (224, 224))

            for aic in range(aug_img_count):

                # 랜덤하게 변수 인자들 생성
                angle = random.randint(0, 360)
                change_amount_x = random.randint(-50, 50)
                change_amount_y = random.randint(-50, 50)

                # 랜덤하게 함수 선택
                rotation_index = random.randint(0, 1)
                hori_flip_index = random.randint(0, 1)
                trans_index = random.randint(0, 1)
                blur_index = random.randint(0, 1)
                # function = ["rotation", "horizontal_flip", "translate", "noise"]
                # func_index = random.randint(0,len(function)-1)

                if rotation_index == 1:
                    transfer_matrix = cv2.getRotationMatrix2D((output.shape[:2][0] / 2, img.shape[:2][1] / 2), angle,
                                                              scale=1)
                    output = cv2.warpAffine(img, transfer_matrix, (img.shape[:2][0], img.shape[:2][1]),
                                            borderValue=(255, 255, 255))
                    print("회전")
                if hori_flip_index == 1:
                    output = np.flip(output, 1)
                    print("좌우대칭")
                if trans_index == 1:
                    transfer_matrix = np.float32([[1, 0, change_amount_x], [0, 1, change_amount_y]])
                    output = cv2.warpAffine(output, transfer_matrix, (224, 224), borderValue=(255, 255, 255))
                    print("위치이동")
                if blur_index == 1:
                    output = cv2.blur(img, (7, 7))
                    print("흐릿함추가")

                # output 저장
                if not os.path.isdir(subtrain_path+"/args"):
                    os.mkdir(subtrain_path+"/args")
                cv2.imwrite(subtrain_path+"/args/" + str(idx) + "_" + str(aic) + '.jpg', output)

                # 이미지 보기
                # cv2.imshow('output', output)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()



