# import cv2
# import numpy as np
#
# # img를 새 창에 띄우기
# img = cv2.imread("originalphoto.png", cv2.IMREAD_COLOR)
# # cv2.imshow("TEST",img)
# # cv2.waitKey(0)
#
# # 다각형 꼭지점 좌표 정의
# pts = np.array([[255, 24], [355, 24], [442, 367], [129, 367]], np.int32)
# pts = pts.reshape((-1, 1, 2))
#
# # 다각형 내부 마스크 생성
# # 흑백의 사진을 만들고 다각형을 그린 다음 내부를 흰색으로 채운 후 기존의 사진과 결합하는 형태
# mask = np.zeros(img.shape[:2], dtype=np.uint8)
# cv2.fillPoly(mask, [pts], 255)
#
# # 이미지와 마스크를 비트와 연산하여 다각형 영역 선택
# result = cv2.bitwise_and(img, img, mask=mask)
#
# # 결과 이미지 출력
# cv2.imshow('Result', result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# # # img 클릭 후 엔터 -> x,y,height,width 값 추출
# # x_pos, y_pos, width, height = cv2.selectROI("point", img, False)
# # print("x, y: ", x_pos, y_pos)
# # print("w, h: ", width, height)
# # cv2.destroyAllWindows()

############################################################################
# import numpy as np
# import cv2
# import glob
#
# # 체커보드 패턴의 가로, 세로 내부 코너 개수
# nx = 12
# ny = 8
#
# # 객체 포인트 초기화 (3D 좌표)
# objp = np.zeros((ny * nx, 3), np.float32)
# objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
#
# # 객체 포인트 및 이미지 포인트 수집
# objpoints = []  # 실제 3D 좌표
# imgpoints = []  # 이미지 상의 2D 좌표
#
# images = glob.glob('checkerboard_img/*.jpg')  # 체커보드 패턴 이미지들의 경로
#
# # 이미지 수집 시작
# for idx, fname in enumerate(images):
#     print(f'Processing image {idx + 1}/{len(images)}')  # 진행 상황 출력
#     img = cv2.imread(fname)
#
#     # 이미지를 반으로 줄임
#     smaller_image = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
#
#     gray = cv2.cvtColor(smaller_image, cv2.COLOR_BGR2GRAY)
#
#     # 체커보드 패턴의 코너 찾기
#     ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
#
#     if ret == True:
#         objpoints.append(objp)
#         imgpoints.append(corners)
#
# # 캘리브레이션 수행
# print('Performing calibration...')  # 진행 상황 출력
# ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
#
# # 결과 확인
# img = cv2.imread('checkerboard_img/15.jpg')
# h, w = img.shape[:2]
# new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
#
# dst = cv2.undistort(img, mtx, dist, None, new_camera_mtx)
# print(new_camera_mtx, roi, dst)
# cv2.imshow('Undistorted Image', dst)
# # cv2.resizeWindow('Undistorted Image', 800, 600)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


import cv2
import numpy as np
import os
import glob

# 체커보드의 차원 정의
CHECKERBOARD = (8, 12)  # 체커보드 행과 열당 내부 코너 수
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 각 체커보드 이미지에 대한 3D 점 벡터를 저장할 벡터 생성
objpoints = []
# 각 체커보드 이미지에 대한 2D 점 벡터를 저장할 벡터 생성
imgpoints = []

# 3D 점의 세계 좌표 정의
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

prev_img_shape = None

# 주어진 디렉터리에 저장된 개별 이미지의 경로 추출
images = glob.glob('checkerboard_img/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    img = cv2.resize(img, (800, 600))

    # 그레이 스케일로 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 체커보드 코너 찾기
    ret, corners = cv2.findChessboardCorners(gray,
                                             CHECKERBOARD,
                                             cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    # 원하는 개수의 코너가 감지되면,
    if ret == True:
        objpoints.append(objp)

        # 주어진 2D 점에 대한 픽셀 좌표 미세조정
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # 코너 그리기 및 표시
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)

    cv2.imshow('img', img)
    cv2.resizeWindow('img', 800, 600)
    cv2.waitKey(0)

cv2.destroyAllWindows()

h, w = img.shape[:2]  # 480, 640

# 알려진 3D 점(objpoints) 값과 감지된 코너의 해당 픽셀 좌표(imgpoints) 전달, 카메라 캘리브레이션 수행
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("Camera matrix : \n")  # 내부 카메라 행렬
print(mtx)

# 새 이미지를 불러옵니다.
orgImage = cv2.imread('originalphoto.jpg')  # 이미지 경로를 실제 이미지의 경로로 바꿔주세요

# 이미지 크기 조정 (만약 필요하다면)
orgImage = cv2.resize(orgImage, (800, 600))  # 800x600으로 크기 조정

# 이미지를 그레이 스케일로 변환
gray = cv2.cvtColor(orgImage, cv2.COLOR_BGR2GRAY)

# 체커보드 코너 찾기
ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                         cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

# 코너 미세조정
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

# 카메라 좌표계의 3D 좌표로 변환
rvecs, tvecs = cv2.solvePnP(objp, corners2, mtx, dist)

# 회전 벡터를 회전 행렬로 변환
rotation_matrix, _ = cv2.Rodrigues(rvecs)

# 카메라 좌표계에서의 3D 좌표 계산
camera_coords = -np.dot(rotation_matrix.T, tvecs)

print("카메라 좌표계에서의 3D 좌표:", camera_coords)
