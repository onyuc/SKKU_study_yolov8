import cv2
from ultralytics import YOLO

# 웹캠 캡처기 생성
cap = cv2.VideoCapture("cat.mp4")

# YOLO 모델 로드
model = YOLO('yolov8n-cls.pt')

while cap.isOpened():
    # 캡처기로부터 프레임 읽기
    ret, frame = cap.read()

    if ret:
        # 읽은 프레임을 출력 윈도우에 표시
        cv2.imshow('frame', frame)

        # 모델을 통한 detection
        results = model(frame)
        inference_plot = results[0].plot()
        cv2.imshow('frame2', inference_plot)
    # 'q' 키를 눌러 종료
    if cv2.waitKey(1) == ord('q'):
        break

# 캡처기와 비디오 쓰기기 해제
cap.release()

# 출력 윈도우 닫기
cv2.destroyAllWindows()