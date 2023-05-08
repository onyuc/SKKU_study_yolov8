import cv2

# 웹캠 캡처기 생성
cap = cv2.VideoCapture(0)

while True:
    # 캡처기로부터 프레임 읽기
    ret, frame = cap.read()

    if ret:
        # 읽은 프레임을 출력 윈도우에 표시
        cv2.imshow('frame', frame)

    # 'q' 키를 눌러 종료
    if cv2.waitKey(1) == ord('q'):
        break

# 캡처기와 비디오 쓰기기 해제
cap.release()

# 출력 윈도우 닫기
cv2.destroyAllWindows()
