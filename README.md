# Back Indicator

MediaPipe pose tracking으로 카메라 영상에서 운동 동작을 분류하고, 화면에 관절 각도와 PTZ 상태를 표시하는 Python 프로젝트입니다.

현재 분류 대상은 다음 세 가지입니다.

- Push-up
- Lunge
- Squat

조건에 맞지 않거나 관절 인식 신뢰도가 낮으면 `Unknown`으로 표시됩니다.

## Features

- OpenCV 카메라 입력 처리
- MediaPipe Pose 기반 관절 추적
- 팔꿈치 각도, 무릎 각도 표시
- 최근 7프레임 기준 운동명 안정화
- 브라우저에서 볼 수 있는 WebRTC 서버
- Digital PTZ 크롭 추적 옵션
- ONVIF PTZ 카메라 제어 옵션

## Requirements

- Python 3.10 이상 권장
- 웹캠 또는 영상 입력 장치
- 선택 사항: ONVIF PTZ 지원 카메라

필요한 Python 패키지:

```powershell
pip install opencv-python mediapipe numpy pandas matplotlib aiohttp aiortc av onvif-zeep
```

설치 확인:

```powershell
python check_lib.py
```

## Usage

### OpenCV 창으로 실행

```powershell
python pose_analysis.py --camera 0
```

기본 카메라 인덱스는 `2`입니다. 카메라가 열리지 않으면 `0`, `1`, `2`처럼 바꿔서 실행합니다.

```powershell
python find_cameras.py
```

사용 가능한 카메라 인덱스를 확인할 수 있습니다.

종료는 OpenCV 창에서 `q`를 누릅니다.

### WebRTC 서버로 실행

```powershell
python webrtc_server.py --host 127.0.0.1 --port 8080 --camera 0
```

브라우저에서 다음 주소를 엽니다.

```text
http://127.0.0.1:8080
```

페이지에서 `Start WebRTC` 버튼을 누르면 분석된 카메라 영상이 표시됩니다.

### Digital PTZ 사용

```powershell
python pose_analysis.py --camera 0 --digital-ptz
```

또는 WebRTC 서버에서:

```powershell
python webrtc_server.py --camera 0 --digital-ptz
```

### ONVIF PTZ 사용

```powershell
python pose_analysis.py --camera 0 --ptz --ptz-ip 192.168.0.20 --ptz-user admin --ptz-password "password"
```

WebRTC 서버에서도 같은 옵션을 사용할 수 있습니다.

```powershell
python webrtc_server.py --camera 0 --ptz --ptz-ip 192.168.0.20 --ptz-user admin --ptz-password "password"
```

## Exercise Rules

운동 분류는 `pose_analysis.py`의 `classify_exercise()`에서 처리합니다.

- 공통: 기준 관절 visibility가 `0.5` 이상이어야 합니다.
- Push-up: 몸통 각도 `< 35`, 손목이 어깨보다 아래에 있음
- Lunge: 양쪽 다리 관절이 보이고, 몸통 각도 `> 45`, 양발 간격 비율 `> 0.45`, 뒤쪽 무릎각 `< 150`, 뒤쪽 힙각 `< 45`
- Squat: 몸통 각도 `> 45`, 무릎각 `< 150`

## Project Structure

```text
.
├── auto_ptz.py          # 자동 PTZ 추적 로직
├── check_lib.py         # 설치된 라이브러리 버전 확인
├── find_cameras.py      # 사용 가능한 카메라 인덱스 확인
├── index.html           # WebRTC 클라이언트 페이지
├── main.py              # MediaPipe Tasks 기반 테스트 코드
├── onvif_controller.py  # ONVIF PTZ 제어
├── pose_analysis.py     # 운동 분류와 화면 오버레이
└── webrtc_server.py     # WebRTC 영상 서버
```

## Notes

- 현재 카메라 프레임은 좌우 반전해서 처리합니다.
- PTZ 옵션은 ONVIF 프로필에 PTZ 설정이 있는 카메라에서만 동작합니다.
- `.venv`, IDE 설정, 캐시, 로그 파일은 Git에 포함하지 않습니다.

