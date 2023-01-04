# pyKinectAzure

Python 3 library for the Azure Kinect DK sensor-SDK.

## Prerequisites
* [Azure-Kinect-Sensor-SDK](https://github.com/microsoft/Azure-Kinect-Sensor-SDK): required to build this library.
  To use the SDK, refer to the installation instructions [here](https://github.com/microsoft/Azure-Kinect-Sensor-SDK).
* **ctypes**: required to read the library.
* **numpy**: required for the matrix calculations
* **opencv-python**: Required for the image transformations and visualization.
- Python3 (over 3.9) cause of multiprocessing shared memory
- Other Python libraries can be installed by `pip install -r requirements.txt`

## How to use

```shell

example 폴더 내 example_for_save_data.py과 example_for_realtime_action_recognition.py 내의 변수
num_source = 2 #컴퓨터에 연결된 kinect 갯수
ip = "192.168.0.4" #master PC IP
send_port = 5555 #Master PC와 연결할 Port
vis = 0 #visualization 유무
```
코드 실행 절차
```shell
cd examples

# 3d 자세 Action Data 추출 시
python example_for_save_data.py

# 실시간 행동인지 시
python example_for_realtime_action_recognition.py
```