import sys
import cv2
sys.path.insert(1, '../')
import pykinect_azure as pykinect
import timeit
import numpy as np
from mlsocket import MLSocket
import socket
import multiprocessing as mp
from multiprocessing import shared_memory
import argparse

num_source = 1
#master_ip = "169.254.164.143"
master_ip = "192.168.0.5"

#my_ip = "169.254.164.143"
my_ip = "192.168.0.5"
vis = True



def stop_recording(PORT, shm_nm, nptype):
    temp_shm = shared_memory.SharedMemory(name = shm_nm)
    arr = np.frombuffer(buffer=temp_shm.buf, dtype=nptype)

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((my_ip, PORT))
    s.listen(0)
    c, addr = s.accept()
    arr[:] = np.ones(1)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Processor collection')
    parser.add_argument('-i', '--index', dest="index", type=int, default = 0)
    arg = parser.parse_args()
    index = arg.index

    send_port = 1144 + 100*index
    print(send_port)
    stop_receive_port = 4444 + 100*index



    # Initialize the library, if the library is not found, add the library path as argument
    # Modify camera configuration
    pykinect.initialize_libraries(track_body=True)

    # Modify camera configuration
    device_config = pykinect.default_configuration
    device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_OFF
    device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
    # Initialize the library, if the library is not found, add the library path as argument
    stop_signal = np.zeros((1), dtype=int)
    shm = shared_memory.SharedMemory(create=True, size=stop_signal.nbytes)
    shared_signal = np.frombuffer(shm.buf, stop_signal.dtype)
    com = mp.Process(target=stop_recording,args=[stop_receive_port, shm.name, stop_signal.dtype])
    com.start()

    if index == 0:
        #device_config.wired_sync_mode = pykinect.K4A_WIRED_SYNC_MODE_MASTER
        device_config.wired_sync_mode = pykinect.K4A_WIRED_SYNC_MODE_STANDALONE
    else:
        #device_config.camera_fps = pykinect.K4A_WIRED_SYNC_MODE_SUBORDINATE
        device_config.wired_sync_mode = pykinect.K4A_WIRED_SYNC_MODE_STANDALONE

    device = pykinect.start_device(config=device_config, device_index=index)
    print("Kinect_" + format(index, "02") + " Ready")

    # Start body tracker
    bodyTracker = pykinect.start_body_tracker(model_type=1, index=index)
    #model_type=pykinect.K4ABT_DEFAULT_MODEL

    joint_ = np.zeros((27 * 4,))

    print("###############All Kinect Ready Waiting the Start Signal##################")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((my_ip, 12345+index*100))
    s.listen(0)
    c, addr = s.accept()
    m = c.recvfrom(1024)
    print(m)
    print("Start")

    shared_signal[0] = 0

    k = 0
    while True:
        start_t = timeit.default_timer()
        # Get capture
        capture= device.update()


        # Get body tracker frame
        body_frame = bodyTracker.update(capture)

        alist = []

        if body_frame.get_num_bodies() != 0:
            joint = body_frame.get_body().numpy()
        else:
            joint = np.zeros((1, 27 * 4))
        joint_[:] = np.ravel(joint[0:27, :])
        #alist.append(locals()[f"joint_{ii}"])
        #joints_ = np.concatenate((alist))

        send_socket = MLSocket()
        send_socket.connect((master_ip, send_port))
        send_socket.send(joint_)

        if vis:
            # Get the color depth image from the capture

            ret, depth_color_image = capture.get_colored_depth_image()
            # Get the colored body segmentation
            # ret, body_image_color = body_frame.get_segmentation_image()
            # ret, body_image_color1 = body_frame1.get_segmentation_image()
            if not ret:
                continue

            # Combine both images
            # combined_image = cv2.addWeighted(depth_color_image, 0.6, body_image_color, 0.4, 0)
            # combined_image1 = cv2.addWeighted(depth_color_image1, 0.6, body_image_color1, 0.4, 0)

                # Draw the skeletons
            combined_image = body_frame.draw_bodies(depth_color_image)
            # Overlay body segmentation on depth image
            cv2.imshow('Depth image with skeleton_' + str(index), combined_image)
            # Press q key to stop
            if cv2.waitKey(1) == ord('q'):
                break

        terminate_t = timeit.default_timer()
        FPS = int(1 / (terminate_t - start_t))
        print("FPS: " + str(FPS))
        print("Frame: " + str(k))
        k += 1

        if shared_signal[0] == 1:
            print("#####################################End############################### ")
            for ii in range(num_source):
                device.close()
            break