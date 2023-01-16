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


num_source = 1
#master_ip = "169.254.164.143"

vis = True


def stop_recording(PORT, shm_nm, nptype):
    temp_shm = shared_memory.SharedMemory(name = shm_nm)
    arr = np.frombuffer(buffer=temp_shm.buf, dtype=nptype)

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((my_ip, PORT))
    s.listen(0)
    c, addr = s.accept()
    arr[:] = np.ones(1)

def pose_est(ii):
    pykinect.initialize_libraries(track_body=True)

    # Modify camera configuration
    device_config = pykinect.default_configuration
    device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_OFF
    device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED


    if ii == 0:
        #device_config.wired_sync_mode = pykinect.K4A_WIRED_SYNC_MODE_MASTER
        device_config.wired_sync_mode = pykinect.K4A_WIRED_SYNC_MODE_STANDALONE
    else:
        #device_config.camera_fps = pykinect.K4A_WIRED_SYNC_MODE_SUBORDINATE
        device_config.wired_sync_mode = pykinect.K4A_WIRED_SYNC_MODE_STANDALONE

    device = pykinect.start_device(config=device_config, device_index=1)
    print("Kinect_" + format(ii, "02") + " Ready")


    # Start body tracker
    bodyTracker = pykinect.start_body_tracker(model_type=0)
    #model_type=pykinect.K4ABT_DEFAULT_MODEL


    while True:
        print('a')
        capture = device.update()
        print('b')
        body_frame = bodyTracker.update(capture)
        print('c')
        if body_frame.get_num_bodies() != 0:
            joint = body_frame.get_body().numpy()
        else:
            joint = np.zeros((1, 27 * 4))
        joint_[:] = np.ravel(joint[0:27, :])
        print('d')
        print(joint_)



if __name__ == "__main__":

    # Initialize the library, if the library is not found, add the library path as argument
    # Modify camera configuration
    pykinect.initialize_libraries(track_body=True)

    # Modify camera configuration
    device_config = pykinect.default_configuration
    device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_OFF
    device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
    # Initialize the library, if the library is not found, add the library path as argument
    # stop_signal = np.zeros((1), dtype=int)
    # shm = shared_memory.SharedMemory(create=True, size=stop_signal.nbytes)
    # shared_signal = np.frombuffer(shm.buf, stop_signal.dtype)
    # com = mp.Process(target=stop_recording,args=[stop_receive_port, shm.name, stop_signal.dtype])
    # com.start()



    # last_action = 0
    #
    for ii in range(num_source):
        if ii == 0:
            #device_config.wired_sync_mode = pykinect.K4A_WIRED_SYNC_MODE_MASTER
            device_config.wired_sync_mode = pykinect.K4A_WIRED_SYNC_MODE_STANDALONE
        else:
            #device_config.camera_fps = pykinect.K4A_WIRED_SYNC_MODE_SUBORDINATE
            device_config.wired_sync_mode = pykinect.K4A_WIRED_SYNC_MODE_STANDALONE

        locals()[f"device_{ii}"] = pykinect.start_device(config=device_config, device_index=ii)
        print("Kinect_" + format(ii, "02") + " Ready")

    for ii in range(num_source):
        # Start body tracker
        locals()[f"bodyTracker_{ii}"] = pykinect.start_body_tracker(model_type=1)
        #model_type=pykinect.K4ABT_DEFAULT_MODEL
    k = 0

    for ii in range(num_source):
        locals()[f"joint_{ii}"] = np.zeros((27 * 4,))

    print("###############All Kinect Ready Waiting the Start Signal##################")

    print("Start")


    k = 0
    while True:
        start_t = timeit.default_timer()
        # Get capture
        for ii in range(num_source):
            locals()[f"capture_{ii}"] = locals()[f"device_{ii}"].update()

        for ii in range(num_source):
            # Get body tracker frame
            locals()[f"body_frame_{ii}"] = locals()[f"bodyTracker_{ii}"].update(locals()[f"capture_{ii}"])
        # for ii in range(num_source):
        #     locals()[f"com_{ii}"] = mp.Process(target=pose_est, args=[ii])
        #     locals()[f"com_{ii}"].start()
        # for ii in range(num_source):
        #     locals()[f"com_{ii}"].join()

        alist = []
        for ii in range(num_source):
            if locals()[f"body_frame_{ii}"].get_num_bodies() != 0:
                joint = locals()[f"body_frame_{ii}"].get_body().numpy()
            else:
                joint = np.zeros((1, 27 * 4))
            locals()[f"joint_{ii}"][:] = np.ravel(joint[0:27, :])
            alist.append(locals()[f"joint_{ii}"])


        joints_ = np.concatenate((alist))
        print(joints_)
        if vis:
            # Get the color depth image from the capture
            for ii in range(num_source):
                ret, locals()[f"depth_color_image_{ii}"] = locals()[f"capture_{ii}"].get_colored_depth_image()
            # Get the colored body segmentation
            # ret, body_image_color = body_frame.get_segmentation_image()
            # ret, body_image_color1 = body_frame1.get_segmentation_image()
            if not ret:
                continue

            # Combine both images
            # combined_image = cv2.addWeighted(depth_color_image, 0.6, body_image_color, 0.4, 0)
            # combined_image1 = cv2.addWeighted(depth_color_image1, 0.6, body_image_color1, 0.4, 0)
            for ii in range(num_source):
                # Draw the skeletons
                locals()[f"combined_image_{ii}"] = locals()[f"body_frame_{ii}"].draw_bodies(
                    locals()[f"depth_color_image_{ii}"])
                # Overlay body segmentation on depth image
                cv2.imshow('Depth image with skeleton_' + str(ii), locals()[f"combined_image_{ii}"])
            #Press q key to stop
            if cv2.waitKey(1) == ord('q'):
                break

        terminate_t = timeit.default_timer()
        FPS = 1 / (terminate_t - start_t)
        print("FPS: " + str(FPS))
        print("Frame: " + str(k))
        k += 1
