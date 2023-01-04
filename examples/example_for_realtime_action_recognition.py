import sys
import cv2
sys.path.insert(1, '../')
import pykinect_azure as pykinect
import timeit
import numpy as np
from mlsocket import MLSocket
import socket

num_source = 2
ip = "192.168.0.5"
send_port = 5555
vis = 1

end_port = 9999


if __name__ == "__main__":
    pykinect.initialize_libraries(track_body=True)

    # Modify camera configuration
    device_config = pykinect.default_configuration
    device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_OFF
    device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
    # Initialize the library, if the library is not found, add the library path as argument

    # Modify camera configuration
    #playback = pykinect.start_playback(video_filename)

    for ii in range(num_source):
        print("device_"+str(ii)+" is ready")
        locals()[f"device_{ii}"] = pykinect.start_device(config=device_config, device_index=ii)

    #device = pykinect.start_device(config=device_config, device_index=0)

    for ii in range(num_source):
        # Start body tracker
        locals()[f"bodyTracker_{ii}"] = pykinect.start_body_tracker()
    k = 0

    for ii in range(num_source):
        locals()[f"joint_{ii}"] = np.zeros((27 * 4,))

    print("Wait")
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(('', 12345))
    m = s.recvfrom(1024)
    print(m)
    k = 0
    while True:
        start_t = timeit.default_timer()

        # Get capture
        for ii in range(num_source):
            locals()[f"capture_{ii}"] = locals()[f"device_{ii}"].update()

        for ii in range(num_source):
            # Get body tracker frame
            locals()[f"body_frame_{ii}"] = locals()[f"bodyTracker_{ii}"].update(locals()[f"capture_{ii}"])

        alist = []
        for ii in range(num_source):
            if locals()[f"body_frame_{ii}"].get_num_bodies() != 0:
                joint = locals()[f"body_frame_{ii}"].get_body().numpy()
            else:
                joint = np.zeros((1,27*4))
            locals()[f"joint_{ii}"][:] = np.ravel(joint[0:27, :])
            alist.append(locals()[f"joint_{ii}"])



        # #print('device1: '+ str(body_frame.get_device_timestamp_usec()))
        # #print('device2: ' + str(body_frame1.get_device_timestamp_usec()))
        # timestamp[k, :] = [body_frame.get_device_timestamp_usec(), body_frame1.get_device_timestamp_usec()]


        joints_ = np.concatenate((alist))
        # print(np.shape(joints_))
        # np.savetxt('ouput1.csv',joints, delimiter=" ")
        # np.savetxt('ouput2.csv', joints1, delimiter=" ")
        # np.savetxt('timestamp.csv', timestamp,delimiter=" ")
        #break

        send_socket = MLSocket()
        send_socket.connect((ip, send_port))
        send_socket.send(joints_)

        if vis:
            # Get the color depth image from the capture
            for ii in range(num_source):
                ret, locals()[f"depth_color_image_{ii}"] = locals()[f"capture_{ii}"].get_colored_depth_image()
            # Get the colored body segmentation
            #ret, body_image_color = body_frame.get_segmentation_image()
            #ret, body_image_color1 = body_frame1.get_segmentation_image()
            if not ret:
                continue

            # Combine both images
            #combined_image = cv2.addWeighted(depth_color_image, 0.6, body_image_color, 0.4, 0)
            #combined_image1 = cv2.addWeighted(depth_color_image1, 0.6, body_image_color1, 0.4, 0)
            for ii in range(num_source):
            # Draw the skeletons
                locals()[f"combined_image_{ii}"] = locals()[f"body_frame_{ii}"].draw_bodies(locals()[f"depth_color_image_{ii}"])
                # Overlay body segmentation on depth image
                cv2.imshow('Depth image with skeleton_'+str(ii), locals()[f"combined_image_{ii}"])
            # Press q key to stop
            if cv2.waitKey(1) == ord('q'):
                break

        terminate_t = timeit.default_timer()
        FPS = int(1 / (terminate_t - start_t))
        print("FPS: " + str(FPS))
        print("Frame: " + str(k))
        k += 1
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    s.sendto(str(1).encode(), (ip, end_port))