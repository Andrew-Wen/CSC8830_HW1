import cv2
import depthai as dai
import numpy as np
import time

def getFrame(queue):
    #Get the frame from the queue
    frame = queue.get()
    #convert the frame to OpenCV format and return
    return frame.getCvFrame()

def getMonoCamera(pipeline, isLeft):
    #configure mono camera
    mono = pipeline.createMonoCamera()

    #Set the camera resolution
    mono.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

    if isLeft:
        #Get left camera
        mono.setBoardSocket(dai.CameraBoardSocket.LEFT)
    else:
        mono.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    return mono

def getStereoPair(pipeline, monoLeft, monoRight):
    #Configure stereo pair for the depth estimation
    stereo = pipeline.createStereoDepth()
    # Checks occuluded pixels and marks them as invalid
    stereo.setLeftRightCheck(True)

    #configure left and right cameras to work as a stereo pair
    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)

    return stereo


def mouseCallback(event, x, y, flags, param):
    global mouseX, mouseY
    if event == cv2.EVENT_LBUTTONDOWN:
        mouseX = x
        mouseY = y

if __name__=='__main__':

    mouseX = 0
    mouseY = 640
    #Start defining a pipeline
    pipeline = dai.Pipeline()

    #set up left and right cameras
    monoLeft = getMonoCamera(pipeline, isLeft = True)
    monoRight = getMonoCamera(pipeline, isLeft = False)
    #combine left and right cameras to form a stereo pair
    stereo = getStereoPair(pipeline, monoLeft, monoRight)

    camRgb = pipeline.create(dai.node.ColorCamera)
    camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_13_MP)
    camRgb.setVideoSize(1080, 720)
    #Set XLinkOut for disparity, rectifiedLeft, and rectifiedRight
    xoutDisp = pipeline.createXLinkOut()
    xoutDisp.setStreamName("disparity")

    xoutRectifiedLeft = pipeline.createXLinkOut()
    xoutRectifiedLeft.setStreamName("rectifiedLeft")

    xoutRectifiedRight = pipeline.createXLinkOut()
    xoutRectifiedRight.setStreamName("rectifiedRight")

    stereo.disparity.link(xoutDisp.input)

    stereo.rectifiedLeft.link(xoutRectifiedLeft.input)

    stereo.rectifiedRight.link(xoutRectifiedRight.input)
    xoutRgb = pipeline.create(dai.node.XLinkOut)
    xoutRgb.setStreamName("color")
    camRgb.video.link(xoutRgb.input)
    xoutRgb.input.setBlocking(False)
    xoutRgb.input.setQueueSize(1)
    # Pipeline is now defined, now we can connect to the device
    with dai.Device(pipeline, True) as device:

        # Output queues will be used to get the rgb frames and nn data from the outputs defined above
        disparityQueue = device.getOutputQueue(name="disparity", maxSize=1, blocking=False)
        rectifiedLeftQueue = device.getOutputQueue(name="rectifiedLeft", maxSize=1, blocking=False)
        rectifiedRightQueue = device.getOutputQueue(name="rectifiedRight", maxSize=1, blocking=False)
        video = device.getOutputQueue(name="color", maxSize=1, blocking=False)

        # Calculate a multiplier for colormapping disparity map
        disparityMultiplier = 255 / stereo.initialConfig.getMaxDisparity()
        # stereo.getMaxDisparity()

        
        prev_frame_time = 0
        new_frame_time = 0

        max_fps_rgb = 0

        while True:
            
            # Get disparity map
            disparity = getFrame(disparityQueue)

            new_frame_time = time.time()
            fps = 1/(new_frame_time-prev_frame_time)
            prev_frame_time = new_frame_time
            fps = int(fps)

            videoIn = getFrame(video)

            if fps >max_fps_rgb:
                max_fps_rgb = fps
                print("Max fps achieved:", max_fps_rgb)
            

            # converting the fps to string so that we can display it on frame
            # by using putText function
            fps = str(fps)

            # putting the FPS count on the frame
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(videoIn, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
            # cv2.putText(disparity, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)

            # Colormap disparity for display
            disparity = (disparity * disparityMultiplier).astype(np.uint8)
            disparity = cv2.applyColorMap(disparity, cv2.COLORMAP_JET)
            # Get left and right rectified frame
            leftFrame = getFrame(rectifiedLeftQueue)
            rightFrame = getFrame(rectifiedRightQueue)
           

            cv2.imshow("Disparity", disparity)
            cv2.imshow("RGB", videoIn)
            # Check for keyboard input
            key = cv2.waitKey(1)
            if key == ord('q'):
                # Quit when q is pressed
                break
        