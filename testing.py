import cv2

list_of=[]
ret, images = cv2.imreadmulti("/Users/Sotiris/Desktop/neurons 2-fura380.tif")
backtorgb = cv2.cvtColor(images[0], cv2.COLOR_GRAY2RGB)
print(backtorgb.shape)
params = cv2.SimpleBlobDetector_Params()
params.minThreshold = 50
params.maxThreshold = 255
detector = cv2.SimpleBlobDetector_create(params)

for i in range(len(images)):
    backtorgb = cv2.cvtColor(images[i], cv2.COLOR_GRAY2RGB)
    inverted = 255 - backtorgb
    keypoints = detector.detect(inverted)
    for key in keypoints:
        coordx = key.pt[0]
        coordy = key.pt[1]
        size = key.size
        #print(coordx)
        # pellet_list.append(Pellet(coordx, coordy, size, time.time()))
        #if size > 12 and size < 30:
            #pellet_list.append(Pellet(coordx, coordy, size, time.time()))
        cv2.circle(backtorgb, (int(coordx),int(coordy)), int(size/2), (0, 0, 255), thickness=1, shift=0)
    list_of.append(backtorgb)
    cv2.imshow('frame', backtorgb)
    cv2.waitKey(10)

out = cv2.VideoWriter("filename.avi", cv2.VideoWriter_fourcc(*'XVID'), 30, (512, 512))
for i in range(len(list_of)):
    out.write(list_of[i])
out.release()


