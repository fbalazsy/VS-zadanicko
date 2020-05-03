# Import OpenCV
import numpy as np
import cv2

#Loading undistortion matrix

mtx = np.loadtxt('./calibration/mtx.txt')
dist = np.loadtxt('./calibration/dist.txt')
tvecs = open('./calibration/tvecs.txt')
rvecs = open('./calibration/rvecs.txt')

#Pixels per milimeter

mierka = 1.35526315789

#Functions

def getTranslationMatrix2d(dx, dy):
    """
    Returns a numpy affine transformation matrix for a 2D translation of
    (dx, dy)
    """
    return np.matrix([[1, 0, dx], [0, 1, dy], [0, 0, 1]])


def rotateImage(image, angle):
    """
    Rotates the given image about it's centre
    """

    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    rot_mat = np.vstack([cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]])
    trans_mat = np.identity(3)

    w2 = image_size[0] * 0.5
    h2 = image_size[1] * 0.5

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    tl = (np.array([-w2, h2]) * rot_mat_notranslate).A[0]
    tr = (np.array([w2, h2]) * rot_mat_notranslate).A[0]
    bl = (np.array([-w2, -h2]) * rot_mat_notranslate).A[0]
    br = (np.array([w2, -h2]) * rot_mat_notranslate).A[0]

    x_coords = [pt[0] for pt in [tl, tr, bl, br]]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in [tl, tr, bl, br]]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))
    new_image_size = (new_w, new_h)

    new_midx = new_w * 0.5
    new_midy = new_h * 0.5

    dx = int(new_midx - w2)
    dy = int(new_midy - h2)

    trans_mat = getTranslationMatrix2d(dx, dy)
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]
    result = cv2.warpAffine(image, affine_mat, new_image_size, flags=cv2.INTER_LINEAR)

    return result

# Initialize camera
cap = cv2.VideoCapture('./videos/minus3.avi') # 0- Primary camera ,1- External camera
#cap = cv2.VideoCapture(0) # 0- Primary camera ,1- External camera

# Video Loop - Real time processing 

# Read the image
ret, frame = cap.read()

#Undistore
h,  w = frame.shape[:2]

newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))


while(1):

    # Read the image
    ret, frame = cap.read()

    dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)

    img = rotateImage(dst, 36.3)
    output = img[1700:2200,175:3980]
    img = img[1850:1860,225:3930]

    offsetY  = 150
    offsetX = 50

    h,  w = img.shape[:2]

    mask = np.zeros(img.shape[0:2], dtype=np.uint8)
    #Maska na rotovane 
    points = np.array([[[0,1],[3705,1],[3705,49],[0,49]]])
    #Maska na original
    cv2.drawContours(mask, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)
    img = cv2.bitwise_and(img,img,mask = mask)

    #convert from BGR to HSV color space
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #apply threshold
    thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)[1]

    # find contours and get one with area about 180*35
    # draw all contours in green and accepted ones in red
    contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    #area_thresh = 0
    min_area = 0.05 * w * h
    max_area = 0.99 * w * h
    result = img.copy()

    for c in contours:
        area = cv2.contourArea(c)
        cv2.drawContours(result, [c], -1, (0, 255, 0), 1)
        if area > min_area and area < max_area:
                x,y,w,h = cv2.boundingRect(c)
                cv2.rectangle(output, (x + offsetX, y + offsetY), (x + offsetX + w, y + offsetY+ h), (255, 0,0), -1)
                cv2.rectangle(output, (offsetX,offsetY), (x + offsetX,y + offsetY+h), (0, 0, 255), -1)
                vzdialenost = 'Position: ' + str(int(x/mierka))
                sirka = 'Width: ' + str(int(w/mierka))
                tmp = vzdialenost + ' mm ' + sirka + ' mm'
                cv2.putText(output, str(tmp), (25,25), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    # show images
    cv2.imshow("GRAY", gray)
    cv2.imshow("THRESH", thresh)
    cv2.imshow('RESULT', result)
    cv2.imshow('OUTPUT', output)

    # End the video loop
    if cv2.waitKey(1) == 27:  # 27 - ASCII for escape key
        break

# Close and exit from camera
cap.release()
cv2.destroyAllWindows()



