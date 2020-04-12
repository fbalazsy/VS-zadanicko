# Import OpenCV
import numpy as np
import cv2

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
cap = cv2.VideoCapture('movie1.mov') # 0- Primary camera ,1- External camera
#cap = cv2.VideoCapture(0) # 0- Primary camera ,1- External camera

# Video Loop - Real time processing 

while(1):
    
    # Read the image
    ret, frame = cap.read()
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    #Rotujem video
    #rotated_frame = rotateImage(frame, 38)
    rotated_frame = frame

    #Masking the part of the image
    mask = np.zeros(rotated_frame.shape[0:2], dtype=np.uint8)
    print(rotated_frame.shape[0:2])
    #Maska na rotovane 
    #points = np.array([[[0,420],[906,420],[906,470],[0,470]]])
    points = np.array([[[0,0],[200,0],[800,450],[800,532],[0,532]]])
    #Maska na original
    cv2.drawContours(mask, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)
    img = cv2.bitwise_and(rotated_frame,rotated_frame,mask = mask)


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 75, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, maxLineGap=250)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # Show the image
    cv2.imshow('Rotated video',rotated_frame)
    cv2.imshow('Cropped image',img)
    
    #cv2.imshow('mask',mask)
    #cv2.imshow('res',res)

    # End the video loop
    if cv2.waitKey(1) == 27:  # 27 - ASCII for escape key
        break

# Close and exit from camera
cap.release()
cv2.destroyAllWindows()