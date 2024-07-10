import cv2
import dlib
import numpy as np
import imutils
from imutils import face_utils

# Initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  # Download this file from dlib's model zoo

# Load multiple sunglasses images with an alpha channel (transparency)
sunglass_imgs = [
    cv2.imread('sunglasses1.png', -1),  # Replace with your first sunglasses image
    cv2.imread('sunglasses2.png', -1),   # Replace with your second sunglasses image
    cv2.imread('sunglasses3.png', -1)   # Replace with your second sunglasses image

]
current_sunglass_index = 0

def overlay(image, overlay, x, y, angle):
    h, w = overlay.shape[0], overlay.shape[1]
    overlay_image = overlay[..., :3]  # The first three channels are the BGR image
    mask = overlay[..., 3:] / 255.0  # The fourth channel is the alpha channel (transparency)

    # Rotate the overlay and mask
    overlay_rotated = imutils.rotate_bound(overlay_image, angle)
    mask_rotated = imutils.rotate_bound(mask, angle)

    y1, y2 = max(0, y), min(image.shape[0], y + overlay_rotated.shape[0])
    x1, x2 = max(0, x), min(image.shape[1], x + overlay_rotated.shape[1])

    y1o, y2o = max(0, -y), min(overlay_rotated.shape[0], image.shape[0] - y)
    x1o, x2o = max(0, -x), min(overlay_rotated.shape[1], image.shape[1] - x)

    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    alpha = mask_rotated[y1o:y2o, x1o:x2o]
    for c in range(0, 3):
        image[y1:y2, x1:x2, c] = (alpha * overlay_rotated[y1o:y2o, x1o:x2o, c] +
                                  (1 - alpha) * image[y1:y2, x1:x2, c])

def main():
    global current_sunglass_index  # Ensure current_sunglass_index is accessible globally within main()
    
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame for faster processing
        frame = imutils.resize(frame, width=600)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        rects = detector(gray, 0)

        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # Extract the coordinates for the eyes
            (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
            (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
            leftEyePts = shape[lStart:lEnd]
            rightEyePts = shape[rStart:rEnd]

            # Compute the center of mass for each eye
            leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
            rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

            # Calculate the angle between the eye centroids
            dY = rightEyeCenter[1] - leftEyeCenter[1]
            dX = rightEyeCenter[0] - leftEyeCenter[0]
            angle = np.degrees(np.arctan2(dY, dX)) - 180

            # Calculate the distance between the eyes for the desired sunglass width
            dist = np.sqrt((dX ** 2) + (dY ** 2))
            desiredWidth = int(dist * 2.3)  # Adjust multiplier for better fit

            # Resize the current sunglasses image to fit the width between the eyes
            sunglass_img_resized = imutils.resize(sunglass_imgs[current_sunglass_index], width=desiredWidth)

            # Calculate the position for overlaying the sunglasses
            mid_point = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
                         (leftEyeCenter[1] + rightEyeCenter[1]) // 2)

            x = mid_point[0] - (sunglass_img_resized.shape[1] // 2)
            y = mid_point[1] - (sunglass_img_resized.shape[0] // 2)

            # Overlay the sunglasses
            overlay(frame, sunglass_img_resized, x, y, angle)

        cv2.imshow("Frame", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):  # Press 'n' key to switch to the next sunglass image
            current_sunglass_index = (current_sunglass_index + 1) % len(sunglass_imgs)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
