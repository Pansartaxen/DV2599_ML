import cv2
from PIL import Image
from numpy import asarray


# Open the camera
cap = cv2.VideoCapture(0)

capturing = True

# Capture frames from the camera
while capturing:
    # Read the next frame from the camera
    ret, frame = cap.read()

    # Check if the frame was successfully read
    if ret:
        # Display the frame
        cv2.imshow('Camera', frame)

        # Check if the user pressed the 'q' key

        if cv2.waitKey(1) & 0xFF == ord('q'):
            capturing = False

        # Check if the user pressed the 'SPACE' key
        if cv2.waitKey(1) & 0xFF == ord(' '):
            # Save the image
            cv2.imwrite('HandsignInterpreter\latest.png', frame)

# Release the camera
cap.release()
cv2.destroyAllWindows()

def png_to_array(path):
    png = Image.open(path)
    png_array = asarray(png)

    print(png_array[0][0])
    print(png_array.shape)

if __name__ == "__main__":
    png_to_array('HandsignInterpreter/test.png')
