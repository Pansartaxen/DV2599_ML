import cv2
from PIL import Image
from numpy import asarray
from algorithms import train_random_forest, classify_image

def png_to_array(path):
    png = Image.open(path)
    png_array = asarray(png)

    return png_array

def run_camera(clf):
    # Open the camera
    cap = cv2.VideoCapture(0)

    # Set the camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    info = """Press space to read a new letter,"""
    info2 = """c to clear text, q to quit"""
    text = ""

    # Loop until the user hits the 'q' key
    while True:
        # Read a frame from the camera
        ret, frame = cap.read()        

        # Check if the frame was successfully read
        if not ret:
            break

        # Wait for the user to hit a key
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    
        if key == ord('c'):
            text = ""

        # If the user pressed 'L', put some text on the frame
        if key == ord(' '):
            box = frame[80:276, 50:246]
            new_letter = classify_image(box, clf)
            text += new_letter
            print("Found letter",new_letter)
        
        cv2.putText(frame, info, (20, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 100), 2)
        cv2.putText(frame, info2, (20, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 100), 2)
        cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 2)
        cv2.rectangle(frame, (50, 80), (246, 276), (0, 255, 0), 2)


        # Show the frame
        cv2.imshow("Camera", frame)

    # Release the camera and destroy all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    #png_to_array('HandsignInterpreter/test.png')
    acc, clf = train_random_forest()
    run_camera(clf)

    # TODO
    # get the 28x28 image from the camera
    # make the image black and white
    # classify the image
