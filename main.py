from collections import Counter
from time import sleep
import cv2
import numpy as np
from keras.models import load_model
from skimage.transform import resize, pyramid_reduce
import PIL
from PIL import Image

model = load_model("model.h5")  # load pre-trained model
output_path = "output.txt"  # path of the output file


def prediction_to_char(pred):
    """
    Convert the prediction to ASCII character

    Parameters
    ----------
    pred: integer
        an integer valued between 0 to 25 that indicates the prediction

    Returns
    -------
    char
        an ASCII character
    """

    return chr(pred + 65)


def predict(model, image):
    """
    predicts the character from input image

    Parameters
    ----------
    model: a keras model instance
        pre-trained CNN model saved in HDF5 format
    image: OpenCV Image
        image of hand captured from camera

    Returns
    -------
    float
        probability of the predicted output
    integer
        an integer valued between 0 to 25 that represents the prediction
    """

    data = np.asarray(image, dtype="int32")
    pred_probab = model.predict(data)[0]
    pred_class = list(pred_probab).index(max(pred_probab))
    return max(pred_probab), pred_class


def crop_image(image, start_x, start_y, width, height):
    """
    Crops an image

    Parameters
    ----------
    image: image
        the image that has to be cropped
    start_x: integer
        x co-ordinate of the starting point for cropping
    start_y: integer
        y co-ordinate of the starting point for cropping
    width: integer
        expected width of the cropped image
    height: integer
        expected height of the cropped image
    
    Returns
    -------
    """

    return image[start_y : start_y + height, start_x : start_x + width]


def show_pred(image, pred, probab):

    # background
    cv2.rectangle(image, (0, 0), (200, 100), (0, 0, 0), cv2.FILLED)

    # prediction
    cv2.putText(
        image,
        pred,
        (20, 40),
        cv2.FONT_HERSHEY_COMPLEX_SMALL,
        2.0,
        (0, 255, 0),
        lineType=cv2.LINE_8,
    )

    # prediction probability
    cv2.putText(
        image,
        probab,
        (20, 80),
        cv2.FONT_HERSHEY_COMPLEX_SMALL,
        2.0,
        (0, 0, 255),
        lineType=cv2.LINE_AA,
    )

    return image


def show_nothing(image):

    # background
    cv2.rectangle(image, (0, 0), (200, 100), (0, 0, 0), cv2.FILLED)

    # nothing
    cv2.putText(
        image,
        "null",
        (20, 50),
        cv2.FONT_HERSHEY_COMPLEX_SMALL,
        2.0,
        (255, 255, 255),
        lineType=cv2.LINE_AA,
    )
    return image


def show_written(image, pred):

    # background
    cv2.rectangle(image, (0, 0), (300, 100), (0, 255, 0), cv2.FILLED)

    # written
    cv2.putText(
        image,
        "written " + pred,
        (20, 40),
        cv2.FONT_HERSHEY_COMPLEX_SMALL,
        2.0,
        (0, 0, 0),
        lineType=cv2.LINE_AA,
    )
    return image


def validate(lyst):
    """
    Validates if the prediction should be written to the file
    
    This function counts the outputs stored in the list, if the
    most predicted output occurs more than 30 times, it returns True,
    otherwise returns False

    Parameters
    ----------
    lyst: list
        a list that contants recently predicted outputs
    
    Returns
    -------
    boolean
        True if the output should be stored to files, otherwise False
    """

    if Counter(lyst).most_common(1)[0][1] > 30:
        return True
    return False


def write_newline():
    """
    Writes a newline to the output file
    """

    with open(output_path, "a") as f:
        f.write("\n")
    f.close()



if __name__ == "__main__":
    lyst = []

    while True:

        validated = False

        cam_capture = cv2.VideoCapture(0)
        _, image_frame = cam_capture.read()

        hand_image = crop_image(image_frame, 300, 300, 300, 300)
        image_grayscale = cv2.cvtColor(hand_image, cv2.COLOR_BGR2GRAY)

        image_grayscale_blurred = cv2.GaussianBlur(image_grayscale, (15, 15), 0)
        hand_image = cv2.resize(
            image_grayscale_blurred, (28, 28), interpolation=cv2.INTER_AREA
        )

        hand_image = np.resize(hand_image, (28, 28, 1))
        hand_image = np.expand_dims(hand_image, axis=0)

        pred_probab, pred_class = predict(model, hand_image)

        cv2.rectangle(image_frame, (300, 300), (600, 600), (255, 255, 00), 2)

        if pred_probab >= 0.800:
            pred = prediction_to_char(pred_class)
            lyst.append(pred_class)

            if validate(lyst):
                lyst.clear()
                with open(output_path, "a") as f:
                    f.write(prediction_to_char(pred_class))
                f.close()
                image_frame = show_written(image_frame, pred)
                validated = True
            else:
                image_frame = show_pred(
                    image=image_frame, pred=pred, probab=str(pred_probab)[:5]
                )
        else:
            image_frame = show_nothing(image=image_frame)
            lyst = []

        cv2.imshow("frame", image_frame)

        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break
    cam_capture.release()
    cv2.destroyAllWindows()
    write_newline()

