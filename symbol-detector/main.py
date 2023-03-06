import cv2
import numpy as np
from keras.models import Sequential, load_model

draw = False

model = load_model("mnist.h5")


def draw_callback(event, x, y, flags, param):
    global draw
    if event == cv2.EVENT_MOUSEMOVE:
        if draw:
            cv2.circle(canvas, (x, y), 10, 200, -1)
    elif event == cv2.EVENT_LBUTTONDOWN:
        draw = True
    elif event == cv2.EVENT_LBUTTONUP:
        draw = False


canvas = np.zeros((512, 512), dtype="uint8")

cv2.namedWindow("MNIST")
cv2.setMouseCallback("MNIST", draw_callback)

while True:
    cv2.imshow("MNIST", canvas)

    key = cv2.waitKey(1) & 0xFF

    if key == 13:  # Press Enter to recognize
        img = cv2.resize(canvas.copy(), (28, 28), interpolation=cv2.INTER_AREA)
        img = img / 255.
        img = img.reshape(1, 28, 28, 1)
        predictions = model.predict(img)
        print(f"Result: {np.argmax(predictions)}")
    elif key == ord('c'):  # Press C to clear
        canvas[:] = 0

    if key == 27:
        break

cv2.destroyAllWindows()
