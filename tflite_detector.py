import tensorflow as tf
import cv2
import numpy as np

interpreter = None

def load_model(model_path):
    global interpreter
    interpreter = tf.lite.Interpreter(model_path)
    interpreter.allocate_tensors()

def inference(image):
    input_details = interpreter.get_input_details()
    input_width = input_details[0]['shape'][2]
    input_height = input_details[0]['shape'][1]

    input_image = cv2.resize(image, (input_width, input_height))
    input_image = np.float32(input_image[None, ...])
    # input_image = input_image /127.5 - 1
    
    interpreter.set_tensor(input_details[0]['index'], input_image)

    interpreter.invoke()
    output_details = interpreter.get_output_details()
    rects = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    return rects, classes, scores