import base64
import cv2
import zmq
import numpy as np

context = zmq.Context()
footage_socket = context.socket(zmq.PUB)
footage_socket.connect('tcp://127.0.0.1:5555')

camera = cv2.VideoCapture(0)  # init the camera

weights = '/Users/domantasskuja/Documents/GitHub/Python/yolov3-tiny.weights'
config = '/Users/domantasskuja/Documents/GitHub/Python/yolov3-tiny.cfg'


def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers



def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])
    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)



while True:
    try:
        grabbed, frame = camera.read()  # grab the current frame
        frame = cv2.resize(frame, (640, 480))  # resize the frame
        

        Width = frame.shape[1]
        Height = frame.shape[0]
        scale = 0.00392

        classes = None

        with open('/Users/domantasskuja/Documents/GitHub/Python/yolov3.txt', 'r') as f:
            classes = [line.strip() for line in f.readlines()]

        COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

        net = cv2.dnn.readNet(weights, config)

        blob = cv2.dnn.blobFromImage(frame, scale, (416,416), (0,0,0), True, crop=False)

        net.setInput(blob)

        outs = net.forward(get_output_layers(net))

        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4


        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])


        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        for i in indices:
            i = i
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            draw_prediction(frame, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))


        encoded, buffer = cv2.imencode('.jpg', frame)
        jpg_as_text = base64.b64encode(buffer)
        footage_socket.send(jpg_as_text)

    except KeyboardInterrupt:
        camera.release()
        cv2.destroyAllWindows()
        break