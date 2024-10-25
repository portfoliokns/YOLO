import cv2
import numpy as np

# YOLOの重みと設定ファイル、クラス名のリストを読み込む
weights_path = '../setting/yolov3.weights'
config_path = '../setting/yolov3.cfg'
class_names_path = '../setting/coco.names'

# クラス名の読み込み
with open(class_names_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# YOLOネットワークの読み込み
net = cv2.dnn.readNet(weights_path, config_path)

# 画像の読み込み
image = cv2.imread('test.png')  # ここに検出したい画像のパスを指定
height, width = image.shape[:2]

# 画像をBlobに変換
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)

# 出力レイヤーの取得
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# 物体検出を実行
outputs = net.forward(output_layers)

# 検出された物体の情報を格納するリスト
boxes = []
confidences = []
class_ids = []

# 検出結果を解析
for output in outputs:
    for detection in output:
        scores = detection[5:]  # クラスのスコア
        class_id = np.argmax(scores)  # 最も確率の高いクラス
        confidence = scores[class_id]  # 確率
        if confidence > 0.5:  # 信頼度が0.5以上の場合
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # ボックスの座標を計算
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# 非最大抑制を適用
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# 検出された物体に対してボックスを描画
for i in range(len(boxes)):
    if i in indexes:
        class_id = class_ids[i]
        x, y, w, h = boxes[i]
        label = str(classes[class_id])
        confidence = confidences[i]
        color = (0, 255, 0)  # 緑色
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)  # ボックスを描画
        cv2.putText(image, f"{label}: {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# 結果を表示
cv2.imshow('Image Detection', image)
cv2.waitKey(0)
cv2.imwrite("new.png",image)
cv2.destroyAllWindows()