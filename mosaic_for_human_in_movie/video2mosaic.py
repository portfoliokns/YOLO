import cv2 as cv2
import numpy as np
from moviepy.editor import VideoFileClip

# モザイク処理
def mosaic(img, x, y, w, h, size):

    # 画像の幅と高さを取得
    h_img, w_img = img.shape[:2]

    # モザイク範囲が画像の境界を超えないように調整
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w_img, x + w)
    y2 = min(h_img, y + h)

    # 調整された領域でスライス
    img_rec = img[y1:y2, x1:x2]

    # モザイク処理：縮小→拡大
    mosaic_w = max(1, x2 - x1)
    mosaic_h = max(1, y2 - y1)

    img_small = cv2.resize(img_rec, (size, size), interpolation=cv2.INTER_LINEAR)
    img_mos = cv2.resize(img_small, (mosaic_w, mosaic_h), interpolation=cv2.INTER_AREA)

    # 画像にモザイク画像を重ねる
    img_out = img.copy()
    img_out[y1:y2, x1:x2] = img_mos

    return img_out

# YOLOの初期設定
def readYOLO():

    # YOLOの重みと設定ファイル、クラス名のリストを読み込む
    weights_path = '../setting/yolov3.weights'
    config_path = '../setting/yolov3.cfg'
    class_names_path = '../setting/coco.names'

    # クラス名の読み込み
    with open(class_names_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    # YOLOの読み込み
    net = cv2.dnn.readNet(weights_path, config_path)

    return net, classes

# YOLOの読み込み
net, classes = readYOLO()

# 動画の読み込み
input_video = "./test.mov"  #モザイク処理したい動画
cap = cv2.VideoCapture(input_video)

# 保存用のVideoWriterオブジェクトを作成
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
output_file = 'output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v'はMP4形式
out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

# モザイク加工
mosaic_para = 40
while True:
    ret, frame = cap.read()

    if not ret:
        break

    # 画像をBlobに変換
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    # 出力レイヤーの取得
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # 物体検出を実行
    outputs = net.forward(output_layers)

    # 検出結果を解析してモザイク加工
    for output in outputs:
        for detection in output:
            scores = detection[5:]  # クラスのスコア
            class_id = np.argmax(scores)  # 最も確率の高いクラス
            confidence = scores[class_id]  # 確率
            if confidence > 0.5 and class_id == 0:  # 信頼度が0.5以上かつ人間の場合
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                w = int(detection[2] * frame_width)
                h = int(detection[3] * frame_height)

                # ボックスの座標を計算
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # 描画
                frame = mosaic(frame, x, y, w, h, mosaic_para)

    out.write(frame)

cv2.destroyAllWindows()
cap.release()
out.release()

# 音声の追加(MoviePyを使って音声を抽出し、映像に結合)
original_video = VideoFileClip(input_video)
audio_clip = original_video.audio  # 元の動画から音声を抽出
new_video = VideoFileClip(output_file)  # OpenCVで作成した動画を読み込み

# 音声付きで新しい動画を書き出す
final_output = 'output_video_audio.mp4'
final_video = new_video.set_audio(audio_clip)
final_video.write_videofile(final_output, codec='libx264', audio_codec='aac', fps=original_video.fps)
