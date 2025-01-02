from flask import Flask, request, jsonify, render_template
import base64
from io import BytesIO
from PIL import Image
from ultralytics import YOLO
from flask_cors import CORS
import os


# Flaskアプリのセットアップ
app = Flask(__name__)
CORS(app)  # CORSを有効化

# YOLOモデルのロード
model = YOLO("yolo11n.pt")

# トップページのルートを追加
@app.route("/")
def index():
    return render_template("index.html")  # フロントエンドのHTMLを返す

# 物体検出エンドポイント
@app.route("/detect", methods=["POST"])
def detect():
    # 画像データを受信
    data = request.get_json()
    image_data = data["image"]

    # Base64デコードして画像を読み込む
    image_data = image_data.split(",")[1]  # "data:image/jpeg;base64,"を取り除く
    image = Image.open(BytesIO(base64.b64decode(image_data)))

    # YOLOで物体検出
    results = model(image)
    detections = results[0].boxes

    # 検出結果をテキスト化
    text_results = []
    for box in detections:
        cls = box.cls.cpu().numpy()[0]
        conf = box.conf.cpu().numpy()[0]
        class_name = model.names[int(cls)]
        text_results.append(f"{class_name} ({conf:.2f})")

    # 結果を返す
    return jsonify({"text": ", ".join(text_results)})

# アプリを起動
if __name__ == "__main__":
    # SSL証明書は開発環境以外では不要。環境変数で動作モードを切り替える
    SSL_CERT = os.getenv("SSL_CERT_PATH", None)
    SSL_KEY = os.getenv("SSL_KEY_PATH", None)

    if SSL_CERT and SSL_KEY:
        app.run(ssl_context=(SSL_CERT, SSL_KEY), host="0.0.0.0", port=5000)
    else:
        app.run(host="0.0.0.0", port=5000)