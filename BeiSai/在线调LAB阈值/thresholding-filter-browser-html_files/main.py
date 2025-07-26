from maix import camera, display, touchscreen, app, image, network
from flask import Flask, Response
from flask_cors import CORS
from werkzeug.serving import make_server
import threading as th
import os

cam = camera.Camera(552,368)
disp = display.Display()
ts = touchscreen.TouchScreen()

Flask_app = Flask(__name__)  # Flask app
CORS(Flask_app)

class Server():
    def __init__(self) -> None:
        self.image_event = th.Event()
        Flask_app.route("/stream")(self.http_stream)

    def stream_worker(self):
        while self.running:
            self.img = cam.read()
            self.jpg=self.img.to_jpeg(90)
            if self.jpg is None:
                print("Encoding error!")    
                os._exit(-1)
            self.image_event.set()

    def get_stream(self):
        while self.running:
            self.image_event.wait()
            self.image_event.clear()
            if self.jpg is None:
                continue
            yield (
                b"Content-Type: data/jpeg\r\n\r\n"
                + self.jpg.to_bytes(False)
                + b"\r\n\r\n--frame\r\n"
            )

    def http_stream(self):
        return Response(
            self.get_stream(), mimetype="multipart/x-mixed-replace; boundary=frame"
        )

    def start_server(self):
        self.running = True
        self.stream_thread = th.Thread(target=self.stream_worker, daemon=True)
        self.stream_thread.start()
        self.server = make_server("0.0.0.0", 8080, Flask_app, threaded=True, processes=1)
        self.server.log_startup()
        self.server_thread = th.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()

    def stop_server(self):
        self.running = False
        self.server.shutdown()
        self.server_thread.join()
        self.stream_thread.join()

wifi = network.wifi.Wifi()

server=Server()
server.start_server()

#print(image.string_size("[OFF]",scale=4))
#print("ip:", wifi.get_ip())

disp_switch=False
while not app.need_exit():

    if disp_switch:
        show_img=server.img.draw_string(10, 20, "[EXIT]", color=image.Color.from_rgb(255, 0, 0), scale=4)
        show_img=show_img.draw_string(210, 20, "[OFF]", color=image.Color.from_rgb(0, 0, 0), scale=4)
        
    else:
        black_img=image.Image(disp.width(), disp.height(), image.Format.FMT_RGB888)
        show_img=black_img.draw_string(10, 20, "[EXIT]", color=image.Color.from_rgb(255, 0, 0), scale=4)
        show_img=show_img.draw_string(375, 20, "[ON ]", color=image.Color.from_rgb(255, 255, 255), scale=4)

    show_img=show_img.draw_string(10, 80, wifi.get_ip()+":8080/stream", color=image.Color.from_rgb(17, 255, 60), scale=2)
    disp.show(show_img)
    touch = ts.read()
    if 10<=touch[0]<=185 and 20<=touch[1]<=57 and touch[2]:
        server.stop_server()
        break
    elif 375<=touch[0]<=540 and 20<=touch[1]<=57 and touch[2] and not disp_switch:
        disp_switch=not disp_switch
    elif 210<=touch[0]<=375 and 20<=touch[1]<=57 and touch[2] and disp_switch:
        disp_switch=not disp_switch

else:
    server.stop_server()