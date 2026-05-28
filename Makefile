.PHONY: run
run:
	python main.py

.PHONY: record
record:
	ffmpeg -f x11grab -r 30 -s 1920x1080 -i :99.0 -pix_fmt yuv420p output.mp4

.PHONY: watch
watch:
	ffmpeg -f x11grab -r 15 -s 800:600 -i :99.0 \
      -c:v libx264 -preset ultrafast -tune zerolatency \
      -f mpegts tcp://0.0.0.0:8080?listen
	# chromium-browser view.html &
	# ffmpeg -f x11grab -r 15 -s 800x600 -i :99.0 \
 #      -c:v mjpeg -q:v 5 -f mpjpeg \
 #      -listen 1 http://0.0.0.0:8080 &

