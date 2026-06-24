.PHONY: debug
debug:
	python src/debug.py

.PHONY: train
train:
	python src/train.py --n-envs 3 

.PHONY: record
record:
	ffmpeg -f x11grab -r 30 -s 800x600 -i :99.0 -pix_fmt yuv420p output.mp4

.PHONY: stats
stats:
	tensorboard --logdir tensorboard/

bin/speedhack.so: speedhack.c
	mkdir -p bin
	gcc -shared -fPIC -o bin/speedhack.so speedhack.c -ldl

.PHONY: build
build: bin/speedhack.so

.PHONY: clean
clean:
	rm -rf tensorboard/
	rm -rf *.zip
	rm -rf *.mp4
	rm -rf *.png
	rm -rf best_runs/
	rm -rf checkpoints/
	rm -rf bin/
