.PHONY: train
train:
	python src/train.py --n-envs 1 --total-timesteps 10000

.PHONY: run
run:
	python src/main.py

.PHONY: record
record:
	ffmpeg -f x11grab -r 30 -s 1920x1080 -i :99.0 -pix_fmt yuv420p output.mp4

.PHONY: stats
stats:
	tensorboard --logdir tensorboard/

.PHONY: clean
clean:
	rm -rf tensorboard/
	rm -rf *.zip
	rm -rf *.mp4
	rm -rf *.png
