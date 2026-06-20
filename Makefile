.PHONY: debug
debug:
	python src/debug.py

.PHONY: train
train:
	python src/train.py --n-envs 2 

.PHONY: record
record:
	ffmpeg -f x11grab -r 30 -s 800x600 -i :99.0 -pix_fmt yuv420p output.mp4

.PHONY: stats
stats:
	tensorboard --logdir tensorboard/

.PHONY: clean
clean:
	rm -rf tensorboard/
	rm -rf *.zip
	rm -rf *.mp4
	rm -rf *.png
	rm -rf best_runs/
	rm -rf checkpoints/
