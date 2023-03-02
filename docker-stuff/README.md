# Docker guide
How to build and run docker. Start by `cd` to your project location. Then specify the file path to your Dockerfile with the `-f` flag:
```
docker build --tag <name> -f docker-stuff/Dockerfile .
docker run --rm --gpus all -it <name> bash
```

Where `--tag` defines the `<name>` you give your container. `--rm` means the container will be removed when we exit the docker. `--gpus all` allows the docker GPU usage. `-it` means it will open an 'interactive terminal' in `bash`.

When you are in the docker you can run the training script:
```
python3 train.py <arguments>
python3 train.py -epochs 100 -id 0
```
And can specify the following arguments:
1. `-epochs`: number of epochs run
2. `-encoder-depth`: number of depth layers in the encoder part of the network
3. `-lr`: learning rate used during training
4. `-batch_size`: batch size used for GD
5. `-l2`: l2 penalization used during optimization
6. `-id`: just an id for how to save the `.csv` file. It looks like: `results{id}.csv`

To save the data on host machine run:
```python
cd docker-stuff
python copy_data.py
```