Docker
===

```console
docker build . -t serenade
docker run -ti --rm \
    -p 8080:8080 \
    -v $HOME/workspace/serenade-2021/code/datasets/rsc15-clicks_train_full.txt:/data/train.txt \
    -e TRAINING_DATA=/data/train.txt \
    serenade
```

With custom application configuration file:

```console
docker run -ti --rm \
    -p 8080:8080 \
    -v $PWD/config.toml:/.config/serenade_serving_config.toml \
    serenade
```
