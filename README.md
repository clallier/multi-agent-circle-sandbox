# technical-exercice-carousel-v2

## Installation

```
pip install -r requirements.txt
```

## Usage

### Training

```
python code/maddpg/experiments/train.py --scenario circle_sandbox --max-episode-len 80 --num-episodes 5000 --save-rate 200 --save-dir ./test_circle_sandbox/
```


### Evaluation

```
python code/maddpg/experiments/train.py --scenario circle_sandbox --load-dir ./test_circle_sandbox/ --display
```

### More info about how to use the code

```
python code/maddpg/experiments/train.py --help
```

Scenario scripts should be defined in the `code/multiagent-particle-envs/multiagent/scenarios/` directory.


# Notes
tensorboard --logdir=D:\Corentin\test_technique_delfox\technical-exercise-carousel-v2-tf2\technical-exercice-carousel-v2-tf2\test_circle_sandbox