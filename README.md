# Dual-Domain Synergy: Enhancing Low-Light Images with Kolmogorov-Arnold Networks

This is the official implementation of the manuscript Dual-Domain Synergy: Enhancing Low-Light Images with Kolmogorov-Arnold Networks submitted to The Visual Computer.

![Fig 1](https://github.com/CrazyAn-JL/DAWN/blob/main/DAWN.png)

## Weights

All the weights that we trained on different datasets is available at [[Baidu Pan](https://pan.baidu.com/s/1Zk1JEZDWc8AI-F4djaPAPQ)] (code: `CrAn`).

## 1. Create Conda Environment

```bash
conda create --name DAWN python=3.10.16
conda activate DAWN
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 3. Training
You can modify the dataset configuration you want to train in the ./data/options.py file.
```bash
python train.py
```

## 4. Testing
You can select the dataset you want to test and simply enter the corresponding command, for example: --lol, --lol_v2_real, etc.

```bash
python eval.py --lol
```

Evaluation metrics

```bash
python measure.py --lol
```

Evaluation no-reference metrics

```bash
python measure_no_re.py --lol
```
