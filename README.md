# Fine-Grained Spatiotemporal \\Air Quality Forecasting \\Guided by Chemical Transport Modeling

<p align="center">
  <img src="figure/model.png" alt="MAPENet Architecture" width="700"/>
</p>

This porject contains the implementation of MAPENet in PyTorch.

## Directory Structure
```text
MAPENet/       
├── data                    # Original data
├── dataset
├── logs
├── weights 
├── model                   # Main structure of MAPENet
├── cfg.py                 
├── flow.py                 # Train / validation / test flow logic
├── pipeline.py             # Main script
├── utils.py                
└── README.md       
```

## How to Run the Code
Use the script below to preprocess pollutant `.csv` files into model-ready tensors including GT pollutant tensor `P` and RBF-interpolated pollutant input `X`:

```bash
python data/air_quality/get_X_and_P.py
```

Then run the code below to train and evaluate
```bash
python pipeline.py ODETAU cuda:0 --il 12 --ol 4
```

