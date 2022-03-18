# DL_3-DTransport
Deep U-Net Neural Network from the paper “Approximating Three-dimensional (3-D) Transport of Atmospheric Pollutants via Deep Learning” in Pytorch is built for replacing and accelerating the transport procedure in the chemical transport models.


# Data
The dataset consisted of meteorological field, emission inventory, and concentrations of pollutant. Meteorological field was estimated by the Weather Research and Forecasting (WRF) model with the initial conditions and observation dataset from the National Centers for Environmental Prediction (NCEP). Emission inventory was derived from the Multi-resolution Emission Inventory for China (MEIC v1.3, 0.25° × 0.25°, 2016, http://www.meicmodel.org) and interpolated to the required 12 km *12 km horizontal resolution. A 3-D CTM numerical model of the Community Multiscale Air Quality (CMAQ) (https://github.com/USEPA/CMAQ/tree/5.3.1) was selected to simulate the concentration evolution with the meteorological field and emission inventory, as well as the benchmark for computation efficiency comparison for the deep learning surrogate.The retrieved training and validation datasets are too large (>1T) to be uploaded for share. Anyone who is interested in them is welcomed to contact the corresponding author for a point-to-point transfer.


<img width="500" alt="image" src="https://user-images.githubusercontent.com/44837025/158928059-dc5264e8-beb6-4615-8505-85dbe23eef96.png">



# Requirements
Numpy
Pytorch
