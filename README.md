# Cargo sorting system

The execution steps are as follows

1. Download data
> python3 data_download.py

After the download is completed, you will get the data/FashionMNIST folder. This data set contains various clothing types.

After the download is complete, you can visualize the data in the visualization/data_analysis.ipynb file.

2. Start training the model
In the train.py file, you can specify model_type for training. Two models are implemented here, lenet and vgg.

> python3 train.py

After training, pth and json files will be output under the results/ folder. The former is the weight of the optimal classification model, and the latter is the curve record of the training process.

Open visualization/show_result.ipynb to visualize the curve of the training process.

3. UI interface visualization

> python3 main.py
