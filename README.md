# Deep Learning with Pytorch


**Image Classifier Project.ipynb:** In this project, a deep learning network is developed to recognize 102 species of flowers from the dataset [1]. A pre-trained network [2] is used and a new classifier is built. The final accuracy was 82%. 

**train.py:** Python script that run from the command line. Will train a new network on a dataset and save the model as a checkpoint. 
  * Basic usage: `python train.py data_directory`
  Prints out training loss, validation loss, and validation accuracy as the network trains
  Options:
    - Set directory to save checkpoints: `python train.py data_dir --save_dir save_directory`
    - Choose architecture: `python train.py data_dir --arch "vgg13"`
    - Set hyperparameters: `python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20`
    - Use GPU for training: `python train.py data_dir --gpu`

**predict.py:** Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the flower name and class probability.

  * Basic usage: `python predict.py /path/to/image checkpoint`
  Options:
    - Return top KK most likely classes: `python predict.py input checkpoint --top_k 3`
    - Use a mapping of categories to real names: `python predict.py input checkpoint --category_names cat_to_name.json`
    - Use GPU for inference: `python predict.py input checkpoint --gpu`

[1]: http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html 
[2]: https://arxiv.org/abs/1409.1556
