# Document Classification with CNN-Char, GRNN and HAN
In the current technological context, the Documents Classification is a fundamental task of Natural Language Processing (NLP), it assigns a class or category to a document, making it easier to manage and order.
This software implements some Document Classification techniques, based on supervised learning. Replicating some of the works of the last decade using Convolutional Neural Networks, Recurrent Neural Networks and Attention Mechanisms. In particular, the following models are used:
- ***CNN-Char*** by Zhang et al. [1], which use a traditional deep learning technique for text classification, namely a character-level Convolutional Network.
- ***GRNN*** by Tang et al. [2], which develops a model with a hierarchical structure (via CNN or LSTM) and based on a module called Gated Recurrent Neural Network
- ***HAN*** by Yang et al. [3] which deals a model with hierarchical structure and attention mechanisms.
## Results
The three models are evaluated on 3 different datasets (two of Sentiment Analysis and one of Topic Classification):
- **Yelp**, a dataset of reviews with a rating of 1 to 5 stars ([here](https://www.yelp.com/dataset.)). in particular:
  - Yelp 2013
  - Yelp 2014
- **Yahoo! Answers**, a dataset containing title, question and best answer on generic topics divided into 10 classes: Society and Culture, Science and Mathematics, Health, Education and Consultation, Computers and Internet, Sports, Business and Finance, Entertainment and Music, Family and Relationships and Politics and Government ([here](https://www.kaggle.com/soumikrakshit/yahoo-answers-dataset)).

And the results obtained in terms of accuracy are similar to the works cited:
| Model | Yelp2013 | Yelp2014 | Yahoo!Answers |
| :---: | :---: | :---: | :---: | 
| CNN-Char | 61.0 | 64.8 | 67.8 | 
| GRNN-LSTM | 63.7 | 65.9 | 68.6 | 
| HAN | 68.1 | 70.1 | 71.5 | 

An example of Visulization of Attention in HAN on the Yelp2013 dataset can be seen below. A 5-star review at the top and a 1-star review at the bottom. (blue represents most relvant words and red most important sentences)

<p align="center">
 <img src="https://github.com/loredeluca/MachineLearning/blob/main/results/5star.jpg" width=" 1000" height=auto>
  
<img src="https://github.com/loredeluca/MachineLearning/blob/main/results/1star.jpg" width=" 1000" height=auto>
</p>

## Getting Started
After cloning the repository, install the dependencies 
<!--- 
with the command line:
```
$ pip install -r requirements.txt
```
--->
Then setup the following file structure within the data directory:
```tree
.
├── datasets
    ├── yelp2013
        ├── train.csv
        ├── val.csv
        └── test.csv
    ├── yelp2014
        ├── train.csv
        ├── val.csv
        └── test.csv
    └── yahoo
        ├── train.csv
        ├── val.csv
        └── test.csv
├── grnn_data
└── han_data
```
<!--
Install dependencies:
- numpy (1.13.3)
- pandas (0.22.0)
- tqdm (4.60.0)
- nltk (3.5)
- regex (2021.4.4)
- torch (1.4.0)
- Pillow (5.1.0)-->

## How to make it works
Run with 
```
$ python3 main.py
```
***Parameter:*** `--model: 0='cnn-char', 1='grnn', 2='han'. default=0`

On the first run it will preprocess the data and create the Word2Vec model (for *grnn* and *han*) and save them in the `model_data`. Then the training and testing begins (a checkpoint is generated after the training).

NB: it's strongly recommended to use a GPU to train the models

### References
[1] Xiang Zhang, Junbo Zhao and Yann LeCun. 2015. [Character-level convolutional networks for text classification](https://proceedings.neurips.cc/paper/2015/hash/250cf8b51c773f3f8dc8b4be867a9a02-Abstract.html). In Advances in Neural Information Processing Systems 28: Annual Conference on Neural Information Processing Systems, pages 649–657.

[2] Duyu Tang,Bing Qin and Ting Liu. 2015. [Document modeling with gated recurrent neural network for sentiment classification](https://www.aclweb.org/anthology/D15-1167/). In Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing, pages 1422–1432.

[3] Zichao Yang, Diyi Yang, Chris Dyer, Xiaodong He, Alex Smola, and Eduard Hovy. 2016. [Hierarchical attention networks for document classification](https://www.aclweb.org/anthology/N16-1174/). In Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Lan- guage Technologies, pages 1480–1489.
