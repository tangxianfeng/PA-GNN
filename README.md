# Transferring Robustness for Graph Neural Network Against Poisoning Attacks
Implementation of paper "[Transferring Robustness for Graph Neural Network Against Poisoning Attacks](https://arxiv.org/abs/1908.07558)"

by Xianfeng Tang, Yandong Li, Yiwei Sun, Huaxiu Yao, Prasenjit Mitra, Suhang Wang  
Published at WSDM 2020, Houston, Texas, USA

## Requirements
* Python 3.7 or newer
* `numpy`
* `tensorflow`
* `scipy`

## Before running
Please download [data.zip](https://drive.google.com/file/d/1N_tA-T8Ldw69PPRz8MKGwXBVYbL_-6--/view?usp=sharing) and extract all contents to `data/`.

## Run the code
Please run with `python main.py`.

## Contact
Please contact tangxianfeng at outlook.com for any questions.

## References
### Dataset
#### Pubmed
We acquire the processed graph from https://github.com/tkipf/gcn/tree/master/gcn/data and put them in `data/gcn_data` (must be unzip from `data.zip` to find it). The original datasets can be found here: http://linqs.cs.umd.edu/projects/projects/lbc/.

#### Reddit
The original Reddit graph can be found here: http://snap.stanford.edu/graphsage/.

#### Yelp small & large
We use [Yelp Dataset](https://www.yelp.com/dataset) to compile these two datasets.

### Code & model design
#### Meta-learning
The design and implenmentation of meta-learning part is inspired by [MAML-TensorFlow](https://github.com/dragen1860/MAML-TensorFlow) and [maml](https://github.com/cbfinn/maml).

#### Graph neural networks
The design of neural networks is inspired by [gcn](https://github.com/tkipf/gcn) and [metattack](https://github.com/danielzuegner/gnn-meta-attack).

#### Graph adversarial attacks
We adopt [nettack](https://github.com/danielzuegner/nettack) and [metattack](https://github.com/danielzuegner/gnn-meta-attack).

## Cite
Please cite our paper if the model or the paper help:

```
@inproceedings{tang2020transferring,
	title = {Transferring Robustness for Graph Neural Network Against Poisoning Attacks},
	author={Tang, Xianfeng and Li, Yandong and Sun, Yiwei and Yao, Huaxiu and Mitra, Prasenjit and Wang, Suhang},
	booktitle={ACM Internatioal Conference on Web Search and Data Mining (WSDM)},
	year = {2020}
}
```
