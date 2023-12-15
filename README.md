# DATIS
This is the implementation repository of paper : Distance-Aware Test Input Selection for Deep Neural Networks
## Description

We propose a novel test input selection approach for DNNs, which is called DATIS.

The key design idea of DATIS is two-fold:

1. Derive improved uncertainty scores for test inputs by leveraging the training data of DNNs, thus achieving better test input selection.
2. Eliminate the redundancies among the selected test input sets based on the distances among the selected test inputs to further enhance the effectiveness of test input selection.

## Reproducibility

### Environments

```tx
tensorflow==2.12.0
tensorflow-estimator==2.12.0
Keras==2.12.0
numpy==1.22.0
scikit-learn==1.2.2
tqdm==4.65.0
datasets=2.12.0
```

## Structure

```
├── cluster_data/ "the results of fault estimation in DNNs"
├── compared approach/ "code of compared test select approaches"
├── corrupted_data/ "corrupted candidate dataset"
├── DATIS/ "implementation of DATIS"
├── model/ "DNNs in our experiments"
├── results/  "pictures and tables of experimental results"
├── mnist_test_selection.py "a demo of test selection in mnist"
├── mnist_dnn_enhancement.py "a demo of dnn enhancement in mnist"
```

## Usage

We prepared a demo for DATIS

- `python mnist_test_selection.py`
- `python mnist_dnn_enhancement.py`

If you want to  run our demo:

1.  download the `corrupted_data` and `model` files by following this link: 

   link：  https://1drv.ms/f/s!Are_aZdXk1FyhUA-M4c6A6rzf_LT?e=hIa4yT

2. experiment

   - `python mnist_test_selection.py`

   ​        a demo for test selection in mnist dataset with LeNet5 model

   - `python mnist_dnn_enhancement.py`

      a demo for dnn enhancement in mnist dataset with LeNet5 model

## Nominal Dataset 

**2 image datasets**

- CIFAR-100 (a 100-class ubiquitous object dataset) [1]
- MNIST (a handwritten digit dataset) [2]

**2 text datasets**

- TREC (a question classification dataset) [3]
- IMDB (a large movie review dataset for binary sentiment classification) [4]





[1] CIFAR http://www.cs.toronto.edu/~kriz/cifar.html

[2] MNIST http://yann.lecun.com/exdb/mnist/

[3] TREC https://cogcomp.seas.upenn.edu/Data/QA/QC/

[4] IMDB http://ai.stanford.edu/~amaas/data/sentiment/


