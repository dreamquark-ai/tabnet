# README

# TabNet : Attentive Interpretable Tabular Learning

This is a pyTorch implementation of Tabnet (Arik, S. O., & Pfister, T. (2019). TabNet: Attentive Interpretable Tabular Learning. arXiv preprint arXiv:1908.07442.) https://arxiv.org/pdf/1908.07442.pdf.

# Installation

You can install using pip by running:
`pip install tabnet`

If you wan to use it locally within a docker container:

`git clone git@github.com:dreamquark-ai/tabnet.git`

`cd tabnet` to get inside the repository

`make start` to build and get inside the container

`poetry install` to install all the dependencies, including jupyter

`make notebook` inside the same terminal

You can then follow the link to a jupyter notebook with tabnet installed.



GPU version is available and should be working but is not supported yet.

# How to use it?

The implementation makes it easy to try different architectures of TabNet.
All you need is to change the  network parameters and training parameters. All parameters are quickly describe bellow, to get a better understanding of what each parameters do please refer to the orginal paper.

You can also get comfortable with the code works by playing with the **notebooks tutorials** for adult census income dataset and forest cover type dataset.

## Network parameters

- input_dim : int

    Number of initial features of the dataset

- output_dim : int

    Size of the desired output. Ex :
    - 1 for regression task
    - 2 for binary classification
    -  N > 2 for multiclass classifcation

- nd : int

    Width of the decision prediction layer. Bigger values gives more capacity to the model with the risk of overfitting.
    Values typically range from 8 to 64.

- na : int

    Width of the attention embedding for each mask.
    According to the paper nd=na is usually a good choice.

- n_steps : int
    Number of steps in the architecture (usually between 3 and 10)

- gamma : float
    This is the coefficient for feature reusage in the masks.
    A value close to 1 will make mask selection least correlated between layers.
    Values range from 1.0 to 2.0
- cat_idxs : list of int

    List of categorical features indices.
- cat_emb_dim : list of int

    List of embeddings size for each categorical features.
- n_independent : int

    Number of independent Gated Linear Units layers at each step.
    Usual values range from 1 to 5 (default=2)
- n_shared : int

    Number of shared Gated Linear Units at each step
    Usual values range from 1 to 5 (default=2)
- virtual_batch_size : int

    Size of the mini batches used for Ghost Batch Normalization

## Training parameters

- max_epochs : int (default = 200)

    Maximum number of epochs for trainng.
- patience : int (default = 15)

    Number of consecutive epochs without improvement before performing early stopping.
- lr : float (default = 0.02)

    Initial learning rate used for training. As mentionned in the original paper, a large initial learning of ```0.02 ```  with decay is a good option.
- clip_value : float (default None)

    If a float is given this will clip the gradient at clip_value.
- lambda_sparse : float (default = 1e-3)

    This is the extra sparsity loss coefficient as proposed in the original paper. The bigger this coefficient is, the sparser your model will be in terms of feature selection. Depending on the difficulty of your problem, reducing this value could help.
- model_name : str (default = 'DQTabNet')

    Name of the model used for saving in disk, you can customize this to easily retrieve and reuse your trained models.
- saving_path : str (default = './')

    Path defining where to save models.
- scheduler_fn : torch.optim.lr_scheduler (default = None)

    Pytorch Scheduler to change learning rates during training.
- scheduler_params: dict

    Parameters dictionnary for the scheduler_fn. Ex : {"gamma": 0.95,                    "step_size": 10}
- verbose : int (default=-1)

    Verbosity for notebooks plots, set to 1 to see every epoch.
