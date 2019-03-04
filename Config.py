# _*_ coding:utf8 _*_
from datetime import datetime


class Config:
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    # Global Hyperparameter  
    max_length = 2000 # longest sequence to parse     
    n_classes = 4 # name entity number  
    batch_size = 32
    n_epochs = 100
    lr = 0.001 # learn rate
    LABELS = ['B', 'M', 'E', 'S'] # Label Strategy
    UNK = "<unk>" # Unknown character
    is_normalize = True # normalize number
    embed_size = 100 # embeddings size
    random_seed = 121 # initialize random seed
    is_report = False  # Report(Config.eval_output)
    dev_seg_size = 2000 # division Verification set size


    # Tcn model Hyperparameter
    filters_size = 100   
    num_layers = 4 # number of hidden layers   
    kernel_size = 3 # Convolution kernel
    dropout = 0.3

    # BiLSTM model Hyperparameter
    bi_dropout_rate = 0.3


    def __init__(self, args):
        self.model = args.model
        self.training = False
        scheme = args.o
        if scheme == 'train':
            self.training = True
        if args.model_path is not None:
            # Where to save things.
            self.output_path = args.model_path
        else:
            self.output_path = "results/{}/{:%Y%m%d_%H%M%S}/".format(self.model, datetime.now())
        self.model_output = self.output_path + "model.weights"
        self.eval_output = self.output_path + "results.txt"
        self.conll_output = self.output_path + "{}_predictions.conll".format(self.model)
        self.log_output = self.output_path + "log"
