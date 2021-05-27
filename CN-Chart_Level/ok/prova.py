import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import datetime

#import torch.nn as nn
#import torch
#import numpy as np
#from sklearn import metrics
#import time

from prep2 import CN_ChartDataset
from model2 import CharacterLevelCNN
from utils2 import get_evaluation, save_checkpoint


def train(data_folder, feature):
    # Training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    training_params = {"batch_size": 64,
                       "shuffle": True,
                       "num_workers": 0}
    # DataLoader
    training_set = CN_ChartDataset(data_folder, 'train')
    train_loader = DataLoader(training_set, **training_params)

    """
    max_length = 1014
    data_folder = "./datasets/"+dataname+"/res"

    # QUI INSERIMENTO METODI PER SALVARE IL MODELLO

    
    test_params = {"batch_size": 128,
                   "shuffle": False,
                   "num_workers": 0}

    training_set = CN_ChartDataset("./datasets/"+dataname+"/train.csv", max_length)
    test_set = CN_ChartDataset("./datasets/"+dataname+"/test.csv", max_length)

    training_generator = DataLoader(training_set, **training_params)
    test_generator = DataLoader(test_set, **test_params)
    """

    if feature == "small":
        model = CharacterLevelCNN(input_length=training_set.max_length, n_classes=training_set.num_classes,
                                  input_dim=len(training_set.vocabulary),
                                  n_conv_filters=256, n_fc_neurons=1024)

    elif feature == "large":
        model = CharacterLevelCNN(input_length=training_set.max_length, n_classes=training_set.num_classes,
                                  input_dim=len(training_set.vocabulary),
                                  n_conv_filters=1024, n_fc_neurons=2048)
    # Training parameters
    lr = 0.001
    momentum = 0.9
    num_epochs = 20
    num_iter_per_epoch = len(train_loader)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    # Move to device
    model = model.to(device)
    criterion = criterion.to(device)

    t0 = time.time()
    for epoch in range(0, num_epochs):
        print('\n============================== Epoch {:} / {:} =============================='.format(epoch + 1, num_epochs))
        # aggiungere qualche metrica?
        start = time.time()
        model.train()
        for step, (data, label) in enumerate(train_loader):
            feature = data.to(device)
            label = label.to(device)

            # Forward prop.
            predictions = model(feature)
            # Loss
            loss = criterion(predictions, label)
            # Back prop.
            optimizer.zero_grad()
            loss.backward()
            # Update
            optimizer.step()

            training_metrics = get_evaluation(label.cpu().numpy(), predictions.cpu().detach().numpy(),
                                              list_metrics=["accuracy", "loss"])
            if step%100 == 0:
                print("Batch: [{:>5}/{:>5}]\t"
                      "Batch Time {}\t"
                      "Lr: {}, Loss: {}, Accuracy: {}".format(step+1, num_iter_per_epoch,
                                                              str(datetime.timedelta(seconds=int(round(time.time()- start)))),
                                                              optimizer.param_groups[0]['lr'], loss, training_metrics["accuracy"]))
                print("loss_Metodo: {}".format(training_metrics["loss"]))
        # Save checkpoint
        save_checkpoint(epoch, model, optimizer)

    print("END TRAINING")
    print("Total training took: {:} (h:mm:ss)".format(str(datetime.timedelta(seconds=int(round(time.time()- t0))))))

from test2 import test
if __name__ == "__main__":
    data_folder = "ag_news_csv"
    train(data_folder, "small")
    test(data_folder)

