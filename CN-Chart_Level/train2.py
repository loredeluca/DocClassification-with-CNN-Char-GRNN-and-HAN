from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import numpy as np
from sklearn import metrics
import time

from prep2 import MyDataset
from model2 import CharacterLevelCNN


def get_evaluation(y_true, y_prob, list_metrics):
    y_pred = np.argmax(y_prob, -1)
    output = {}
    if 'accuracy' in list_metrics:
        output['accuracy'] = metrics.accuracy_score(y_true, y_pred)
    if 'loss' in list_metrics:
        try:
            output['loss'] = metrics.log_loss(y_true, y_prob)
        except ValueError:
            output['loss'] = -1
    if 'confusion_matrix' in list_metrics:
        output['confusion_matrix'] = str(metrics.confusion_matrix(y_true, y_pred))
    return output

def train(dataname, feature):
    max_length = 1014
    dataname = "ag_news_csv"
    data_folder = "./datasets/"+dataname+"/res"

    # QUI INSERIMENTO METODI PER SALVARE IL MODELLO

    training_params = {"batch_size": 128,
                       "shuffle": True,
                       "num_workers": 0}
    test_params = {"batch_size": 128,
                   "shuffle": False,
                   "num_workers": 0}

    training_set = MyDataset("./datasets/"+dataname+"/train.csv", max_length)
    test_set = MyDataset("./datasets/"+dataname+"/test.csv", max_length)

    training_generator = DataLoader(training_set, **training_params)
    test_generator = DataLoader(test_set, **test_params)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if feature == "small":
        model = CharacterLevelCNN(input_length=max_length, n_classes=training_set.num_classes,
                                  input_dim=len(training_set.vocabulary),
                                  n_conv_filters=256, n_fc_neurons=1024)

    elif feature == "large":
        model = CharacterLevelCNN(input_length=max_length, n_classes=training_set.num_classes,
                                  input_dim=len(training_set.vocabulary),
                                  n_conv_filters=1024, n_fc_neurons=2048)

    #if torch.cuda.is_available():
    #    model.cuda() # model.to(device) penso sia la stessa cosa

    lr = 0.001

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    model = model.to(device)
    criterion = criterion.to(device)

    # parameters
    best_loss = 1e5
    best_epoch = 0
    num_iter_per_epoch = len(training_generator)
    num_epochs = 20

    #model.train()
    t0 = time.time()
    for epoch in range(num_epochs):
        model.train()
        start = time.time()
        for iter, batch in enumerate(training_generator):
            feature, label = batch.to(device)

            #if torch.cuda.is_available():
            #    feature = feature.cuda()
            #    label = label.cuda()
            predictions = model(feature)
            loss = criterion(predictions, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            training_metrics = get_evaluation(label.cpu().numpy(), predictions.cpu().detach().numpy(),
                                              list_metrics=["accuracy", "loss"])
            print("Epoch: {}/{} - time {}, Iteration: {}/{}, Lr: {}, Loss: {}, Accuracy: {}".format(
                epoch + 1,
                num_epochs, (time.time()-start),
                iter + 1,
                num_iter_per_epoch,
                optimizer.param_groups[0]['lr'],
                loss, training_metrics["accuracy"]))
            print("lossMetodo: {}".format(training_metrics["loss"]))
            # SaveCheckpoint()
        print("END TRAINING")
            #writer.add_scalar('Train/Loss', loss, epoch * num_iter_per_epoch + iter)
            #writer.add_scalar('Train/Accuracy', training_metrics["accuracy"], epoch * num_iter_per_epoch + iter)
        print("\nSTART TEST...")
        model.eval()
        loss_ls = []
        te_label_ls = []
        te_pred_ls = []
        s2 = time.time()
        for batch in test_generator:
            te_feature, te_label = batch.to(device)
            num_sample = len(te_label)
            #if torch.cuda.is_available():
            #    te_feature = te_feature.cuda()
            #    te_label = te_label.cuda()
            with torch.no_grad():
                te_predictions = model(te_feature)
            te_loss = criterion(te_predictions, te_label)
            loss_ls.append(te_loss * num_sample)
            te_label_ls.extend(te_label.clone().cpu())
            te_pred_ls.append(te_predictions.clone().cpu())

        te_loss = sum(loss_ls) / test_set.__len__()
        te_pred = torch.cat(te_pred_ls, 0)
        te_label = np.array(te_label_ls)
        test_metrics = get_evaluation(te_label, te_pred.numpy(), list_metrics=["accuracy", "confusion_matrix"])
        #output_file.write(
        #    "Epoch: {}/{} \nTest loss: {} Test accuracy: {} \nTest confusion matrix: \n{}\n\n".format(
        #        epoch + 1, num_epochs,
        #        te_loss,
        #        test_metrics["accuracy"],
        #        test_metrics["confusion_matrix"]))
        print("Epoch: {}/{} - time {}, Lr: {}, Loss: {}, Accuracy: {}".format(
            epoch + 1,
            num_epochs, (time.time()-s2),
            optimizer.param_groups[0]['lr'],
            te_loss, test_metrics["accuracy"]))
        #writer.add_scalar('Test/Loss', te_loss, epoch)
        #writer.add_scalar('Test/Accuracy', test_metrics["accuracy"], epoch)

        print("END TEST")
    print("END - total time: {}".format(time.time()-t0))

    """
        model.train()
        if te_loss + opt.es_min_delta < best_loss:
            best_loss = te_loss
            best_epoch = epoch
            torch.save(model, "{}/char-cnn_{}_{}".format(opt.output, opt.dataset, opt.feature))
        # Early stopping
        if epoch - best_epoch > opt.es_patience > 0:
            print("Stop training at epoch {}. The lowest loss achieved is {} at epoch {}".format(epoch, te_loss, best_epoch))
            break
        if opt.optimizer == "sgd" and epoch % 3 == 0 and epoch > 0:
            current_lr = optimizer.state_dict()['param_groups'][0]['lr']
            current_lr /= 2
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
    """

if __name__ == "__main__":
    train("ag_news_csv", "small")
