import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import datetime
from torch.utils.data import DataLoader

from utils import split_data, CustomDataloader
from model import DocSenModel
from preprocessing import GRNNDataset


def train(data_folder, n_classes, embedding, sentence_model, gnn_type, gnn_output):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    # load Checkpoint

    training_params = {"batch_size": 64,
                       "shuffle": True,
                       "num_workers": 4,
                       "pin_memory": True}

    # DataLoaders
    # training_set = HANDataset(data_folder, 'train')
    train_loader = DataLoader(GRNNDataset(data_folder, 'train'), **training_params)
    # VALIDATION
    val_loader = DataLoader(GRNNDataset(data_folder, 'val'), batch_size=64)

    # Model parameter
    if sentence_model == 0:
        print("Sentence model: convolution")
        sentence_model = DocSenModel.SentenceModel.CONV
    else:
        print("Sentence model: LSTM")
        sentence_model = DocSenModel.SentenceModel.LSTM

    if gnn_type == 0:
        print("GNN type: forward")
        gnn_type = DocSenModel.GnnType.FORWARD
    else:
        print("GNN type: forward-backward")
        gnn_type = DocSenModel.GnnType.FORWARD_BACKWARD

    if gnn_output == 0:
        print("GNN output: last")
        gnn_output = DocSenModel.GnnOutput.LAST
    else:
        print("GNN output: avg")
        gnn_output = DocSenModel.GnnOutput.AVG

    # Model
    model = DocSenModel(n_classes, sentence_model, gnn_output, gnn_type, embedding, cuda=True)

    # Training parameters
    lr = 0.03
    l2_reg = 1e-5
    epochs = 70

    criterion = nn.CrossEntropyLoss() #loss_function
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=l2_reg)

    # Move to device
    model = model.to(device)
    criterion = criterion.to(device)

    train_loss = []
    valid_loss = []
    train_acc = []
    valid_acc = []

    t0 = time.time()
    for epoch in range(0, epochs):
        print("\n ======================= Epoch {:}/{:} =====================".format(epoch+1, epochs))
        start = time.time()
        model.train()
        '''
        EARLY STOPPING
        if early_stopping > 0:
            if epoch - min_loss_epoch >= early_stopping:
                print(f"No training improvement over the last {early_stopping} epochs. Aborting.")
                break
        '''

        for step, (doc, label) in enumerate(train_loader):

            documents = doc.to(device)
            outputs = model(documents)
            labels = label.to(device)

            '''
            predictions = None
            labels = None
            matches = 0
            
            prediction = model(doc)
            prediction = prediction.unsqueeze(0)
            predictions = prediction if predictions is None else torch.cat((predictions, prediction))
            label = torch.Tensor([label])
            label = label.long().to(model._device)
            labels = label if labels is None else torch.cat((labels, label))

            if label == torch.argmax(prediction):
                matches += 1
            '''

            # Loss
            loss = criterion(outputs, labels)
            # Back prop.
            optimizer.zero_grad()
            loss.backward()
            # Update
            optimizer.step()

            # Find accuracy
            _, predictions_ = outputs.max(dim=1)
            correct_predictions = torch.eq(predictions_, labels).sum().item()
            accuracy = correct_predictions / labels.size(0)
            '''
            accuracy = float(matches) / float(len(predictions))
            '''
            train_acc.append(accuracy)
            train_loss.append(loss.item())

            if step % 1000 == 0:
                print("Batch: [{:>5}/{:>5}]\t"
                      "Batch Time {}\t"
                      "Loss: {}, Accuracy: {}".format(step + 1, len(train_loader),
                                                      str(datetime.timedelta(
                                                          seconds=int(round(time.time() - start)))),
                                                      loss, accuracy))

            adjust_learning_rate(optimizer, 0.8)
            # Update Learning Rate
            # learning_rate = 0.8 * learning_rate
            # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=l2_reg)

            # Save checkpoint
            save_checkpoint(epoch, model, optimizer)

            # VALIDATION
            n_correct, n_total = 0, 0

            model.eval()
            with torch.no_grad():
                for t_step, (t_doc, t_label) in enumerate(val_loader):
                    doc = t_doc.to(device)
                    labels = t_label.to(device)
                    outputs = model(doc)

                    n_correct += (torch.argmax(outputs, -1) == labels).sum().item()
                    n_total += len(outputs)

                val_acc = n_correct / n_total
                valid_acc.append(val_acc)

            print("Validation Accuracy: {}".format(val_acc))

    print("END TRAINING\n")
    print("Total training time {:} (h:mm:ss)".format(
                str(datetime.timedelta(seconds=int(round(time.time() - t0))))))


def save_checkpoint(epoch, model, optimizer):  # , word_map):
    state = {'epoch': epoch,
             'model': model,
             'optimizer': optimizer}
             # ,'word_map': word_map}
    filename = 'checkpoint_grnn.pth.tar'
    torch.save(state, filename)

def adjust_learning_rate(optimizer, scale_factor):
    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * scale_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


