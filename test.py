import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import time
import datetime
from sklearn import metrics
from statistics import mean

from CNNChar_preprocessing import CN_CharDataset
from HAN_preprocessing import HANDataset
from GRNN_preprocessing import GRNNDataset, CustomDataloader


def CNNChart_test(data_folder):
    """
    :param data_folder: folder where data files are stored
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    test_params = {"batch_size": 128,
                   "shuffle": False,
                   "num_workers": 0}
    # DataLoader
    test_set = CN_CharDataset(data_folder, 'test')
    test_loader = DataLoader(test_set, **test_params)

    # Load Model
    checkpoint = torch.load("checkpoint_CN-Char.pth.tar")
    model = checkpoint['model']
    model = model.to(device)
    model.eval()

    test_acc = []

    start = time.time()
    for step, (data, label) in enumerate(tqdm(test_loader, desc='Evaluating')):
        test_feature = data.to(device)
        test_label = label.to(device)

        with torch.no_grad():
            test_predictions = model(test_feature)

        if step % 100 == 0:
            print(" Batch {:>5,} of {:>5,}. Batch Time: {:}.".format(step, len(test_loader), str(datetime.timedelta(
                seconds=int(round(time.time()-start))))))

        _, predictions = test_predictions.max(dim=1)
        correct_predictions = torch.eq(predictions, test_label).sum().item()
        accuracy = correct_predictions / test_label.size(0)

        test_acc.append(accuracy)
  
    print("\n * TEST ACCURACY - {}".format(mean(test_acc)*100))
    print("END TESTING\n")
    print("Total testing time {:} (h:mm:ss)".format(str(datetime.timedelta(seconds=int(round(time.time()-start))))))


def HAN_test(data_folder):
    """
    :param data_folder: folder where data files are stored
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    test_params = {'batch_size': 64,
                   'shuffle': False,
                   'num_workers': 4,
                   'pin_memory': True}
    # DataLoader
    test_set = HANDataset(data_folder, 'test')
    test_loader = DataLoader(test_set, **test_params)

    # Load model
    checkpoint = torch.load("checkpoint_han.pth.tar")
    model = checkpoint['model']
    model = model.to(device)
    model.eval()

    test_acc = []

    start = time.time()
    for step, (documents, sentences_per_document, words_per_sentence, labels) in enumerate(tqdm(test_loader, desc='Evaluating')):
        documents = documents.to(device)
        sentences_per_document = sentences_per_document.squeeze(1).to(device)
        words_per_sentence = words_per_sentence.to(device)
        labels = labels.squeeze(1).to(device)

        with torch.no_grad():
            test_predictions, _, _ = model(documents, sentences_per_document,words_per_sentence)

        if step % 100 == 0:
            print('  Batch {:>5,}  of  {:>5,}. Batch Time: {:}.'.format(step, len(test_loader), str(datetime.timedelta(
                seconds=int(round(time.time()-start))))))

        _, predictions = test_predictions.max(dim=1)
        correct_predictions = torch.eq(predictions, labels).sum().item()
        accuracy = correct_predictions / labels.size(0)

        test_acc.append(accuracy)

    print('\n * TEST ACCURACY: {0:.2f}%\n'.format(mean(test_acc) * 100))
    print("END TESTING\n")
    print("Total testing time {:} (h:mm:ss)".format(str(datetime.timedelta(seconds=int(round(time.time() - start))))))


def GRNN_test(data_folder, test_size):
    """
    :param data_folder: folder where data files are stored
    :param test_size: test set size
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    # DataLoader
    test_set = GRNNDataset(data_folder, 'test')
    test_indices = [*range(0, test_size, 1)]
    # test_loader = CustomDataloader(test_set, test_size)

    # Load model
    checkpoint = torch.load('checkpoint_GRNN.pth.tar')
    model = checkpoint['model']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()#

    test_acc = []
    matches = 0
    processed_docs = 0
    start = time.time()
    for step, i in enumerate(tqdm(test_indices, desc="Evaluating")):
        (doc, label) = test_set[i]
        try:
            prediction = torch.argmax(model(doc))
        except AttributeError as e:
            print("Some error occurred. Ignoring this document.")
            continue
        label = torch.Tensor([label])
        label = label.long().to(device)

        if step % 100 == 0:
            print('  Batch {:>5,}  of  {:>5,}. Batch Time: {:}.'.format(step, len(test_indices), str(datetime.timedelta(
                seconds=int(round(time.time() - start))))))

        if label == prediction:
            matches += 1
        processed_docs += 1
        accuracy = float(matches) / float(processed_docs)
        test_acc.append(accuracy)

    accuracy = float(matches) / float(processed_docs)
    print(f"Accuracy: {accuracy}")
    print('\n * TEST ACCURACY: {0:.2f}%\n'.format(mean(test_acc) * 100))
    print("END TESTING\n")
    print("Total testing time {:} (h:mm:ss)".format(str(datetime.timedelta(seconds=int(round(time.time() - start))))))


