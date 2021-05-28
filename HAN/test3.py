import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import datetime

from preprocessing3 import HANDataset
from utils3 import AverageMeter


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
    test_loader = DataLoader(HANDataset(data_folder, 'test'), **test_params)

    # Load model
    checkpoint = torch.load("checkpoint_han.pth.tar")
    model = checkpoint['model']
    model = model.to(device)
    model.eval()

    # Track metrics
    accs = AverageMeter()  # accuracies

    start = time.time()
    for step, (documents, sentences_per_document, words_per_sentence, labels) in enumerate(tqdm(test_loader, desc='Evaluating')):
        documents = documents.to(device)  # (batch_size, sentence_limit, word_limit)
        sentences_per_document = sentences_per_document.squeeze(1).to(device)  # (batch_size)
        words_per_sentence = words_per_sentence.to(device)  # (batch_size, sentence_limit)
        labels = labels.squeeze(1).to(device)  # (batch_size)

        with torch.no_grad():
            test_predictions, _, _ = model(documents, sentences_per_document,words_per_sentence)

        if step % 100 == 0:
            print('  Batch {:>5,}  of  {:>5,}. Batch Time: {:}.'.format(step, len(test_loader), str(datetime.timedelta(
                seconds=int(round(time.time()-start))))))

        # Find accuracy
        _, predictions = test_predictions.max(dim=1)  # (n_documents)
        correct_predictions = torch.eq(predictions, labels).sum().item()
        accuracy = correct_predictions / labels.size(0)

        # Keep track of metrics
        accs.update(accuracy, labels.size(0))

    # Print final result
    print('\n * TEST ACCURACY - %.1f per cent\n' % (accs.avg * 100))
    print('\n * TEST ACCURACY: {0:.2f}%\n'.format(accs.avg * 100))
    print("END TESTING\n")
    print("Total testing time {:} (h:mm:ss)".format(str(datetime.timedelta(seconds=int(round(time.time() - start))))))

