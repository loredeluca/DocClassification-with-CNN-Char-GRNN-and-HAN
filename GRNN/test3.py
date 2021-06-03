import torch
import time
import datetime
from torch.utils.data import DataLoader
import numpy as np

from preprocessing import GRNNDataset

def test(dataset, data_folder):
    checkpoint = torch.load('checkpoint_grnn.pth.tar')
    test_indices = checkpoint['test_indices']

    # test_loader
    test_loader = DataLoader(GRNNDataset(data_folder, 'test'), batch_size=64)

    # Load Model
    model = checkpoint['model']

    #model.load_state_dict(checkpoint['model_state_dict'])
    #epoch = checkpoint['epoch']
    #val_indices = checkpoint['val_indices']
    #learning_rate = checkpoint['learning_rate']

    matches = 0
    diffs = []
    processed_docs = 0
    start = time.time()
    for step, i in enumerate(test_indices):
        if step % 100 == 0:
            print(f"Data sample {step + 1:>7} of {len(test_indices)}")
        (doc, label) = dataset[i]
        try:
            prediction = torch.argmax(model(doc))
        except AttributeError as e:
            print("Something went wrong. Ignoring this document.")
            continue
        label = torch.Tensor([label])
        label = label.long()
        if label == prediction:
            matches += 1
        processed_docs += 1
        diffs.append(int(np.abs(label - prediction)))
    accuracy = float(matches) / float(processed_docs)
    diffs = np.array(diffs)
    mae = diffs.mean()
    aev = diffs.var()
    print(f"Accuracy: {accuracy}")
    print(f"Mean Absolute Error: {mae}")
    print(f"Absolute Error Variance: {aev}")
    print("END TESTING")
    print("Total testing time {:} (h:mm:ss)".format(
        str(datetime.timedelta(seconds=int(round(time.time() - start))))))
