import os
import torch
import numpy as np


def test(dataset, test_indices):

    checkpoint = torch.load('checkpoint_GRNN_prova.pth.tar')
    model = checkpoint['model']
    model.load_state_dict(checkpoint['model_state_dict'])
    #epoch = checkpoint['epoch']
    #val_indices = checkpoint['val_indices']
    #learning_rate = checkpoint['learning_rate']

    matches = 0
    diffs = []
    processed_docs = 0
    for k, i in enumerate(test_indices):
        if k % 100 == 0:
            print(f"Data sample {k+1:>7} of {len(test_indices)}")
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
