import os
import torch
import numpy as np


def test(dataset, model_path):

    checkpoint_path = model_path + '_checkpoint.tar'
    if not os.path.isfile(checkpoint_path):
        print("Couldn't find the model checkpoint.")
        return

    checkpoint = torch.load(checkpoint_path)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    val_indices = checkpoint['val_indices']
    learning_rate = checkpoint['learning_rate']

    print(f"Calculating accuracy of the model after {epoch+1} epochs of training (last lr: {learning_rate}...")

    matches = 0
    diffs = []
    processed_docs = 0
    for k, i in enumerate(val_indices):
        if k % 100 == 0:
            print(f"Data sample {k+1:>7} of {len(val_indices)}")
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