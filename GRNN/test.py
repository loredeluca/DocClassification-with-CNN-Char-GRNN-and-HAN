import torch
import time
import datetime
from torch.utils.data import DataLoader

from preprocessing import GRNNDataset

def test(data_folder):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    # test_loader
    test_loader = DataLoader(GRNNDataset(data_folder, 'test'), batch_size=64)

    # Load Model
    checkpoint = torch.load('checkpoint_grnn.pth.tar')

    model = checkpoint['model']
    model.to(device)
    #model.load_state_dict(checkpoint['model_state_dict'])
    #epoch = checkpoint['epoch']
    #val_indices = checkpoint['val_indices']
    #learning_rate = checkpoint['learning_rate']

    print("Evaluating network on Test set")
    model.eval()
    start = time.time()
    matches = 0
    diffs = []
    processed_docs = 0
    for step, (doc, label) in enumerate(test_loader):
        documents = doc.to(device)
        labels = label.to(device)
        #(doc, label) = dataset[i] # al posto di (doc, label) ci sarebbe i

        with torch.no_grad():
            prediction = torch.argmax(model(documents))

        # Find accuracy
        _, predictions_ = prediction.max(dim=1)
        correct_predictions = torch.eq(predictions_, labels).sum().item()
        accuracy = correct_predictions / labels.size(0)

    print("TEST Accuracy: {}".format(accuracy))
    print("END TESTING")
    print("Total testing time {:} (h:mm:ss)".format(
        str(datetime.timedelta(seconds=int(round(time.time() - start))))))
