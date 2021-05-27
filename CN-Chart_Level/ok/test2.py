import torch
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
import numpy as np
import datetime

from prep2 import CN_ChartDataset
from utils2 import get_evaluation


def test(data_folder):
    # Test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    test_params = {"batch_size": 128,
                   "shuffle": False,
                   "num_workers": 0}
    # DataLoader
    test_set = CN_ChartDataset(data_folder, 'test')
    test_loader = DataLoader(test_set, **test_params)

    checkpoint = "checkpoint_CN-Char.pth.tar"
    # Load Model
    checkpoint = torch.load(checkpoint)
    model = checkpoint['model']
    model = model.to(device)
    model.eval()

    test_label_ls = []
    test_pred_ls = []

    start = time.time()
    for step, (data, label) in enumerate(tqdm(test_loader)):
        test_feature = data.to(device)
        test_label = label.to(device)

        # Forward prop.
        with torch.no_grad():
            test_predictions = model(test_feature)

        if step%100 == 0:
            elapsed = str(datetime.timedelta(seconds=int(round(time.time()-start))))
            print(" Batch {:>5,} of {:>5,}. Elapsed: {:}.".format(step, len(test_loader), elapsed))

        test_label_ls.extend(test_label.clone().cpu())
        test_pred_ls.append(test_predictions.clone().cpu())

    test_pred = torch.cat(test_pred_ls, 0)
    test_label = np.array(test_label_ls)
    test_metrics = get_evaluation(test_label, test_pred.numpy(), list_metrics=["accuracy", "confusion_matrix"])

    print("\n * TEST ACCURACY - {}".format(test_metrics["accuracy"]))
    print("END TESTING\n")
    print("Total testing took {:} (h:mm:ss)".format(str(datetime.timedelta(seconds=int(round(time.time()-start))))))
