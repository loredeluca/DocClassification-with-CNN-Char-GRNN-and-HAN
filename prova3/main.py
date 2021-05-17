from preprocessing import hanPreprocessing
from train import hanTrain

trainSet, label, unique_tokens = hanPreprocessing()
hanTrain(trainSet, label, unique_tokens)
