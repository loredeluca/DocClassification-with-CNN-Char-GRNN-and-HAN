import argparse
from HAN_preprocessing import HAN_preprocess
from GRNN_preprocessing import GRNN_preprocess
from train import CNNChart_train, HAN_train, GRNN_train
from test import CNNChart_test, HAN_test, GRNN_test

def main(args):
    if args.model == 0:
        # CNN-Char
        data_folder = "./datasets/yelp2013"

        print("START TRAINING\n")
        CNNChart_train(data_folder, "small")

        print("START TESTING\n")
        CNNChart_test(data_folder)
    elif args.model == 1:
        # GRNN
        sentence_model = 1  # 'conv':0, 'lstm':1

        data_folder = "./datasets/yelp2013"
        output_folder = "./grnn_data"

        embedding, _, train_size, val_size, test_size = GRNN_preprocess(data_folder, output_folder)
        print("START TRAINING\n")
        GRNN_train(output_folder, embedding, sentence_model, train_size, val_size)

        print("START TEST\n")
        GRNN_test(output_folder, test_size)
    elif args.model == 2:
        # HAN
        data_folder = "./datasets/yelp2013"
        output_folder = "./han_data"

        word_vocab, classes = HAN_preprocess(data_folder, output_folder, sentence_limit=15, word_limit=20)
        print("START TRAINING\n")
        HAN_train(output_folder, word_map=word_vocab, n_classes=classes)

        print("START TESTING\n")
        HAN_test(output_folder)
    else:
        print("Didn't match a case")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Document Classification")
    parser.add_argument("--model", help="0='cnn-char', 1='grnn', 2='han'", type=int, default=0, choices=[0, 1, 2])
    args = parser.parse_args()

    main(args)
