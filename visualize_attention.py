import torch
from torch import nn
import json
import os
from nltk.tokenize import PunktSentenceTokenizer, TreebankWordTokenizer
from PIL import Image, ImageDraw, ImageFont

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
checkpoint = 'BEST_checkpoint_han.pth.tar'
checkpoint = torch.load(checkpoint, map_location=device)
model = checkpoint['model']
model = model.to(device)
model.eval()

# Pad limits, can use any high-enough value since our model does not compute over the pads
sentence_limit = 15
word_limit = 20

# Word map to encode with
data_folder = './han_data'
with open(os.path.join(data_folder, 'word_map.json'), 'r') as j:
    word_map = json.load(j)

# Tokenizers
sent_tokenizer = PunktSentenceTokenizer()
word_tokenizer = TreebankWordTokenizer()

classes = ["1", "2", "3", "4", "5"]
label_map = {k: v for v, k in enumerate(classes)}
rev_label_map = {v: k for k, v in label_map.items()}


def classify(document):
    doc = list()

    # Tokenize document into sentences
    sentences = list()
    for paragraph in preprocess(document).splitlines():
        sentences.extend([s for s in sent_tokenizer.tokenize(paragraph)])

    # Tokenize sentences into words
    for s in sentences[:sentence_limit]:
        w = word_tokenizer.tokenize(s)[:word_limit]
        if len(w) == 0:
            continue
        doc.append(w)

    # Number of sentences in the document
    sentences_in_doc = len(doc)
    sentences_in_doc = torch.LongTensor([sentences_in_doc]).to(device)

    # Number of words in each sentence
    words_in_each_sentence = list(map(lambda s: len(s), doc))
    words_in_each_sentence = torch.LongTensor(words_in_each_sentence).unsqueeze(0).to(device)

    # Encode document with indices from the word map
    encoded_doc = list(
        map(lambda s: list(map(lambda w: word_map.get(w, word_map['<unk>']), s)) + [0] * (word_limit - len(s)),
            doc)) + [[0] * word_limit] * (sentence_limit - len(doc))
    encoded_doc = torch.LongTensor(encoded_doc).unsqueeze(0).to(device)

    # Apply the HAN model
    scores, word_alphas, sentence_alphas = model(encoded_doc, sentences_in_doc,
                                                 words_in_each_sentence)
    scores = scores.squeeze(0)
    scores = nn.functional.softmax(scores, dim=0)
    word_alphas = word_alphas.squeeze(0)
    sentence_alphas = sentence_alphas.squeeze(0)
    words_in_each_sentence = words_in_each_sentence.squeeze(0)

    return doc, scores, word_alphas, sentence_alphas, words_in_each_sentence


def visualize_attention(doc, scores, word_alphas, sentence_alphas, words_in_each_sentence):
    # Find best prediction
    score, prediction = scores.max(dim=0)
    prediction = '{category} ({score:.2f}%)'.format(category=rev_label_map[prediction.item()], score=score.item() * 100)
    print("PREDICTION ", prediction)

    # For each word, find it's effective importance (sentence alpha * word alpha)
    alphas = (sentence_alphas.unsqueeze(1) * word_alphas * words_in_each_sentence.unsqueeze(1).float() / words_in_each_sentence.max().float())
    alphas = alphas.to(device)

    # Determine size of the image, visualization properties for each word, and each sentence
    min_font_size = 15  # minimum size possible for a word
    max_font_size = 55  # maximum size possible for a word
    space_size = ImageFont.truetype("./calibril.ttf", 35).getsize(' ')  # use spaces of maximum font size
    line_spacing = 0  # 15 # spacing between sentences
    left_buffer = 100  # initial empty space on the left where sentence-rectangles will be drawn
    top_buffer = 2 * min_font_size + 3 * line_spacing  # initial empty space on the top where the detected category will be displayed
    image_width = left_buffer  # width of the entire image so far
    image_height = top_buffer + line_spacing  # height of the entire image so far
    word_loc = [image_width, image_height]  # top-left coordinates of the next word that will be printed
    rectangle_height = 0.75 * max_font_size  # height of the rectangles that will represent sentence alphas
    rectangle_loc = [0.9 * left_buffer, image_height + rectangle_height]  # bottom-right coordinates of next rectangle that will be printed
    word_viz_properties = list()
    sentence_viz_properties = list()

    S1 = []
    W1 = []

    S = [(255, 255, 255), (255, 192, 182), (255, 128, 128), (255, 31, 53)]
    W = [(255, 255, 255), (224, 224, 255), (82, 128, 255), (82, 82, 255)]

    for s, sentence in enumerate(doc):
        sentence_factor = sentence_alphas[s].item() / sentence_alphas.max().item()

        S1.append(sentence_factor)
        if sentence_factor < 0.95:  # 97
            if sentence_factor < 0.90:  # 95
                if sentence_factor < 0.83:  # 93
                    rectangle_color = S[0]
                else:
                    rectangle_color = S[1]
            else:
                rectangle_color = S[2]
        else:
            rectangle_color = S[3]

        rectangle_bounds = [50.0, 30.0 + (s * 55), 90.0, 70 + (s * 55)]  # 90-130

        # Save sentence's rectangle's properties
        sentence_viz_properties.append({'bounds': rectangle_bounds.copy(),
                                        'color': rectangle_color})
        tmp = word_loc[0]
        for w, word in enumerate(sentence):
            word_factor = alphas[s, w].item() / alphas.max().item()
            if word_factor < 0.90:  # 90
                if word_factor < 0.75:  # 75
                    if word_factor < 0.55:  # 60
                        word_color = W[0]
                    else:
                        print(word_factor)
                        word_color = W[1]
                else:
                    word_color = W[2]
            else:
                word_color = W[3]

            W1.append(word_factor)

            word_font = ImageFont.truetype("./calibril.ttf", 35)

            w_size = word_font.getsize(word)

            if w == 0:
                x = tmp
            word_box_loc = [x, 32 + (s * 55), x+w_size[0], 63 + (s * 55)]  # 90-130 #92-123
            x += w_size[0]+space_size[0]

            # Save word's properties
            word_viz_properties.append({'loc': word_loc.copy(),
                                        'locB': word_box_loc,
                                        'word': word,
                                        'font': word_font,
                                        'color': word_color})

            # Update word and sentence locations for next word, height, width values
            word_size = word_font.getsize(word)
            word_loc[0] += word_size[0] + space_size[0]
            image_width = max(image_width, word_loc[0])
        word_loc[0] = left_buffer
        word_loc[1] += max_font_size + line_spacing
        image_height = max(image_height, word_loc[1])
        rectangle_loc[1] += max_font_size + line_spacing

    # Create blank image
    img = Image.new('RGB', (image_width, image_height), (255, 255, 255))

    # Draw
    draw = ImageDraw.Draw(img)
    # Words
    for viz in word_viz_properties:
        draw.rectangle(xy=viz['locB'], fill=viz['color'])
        draw.text(xy=viz['loc'], text=viz['word'], fill=(0, 0, 0), font=viz['font'])
    # Rectangles that represent sentences
    for viz in sentence_viz_properties:
        draw.rectangle(xy=viz['bounds'], fill=viz['color'])

    # Detected category/topic
    # category_font = ImageFont.truetype("./calibril.ttf", min_font_size)
    # draw.text(xy=[line_spacing, line_spacing], text='Detected Category:', fill='grey', font=category_font)
    # draw.text(xy=[line_spacing, line_spacing + category_font.getsize('Detected Category:')[1] + line_spacing],
    #          text=prediction.upper(), fill='black',
    #          font=category_font)
    del draw

    print("S", S1)
    print("W", W1)
    img.save("result.jpg")


def preprocess(text):
    if isinstance(text, float):
        return ''

    return text.lower().replace('<br />', '\n').replace('<br>', '\n').replace('\\n', '\n').replace('&#xd;', '\n')


if __name__ == '__main__':
    doc5star = "We had an amazing dinner last night at 'Coppia'.  There were four of us, and we like to share tastes.  When each dish is wonderful, it makes for some great bites. We had three of the pastas on the menu and all were perfect.  Equal to what you would expect in the best trattoria or ristorante in Italy.  The pasta al dente and each of the sauces were unique and delicious from the ragu to the sugo.  We will be back and would recommend this local treasure."
    doc1star = "Went here the other night with Mom, sister, and nieces.  Our server was sooooo slow, I swear we felt like we needed to send a search party after him!  Prices were OUTRAGEOUS!!!! The food was ok not amazing for the prices that they ask. This place will never come to my mind as a restaurant to spend my hard earned money at ever again. Mediocre is an understatement for this restaurant!"
    visualize_attention(*classify(doc1star))
