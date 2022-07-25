# -*-coding:utf-8 -*-
import os
from trainsample.converter import single_text, read_text

Label2IDX = {
    'O': 0,
    'B-ORG': 1,
    'I-ORG': 2,
    'B-PER': 3,
    'I-PER': 4,
    'B-LOC': 5,
    'I-LOC': 6,
}


def load_data(file_name):
    """
    Load line and generate sentence and tag list with char split by ' '
    sentences: ['今 天 天 气 好 ', ]
    tags: ['O O O O O ', ]
    """
    tokens = read_text(file_name)
    sentences = []
    tags = []
    tag = []
    sentence = []
    for i in tokens:
        if i == '':
            # Here join by ' ' to avoid bert_tokenizer merging tokens
            sentences.append(' '.join(sentence))
            tags.append(' '.join(tag))
            tag = []
            sentence = []
        else:
            s, t = i.split(' ')
            tag.append(t)
            sentence.append(s)
    return sentences, tags


def main():
    data_dir ='./trainsample/people_daily'

    train_x, train_y = load_data(os.path.join(data_dir, 'example.train'))
    test_x, test_y = load_data(os.path.join(data_dir, 'example.test'))
    valid_x, valid_y = load_data(os.path.join(data_dir, 'example.dev'))

    train_y = [[Label2IDX[j] for j in i.split(' ')] for i in train_y]
    test_y = [[Label2IDX[j] for j in i.split(' ')] for i in test_y]
    valid_y = [[Label2IDX[j] for j in i.split(' ')] for i in valid_y]

    single_text(train_x, train_y, data_dir, output_file='train')
    single_text(test_x, test_y, data_dir, output_file='test')
    single_text(valid_x, valid_y, data_dir, output_file='valid')


if __name__ == '__main__':
    main()



