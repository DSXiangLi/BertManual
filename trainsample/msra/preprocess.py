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
    'I-LOC': 6
}

MAPPING = {'train': 'train',
            'val': 'valid',
            'test': 'test'
           }


def load_data(file_name):
    """
    Load sentence and tags
    """
    sentences = read_text(os.path.join(file_name, 'sentences.txt'))
    tags = read_text(os.path.join(file_name, 'tags.txt'))
    assert len(sentences) == len(tags)
    return sentences, tags


def main():
    data_dir = './trainsample/msra'

    # Attention! for NER task, text is joined by ' '
    train_x, train_y = load_data(os.path.join(data_dir, 'train'))
    test_x, test_y = load_data(os.path.join(data_dir, 'test'))
    valid_x, valid_y = load_data(os.path.join(data_dir, 'val'))

    train_y = [[Label2IDX[j] for j in i.split(' ')]for i in train_y]
    test_y = [[Label2IDX[j] for j in i.split(' ')]for i in test_y]
    valid_y = [[Label2IDX[j] for j in i.split(' ')]for i in valid_y]

    single_text(train_x, train_y, data_dir, output_file='train')
    single_text(test_x, test_y, data_dir, output_file='test')
    single_text(valid_x, valid_y, data_dir, output_file='valid')


if __name__ == '__main__':
    main()