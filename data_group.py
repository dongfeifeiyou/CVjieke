import os
import numpy as np
from sklearn.model_selection import train_test_split
import json

if __name__ == '__main__':
    dir_imgs = r'H:\UTKFace\UTKFace'
    files = os.listdir(dir_imgs)
    labels = [int(item.split('_')[1]) for item in files]
    X_train,X_test, y_train, y_test =train_test_split(files,
                                                      labels,
                                                      test_size=0.4,
                                                      random_state=0,
                                                      stratify=labels)
    X_val,X_test, y_val, y_test =train_test_split(X_test,
                                                  y_test,
                                                  test_size=0.5,
                                                  random_state=0,
                                                  stratify=y_test)
    print(len(X_train))
    print(len(X_val))
    print(len(X_test))
    print(sum(y_train))
    print(sum(y_val))
    print(sum(y_test))
    data_dict = {
        'train': {
            'data': X_train,
            'label': y_train
        },
        'val': {
            'data': X_val,
            'label': y_val
        },
        'test': {
            'data': X_test,
            'label': y_test
        },
    }
    json_str = json.dumps(data_dict)
    with open('data.json', 'w', encoding='utf-8') as f:
        json.dump(data_dict, f)