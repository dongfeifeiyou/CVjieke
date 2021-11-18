class DefaultConfig(object):
    env = 'default'
    #model = 'AlexNet'
    #batch_size = 32
    model = 'ResNet18'
    batch_size = 1

    train_data_root = r'C:\Users\Administrator\Documents\WeChat Files\wxid_uojrn4k99e3y22\FileStorage\File\2021-11\UTKFace'
    test_data_root = r'C:\Users\Administrator\Documents\WeChat Files\wxid_uojrn4k99e3y22\FileStorage\File\2021-11\UTKFace'

    # train_data_root = r'C:\dataset\DogVSCat\train'
    # test_data_root = r'C:\dataset\DogVSCat\test1'

    load_mode_path = ''


    use_gpu = False
    num_workers = 0
    print_freq = 20

    debug_file = r'/tmp/debug'
    result_file = 'result.csv'

    max_epoch = 100
    lr = 0.001
    lr_decay = 0.95
    weight_decay = 1e-4 # 损失函数
    early_stop = 20

opt = DefaultConfig()

