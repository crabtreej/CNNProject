import evaluateKerasModel as EK
import csv
import keras
from keras.models import load_model

with open('cifar_results.csv', 'w') as write_file:
    pool_types = ['max']#, 'average']
    base_path = 'saved_models/'

    pool_params = (((2,2), 1), ((2,2), 2), ((2,2), 3), ((2,2), 4), ((3,3), 1), 
               ((3,3), 2), ((3,3), 3), ((3,3), 4), ((4,4), 1), ((4,4), 2), 
               ((4,4), 3), ((4,4), 4), ((8,8), 1), ((8,8), 2), ((8,8), 3), 
               ((8,8), 4), ((8,8), 8), ((16,16), 1), ((16,16), 2), ((16,16), 3), 
               ((14,16), 4), ((16,16), 14))
    miss_params = [((16,16), 1), ((16,16), 2), ((16,16), 3), 
               ((16,16), 4), ((16,16), 16)]

    writer = csv.writer(write_file)
    writer.writerow(['dataset', 'pool type', 'pool size', 'pool stride', 'accuracy'])
    for pool_type in pool_types:
        for pool_size, stride in miss_params:
            model = EK.reload_keras_model(base_path + f'{pool_type}_pool_{str(pool_size[0])}_{str(pool_size[1])}_{str(stride)}.h5')
            x_test, y_test = EK.get_cifar_data()
            _, accuracy = EK.report_test_metrics(model, x_test, y_test)
            writer.writerow(['cifar', pool_type, str(pool_size[0]) + 'x' + str(pool_size[1]), str(stride), str(accuracy)])
            write_file.flush()

