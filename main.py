from models.Tab_Attention import *
from tensorflow.keras.callbacks import ModelCheckpoint
from models.feature_importance import feature_group_multi_way
from sklearn.model_selection import train_test_split
from models.feature_group import *
from function_c.base_util import *
from function_c.result_output import *
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# torch.backends.cudnn.benchmark = True
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
print('ßstart')

data_name = 'taiwan'
monitor = 'val_output_f1_score'

x, y, learning_rate, epochs, batch_size, threshold, label_col = data_read(data_name)

if not os.path.exists('model_file'):
    os.mkdir('model_file')
t = 20
result = pd.DataFrame(index=np.arange(t), columns=['Acc', 'AUC', 'KS', 'Precision_0', 'Recall_0', 'f1_0', 'p1', 'Recall_1', 'f1_1', 'epoch'])

for k in range(t):
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, stratify=y, random_state=k)
    groups_index, groups_num = feature_group_multi_way(train_x, train_y)
    groups_index = [np.arange(x.shape[1])] + groups_index
    groups_num = [x.shape[1]] + groups_num

    len_ = len(groups_num)
    train_x = feature_group(train_x, groups_index)
    test_x = feature_group(test_x, groups_index)

    if (monitor == 'val_output_auc') & (k>0):
        monitor_ = monitor+'_'+str(k)
    else:
        monitor_ = monitor

    test_pred_y = pd.DataFrame(index=np.arange(len(test_y)), columns=['test_y'] + [f'pred_y_{i}' for i in range(len_)])
    model = build_gnns(output_shape=1, groups_num=groups_num, data_name=data_name, attention=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer,
                  loss=['binary_crossentropy'],
                  metrics=[tf.keras.metrics.AUC(), F1_Score(),'accuracy'])

    checkpoint_filepath = f'model_file/{monitor}_{data_name}_Tab_Attention_{k}.h5'
    if os.path.exists(checkpoint_filepath):
        os.remove(checkpoint_filepath)
        print(f'{checkpoint_filepath} has been removed')
    else:
        print(f'{checkpoint_filepath} does not exists')

    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor=monitor_,
        mode='max',
        save_best_only=True, verbose=1)

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor=monitor_, patience=20, verbose=1, mode='max')
    callbacks = [model_checkpoint_callback,
                 tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=10, monitor=monitor_, verbose=1),
                 early_stopping]

    hist = model.fit(x=train_x, y=[train_y] * len_, batch_size=batch_size, epochs=epochs,
                     validation_data=(train_x, [train_y] * len_),
                     callbacks=callbacks)

    if os.path.exists(checkpoint_filepath):
        model.load_weights(checkpoint_filepath)
        print(f'{checkpoint_filepath} has been loaded')

    result.iloc[k, -1] = len(hist.epoch)
    print(f'{k}-time custom dnn for {data_name} ****')
    y_pred = model.predict(test_x)
    test_pred_y['test_y'] = test_y.values
    test_pred_y.iloc[:, 1:] = np.array(y_pred)
    if not os.path.exists(f'result/{data_name}'):
        os.mkdir(f'result/{data_name}')
    test_pred_y.to_csv(f'result/{data_name}/{monitor}_{k}.csv', index=False)
    result.iloc[k, :-1] = result_output(test_y, y_pred[0])

    model.save(checkpoint_filepath)

ci_95(result,sl=3)
