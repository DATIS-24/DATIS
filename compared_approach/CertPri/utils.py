import os
import numpy as np
import keras
from progressbar import ProgressBar
from keras.models import clone_model
import keras.backend as BE
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
#from pyecharts import options as opts
#from pyecharts.charts import Bar
from keras.utils import to_categorical

def get_acc(predict_label, ground_truth=None):
    for i in predict_label[0]:
        if i[0] == ground_truth:
            return True
    return False

def softmax( f ):
    # instead: first shift the values of f so that the highest number is 0:
    p = f.copy()
    p -= np.max(p) # f becomes [-666, -333, 0]
    return np.exp(p) / np.sum(np.exp(p))  # safe to do, gives the correct answer


# def get_APFD_ImageNet(Gini, WNID, predicted_confidence, top_set=None):
#     Gini = np.array(Gini)
#     indexs = np.argsort(Gini)[::-1]

#     o_i = 0
#     pbar = ProgressBar()
#     wrong_num = 0
#     for i in pbar(range(0, len(Gini))):
#         if top_set is not None:
#             if not get_acc(predict_label=decode_predictions(predicted_confidence[indexs[i]], top=top_set), ground_truth=WNID[val_ground_truth[indexs[i]]-1]):
#                 o_i = o_i+i
#                 print(i, o_i)
#                 wrong_num = wrong_num+1

#     APFD = 1 - o_i/(len(Gini)*wrong_num) + 1/(2*len(Gini))
#     print(o_i, len(Gini),wrong_num)
#     return APFD


def get_APFD(Gini_indexs, ground_truth_label, predicted_confidence, top_set=None, decode_predictions=None):
    o_i = 0
    pbar = ProgressBar()
    wrong_num = 0
    wrong_num_index = []
    for i in pbar(range(0, len(Gini_indexs))):
        if top_set is not None:
            if decode_predictions is None:
                print('Error: decode_predictions can not be None!')
                return
            if not get_acc(predict_label=decode_predictions(predicted_confidence[Gini_indexs[i]], top=top_set), 
                           ground_truth=ground_truth_label[Gini_indexs[i]]):
                o_i = o_i+i
#                 print(i, o_i)
                wrong_num = wrong_num+1
                wrong_num_index.append(Gini_indexs[i])
        else:
            if np.argmax(ground_truth_label[Gini_indexs[i]]) != np.argmax(predicted_confidence[Gini_indexs[i]]):
                o_i = o_i+i
                wrong_num = wrong_num+1
                wrong_num_index.append(Gini_indexs[i])
    APFD = 1 - o_i/(len(Gini_indexs)*wrong_num) + 1/(2*len(Gini_indexs))
    return APFD, wrong_num, np.array(wrong_num_index).reshape(-1)

def get_APFD_reg(Gini_indexs, ground_truth_label, predicted_confidence, thre_mse=None):
    o_i = 0
    pbar = ProgressBar()
    wrong_num = 0
    wrong_num_index = []
    
    if thre_mse is None:
        mse_tmp = []
        for i in range(len(Gini_indexs)):
            mse_tmp.append( np.sum(pow( predicted_confidence[i][0] - ground_truth_label[i], 2)))
        thre_mse = np.mean(mse_tmp)
        # thre_rmse = np.sqrt(thre_mse)

    for i in pbar(range(0, len(Gini_indexs))):
#         mse = np.sum(pow( ground_truth_label[i] - predicted_confidence[i][0], 2))
        mse = np.sum(pow( ground_truth_label[Gini_indexs[i]] - predicted_confidence[Gini_indexs[i]][0], 2))
        if mse > thre_mse:
            o_i = o_i+i
            wrong_num = wrong_num+1
            wrong_num_index.append(Gini_indexs[i])
    APFD = 1 - o_i/(len(Gini_indexs)*wrong_num) + 1/(2*len(Gini_indexs))
    return APFD, wrong_num, np.array(wrong_num_index).reshape(-1)

def get_RAUC(Gini_indexs, ground_truth_label, predicted_confidence, top_set=None, decode_predictions=None):
    pre_y_axis = []
    o_i = 0
    wrong_num = 0
    pbar = ProgressBar()
    for i in pbar(range(0, len(Gini_indexs))):
        if top_set is not None:
            if decode_predictions is None:
                print('Error: decode_predictions can not be None!')
                return
            if not get_acc(predict_label=decode_predictions(predicted_confidence[Gini_indexs[i]], top=top_set), 
                           ground_truth=ground_truth_label[Gini_indexs[i]]):  
                o_i = o_i+1
                wrong_num = wrong_num+1
                pre_y_axis.append(o_i)
            else:
                pre_y_axis.append(o_i)
        else:
            if np.argmax(ground_truth_label[Gini_indexs[i]]) != np.argmax(predicted_confidence[Gini_indexs[i]]):
                o_i = o_i+1
                wrong_num = wrong_num+1
                pre_y_axis.append(o_i)
            else:
                pre_y_axis.append(o_i)
    true_y_axis = wrong_num*(len(Gini_indexs)-wrong_num) + (wrong_num+1)*wrong_num/2
    RAUC = np.sum(pre_y_axis)/true_y_axis
#     print("RAUC: ", RAUC)
    return RAUC, len(Gini_indexs), wrong_num

def get_RAUC_reg(Gini_indexs, ground_truth_label, predicted_confidence, thre_mse=None):
    pre_y_axis = []
    o_i = 0
    wrong_num = 0
    pbar = ProgressBar()
    
    if thre_mse is None:
        mse_tmp = []
        for i in range(len(Gini_indexs)):
            mse_tmp.append( np.sum(pow( predicted_confidence[i][0] - ground_truth_label[i], 2)))
        thre_mse = np.mean(mse_tmp)
    
    for i in pbar(range(0, len(Gini_indexs))):
        mse = np.sum(pow( ground_truth_label[Gini_indexs[i]] - predicted_confidence[Gini_indexs[i]][0], 2))
        if mse > thre_mse:
            o_i = o_i+1
            wrong_num = wrong_num+1
            pre_y_axis.append(o_i)
        else:
            pre_y_axis.append(o_i)
    true_y_axis = wrong_num*(len(Gini_indexs)-wrong_num) + (wrong_num+1)*wrong_num/2
    RAUC = np.sum(pre_y_axis)/true_y_axis
#     print("RAUC: ", RAUC)
    return RAUC, len(Gini_indexs), wrong_num

def get_loss_gradients(img_input, model, target_one_hot, from_logits=False):
    images = tf.cast(img_input, tf.float32)
    images=np.expand_dims(images, axis=0)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=from_logits)
    images = tf.convert_to_tensor(images)
    images = tf.Variable(images, dtype=tf.float32)
    labels = np.argmax(target_one_hot, axis=1)
   
    with tf.GradientTape() as tape:
        tape.watch(images)
        preds = model(images)
        loss = cce(target_one_hot,preds)
        #loss = loss_fn(labels, preds)
#         top_class = preds[:, top_pred_idx]

    grads = tape.gradient(loss, images)
    #grads = tape.gradient(loss, model.trainable_variables)
    #grads_loss_to_input = tf.gradients(loss, preds, grad_ys=grads_1)[0]

    return grads

def get_img_array(img_path, size=(299, 299)):
    # `img` is a PIL image of size 299x299
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array

# 根据损失函数估计梯度
def GradientEstimator(samples, sigma, model, x, bounds, noise_mu, nise_std, clip=True):
#     value = loss_fn(x)
    x = tf.convert_to_tensor(x)
    gradient = np.zeros_like(x)
    bounds_lower, bounds_upper = bounds
    for k in range(samples // 2):
        noise = np.random.normal(noise_mu, nise_std, x.shape)

        pos_theta = x + sigma * noise
        neg_theta = x - sigma * noise

        if clip:
            pos_theta = pos_theta.clip(bounds_lower, bounds_upper)
            neg_theta = neg_theta.clip(bounds_lower, bounds_upper)

        pos_preds = model.predict(pos_theta)
        pos_loss = BE.categorical_crossentropy( to_categorical(np.argmax(pos_preds), len(pos_preds[0])), pos_preds[0] )
        neg_preds = model.predict(neg_theta)
        neg_loss = BE.categorical_crossentropy( to_categorical(np.argmax(neg_preds), len(neg_preds[0])), neg_preds[0] )
#         pos_loss = loss_fn(pos_theta)
#         neg_loss = loss_fn(neg_theta)

        gradient += (pos_loss - neg_loss) * noise

    gradient /= 2 * sigma * 2 * samples

    return gradient

# 根据某一类的置信度估计梯度
def PreGradientEstimator(samples, sigma, model, x, bounds, noise_mu, nise_std, top_pred_idx, clip=True):
#     value = loss_fn(x)
    x = tf.convert_to_tensor(x)
    gradient = np.zeros_like(x)
    bounds_lower, bounds_upper = bounds
    for k in range(samples // 2):
        noise = np.random.normal(noise_mu, nise_std, x.shape)

        pos_theta = x + sigma * noise
        neg_theta = x - sigma * noise

        if clip:
            pos_theta = pos_theta.clip(bounds_lower, bounds_upper)
            neg_theta = neg_theta.clip(bounds_lower, bounds_upper)

        pos_preds = model.predict(pos_theta)
#         pos_loss = BE.categorical_crossentropy( to_categorical(np.argmax(pos_preds), len(pos_preds[0])), pos_preds[0] )
        pos_loss = pos_preds[:, top_pred_idx]
        neg_preds = model.predict(neg_theta)
#         neg_loss = BE.categorical_crossentropy( to_categorical(np.argmax(neg_preds), len(neg_preds[0])), neg_preds[0] )
        neg_loss = neg_preds[:, top_pred_idx]
#         pos_loss = loss_fn(pos_theta)
#         neg_loss = loss_fn(neg_theta)

        gradient += (pos_loss - neg_loss) * noise

    gradient /= 2 * sigma * 2 * samples

    return gradient

#     输出层对模型的某一层求导
def get_hidden_layer_gradient(x_input, model, pre_conf, layers_names):
    top_pred_idx = np.argmax(pre_conf)
    hidden_layer = model.get_layer(layers_names).output
    grads = BE.gradients(loss = model.layers[-1].output[:, top_pred_idx], variables = hidden_layer)
    get_gradients = BE.function(inputs=model.inputs[0], outputs=grads)
    layer_grad = get_gradients(x_input)[0]
    return layer_grad
    
#     损失函数对模型的某一层求导
def get_hidden_layer_loss_gradient(x_input, model, pre_conf, layers_names):
    top_pred_idx = np.argmax(pre_conf)
    num_class = np.shape(pre_conf)[-1]
    hidden_layer = model.get_layer(layers_names).output
    loss = BE.categorical_crossentropy(to_categorical(top_pred_idx, num_class), model.layers[-1].output[:,:][0])
#     loss = BE.categorical_crossentropy(to_categorical(label_tmp, num_class), pre_conf[0]) # 用 preds[0]会报NoneType错
    grads = BE.gradients(loss = loss, variables = hidden_layer)
    get_gradients = BE.function(inputs=model.inputs[0], outputs=grads)
    layer_grad = get_gradients(x_input)[0]
    return layer_grad

# def GradientEstimator_layer(samples, sigma, model, x, bounds, noise_mu, nise_std, layer_name, clip=True):
# #     value = loss_fn(x)
#     model_hidden_layer = Model(inputs=model.input, outputs=model.get_layer(layers_names).output)
#     gradient = np.zeros_like(model_hidden_layer)
#     bounds_lower, bounds_upper = bounds
#     for k in range(samples // 2):
#         noise = np.random.normal(noise_mu, nise_std, x.shape)

#         pos_theta = x + sigma * noise
#         neg_theta = x - sigma * noise

#         if clip:
#             pos_theta = pos_theta.clip(bounds_lower, bounds_upper)
#             neg_theta = neg_theta.clip(bounds_lower, bounds_upper)

#         pos_preds = model.predict(pos_theta)
#         pos_features = model_hidden_layer.predict(pos_theta)
#         pos_loss = BE.categorical_crossentropy( to_categorical(np.argmax(pos_preds), len(pos_preds[0])), pos_preds[0] )
        
#         neg_preds = model.predict(neg_theta)
#         neg_features = model_hidden_layer.predict(neg_theta)
#         neg_loss = BE.categorical_crossentropy( to_categorical(np.argmax(neg_preds), len(neg_preds[0])), neg_preds[0] )
        
#         noise_feature = (pos_features-neg_features)/(2*sigma)
# #         pos_loss = loss_fn(pos_theta)
# #         neg_loss = loss_fn(neg_theta)
#         gradient += (pos_loss - neg_loss) * noise_feature
#     gradient /= 2 * sigma * 2 * samples
#     return gradient

def get_gradients(img_input, model, top_pred_idx):
    images = tf.cast(img_input, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(images)
        preds = model(images)
        top_class = preds[:, top_pred_idx]

    grads = tape.gradient(top_class, images)
    return grads

def get_gradients_regression_one_out(img_input, model):
    images = tf.cast(img_input, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(images)
        preds = model(images)

    grads = tape.gradient(preds, images)
    return grads

# def get_loss_gradients_regression_one_out(img_input, model, epsilon):
#     images = tf.cast(img_input, tf.float32)
#     cce = tf.keras.losses.MeanSquaredError(reduction=losses_utils.ReductionV2.AUTO, name='mean_squared_error')
#     with tf.GradientTape() as tape:
#         tape.watch(images)
#         preds = model(images)
#         loss = cce(preds + epsilon, preds)
# #         top_class = preds[:, top_pred_idx]

#     grads = tape.gradient(loss, images)
#     return grads

def getfile_name(file_dir):   
    for root, dirs, files in os.walk(file_dir):
        return files
#          print(root) #当前目录路径  
#          print(dirs) #当前路径下所有子目录  
#          print(files) #当前路径下所有非目录子文件 

def getLayerIndexByName(model, layername):
    for idx, layer in enumerate(model.layers):
        if layer.name == layername:
            return idx
        
# PRIMA 中使用的函数
def cos_distribution(cos_array):
    cos_distribute = [0 for i in range(10)]
    for i in cos_array:
        if i >= 0 and i < 0.1:
            cos_distribute[0] += 1
        elif i >= 0.1 and i < 0.2:
            cos_distribute[1] += 1
        elif i >= 0.2 and i < 0.3:
            cos_distribute[2] += 1
        elif i >= 0.3 and i < 0.4:
            cos_distribute[3] += 1
        elif i >= 0.4 and i < 0.5:
            cos_distribute[4] += 1
        elif i >= 0.5 and i < 0.6:
            cos_distribute[5] += 1
        elif i >= 0.6 and i < 0.7:
            cos_distribute[6] += 1
        elif i >= 0.7 and i < 0.8:
            cos_distribute[7] += 1
        elif i >= 0.8 and i < 0.9:
            cos_distribute[8] += 1
        elif i >= 0.9 and i <= 1.0:
            cos_distribute[9] += 1
    return cos_distribute

# PRIMA 中使用的函数
def generate_ratio_vector(num,ratio):
    import math
    perturbate_num = math.ceil(num * ratio)
    non_perturbate_num = num - perturbate_num
    a = np.zeros(perturbate_num)+1
    b = np.zeros(non_perturbate_num)
    a_b = np.concatenate((a,b), axis=0)
    np.random.shuffle(a_b)
    return a_b
   
# PRIMA 中使用的函数
def black(image,i=0,j=0):
    image = np.array(image, dtype=float)
    image[0+2*i:2+2*i,0+2*j:2+2*j]=0
    return image.copy()

# PRIMA 中使用的函数
def white(image,i=0,j=0):
    image = np.array(image, dtype=float)
    image[0+2*i:2+2*i,0+2*j:2+2*j]=255
    return image.copy()

# PRIMA 中使用的函数
def reverse_color(image,i=0,j=0):
    image = np.array(image, dtype=float)
    part = image[0+2*i:2+2*i,0+2*j:2+2*j].copy()
    reversed_part = 255-part
    image[0+2*i:2+2*i,0+2*j:2+2*j] = reversed_part
    return image

# PRIMA 中使用的函数   
def gauss_noise(image,i=0,j=0,mean=0, var=0.1,ratio=1.0):
    image = np.array(image, dtype=float)
    image = image.astype('float32') / 255
    part = image[0+2*i:2+2*i,0+2*j:2+2*j].copy()
    ratio_vector = generate_ratio_vector(len(part.ravel()),ratio).reshape(part.shape)
    noise = np.random.normal(mean, var ** 0.5, part.shape)
    noise = noise * ratio_vector
    image[0+2*i:2+2*i,0+2*j:2+2*j] += noise
    image = np.clip(image, 0, 1)
    image *= 255
    return image.copy()

# PRIMA 中使用的函数
def shuffle_pixel(image,i=0,j=0):
    image = np.array(image, dtype=float)
    # image /= 255
    part = image[0+2*i:2+2*i,0+2*j:2+2*j].copy()
    part_r = part.reshape(-1,1)
    np.random.shuffle(part_r)
    part_r = part_r.reshape(part.shape)
    image[0+2*i:2+2*i,0+2*j:2+2*j] = part_r
    return image

# PRIMA 中使用的函数:翻转某些神经元的激活状态
def NeuActInverse_confidence(img_input, hidden_layer_Model, partial_model_Model, mutant_rate):
    hidden_layer_output = hidden_layer_Model(img_input)  # 获取模型的某一层输出
    position_NAI = np.zeros_like(hidden_layer_output)  # 对隐层的激活输出的mutant_rate%进行翻转
    position_NAI = position_NAI.reshape(-1)
    position_NAI[:int(mutant_rate*len(position_NAI))] = -1
    np.random.shuffle(position_NAI)
    position_NAI = position_NAI.reshape( np.shape(hidden_layer_output) ) 
    hidden_layer_NAI = hidden_layer_output * position_NAI
    preds = partial_model_Model.predict(hidden_layer_NAI)
    return preds

# PRIMA 中使用的函数:将某些神经元的激活值置零
def NeuEffBlock_confidence(img_input, hidden_layer_Model, partial_model_Model, mutant_rate):
    hidden_layer_output = hidden_layer_Model(img_input)  # 获取模型的某一层输出
    position_NAI = np.ones_like(hidden_layer_output)
    position_NAI = position_NAI.reshape(-1)
    position_NAI[:int(mutant_rate*len(position_NAI))] = 0
    np.random.shuffle(position_NAI)
    position_NAI = position_NAI.reshape( np.shape(hidden_layer_output) ) 
    hidden_layer_NAI = hidden_layer_output * position_NAI
    preds = partial_model_Model.predict(hidden_layer_NAI)
    return preds

# PRIMA 中使用的函数:在某层权重上添加高斯噪声
def GaussFuzz_confidence(img_input, model, mean, var, layer_name):
    model_tmp = clone_model(model)
    weight_tmp = model_tmp.get_layer(layer_name).get_weights()
    noise = np.random.normal(mean, var, np.shape(weight_tmp[0]) )
    weight_tmp[0] = weight_tmp[0] + noise
    model_tmp.get_layer(layer_name).set_weights(weight_tmp)
    preds = model_tmp.predict(img_input)
    del model_tmp
    return preds

# PRIMA 中使用的函数:shuffling某一层的部分权重值
def WeightShuffl_confidence(img_input, model, layer_name, mutant_rate):
    model_tmp = clone_model(model)
    weight_tmp = model_tmp.get_layer(layer_name).get_weights()
    w1 = weight_tmp[0].copy()
    pos1 = np.zeros_like(w1)
    pos1 = pos1.reshape(-1)
    pos1[:int(mutant_rate*len(pos1))] = 1
    np.random.shuffle(pos1)
    pos1 = pos1.reshape(weight_tmp[0].shape)
    np.random.shuffle(w1)
    pos2 = 1-pos1
    weight_tmp[0] = weight_tmp[0]*pos2 + w1*pos1
    model_tmp.get_layer(layer_name).set_weights(weight_tmp)
    preds = model_tmp.predict(img_input)
    del model_tmp
    return preds

# # 用不同的颜色绘制数据点
# mark = ['or', 'og']
# for i, d in enumerate(data):
#     plt.plot(d[0], d[1], mark[ground_truth_cluster[i]])
# # 画出各个分类的中心点
# mark = ['*b', '*y']
# for i, center in enumerate(centers):
#     plt.plot(center[0], center[1], mark[i], markersize=20)

# plt.figure()
# hang = 4
# lie = 5
# for i in range(0,hang):
#     for j in range(0,lie):
#         plt.subplot(hang,lie,i*lie+j+1)
#         plt.imshow(X_test[i*lie+j].reshape((28,28,1)))
# #         print(np.argmax(model.predict(X_test_part[i*lie+j].reshape(-1, 28*28))))