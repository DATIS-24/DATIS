import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
from keras.models import load_model
from keras import Model
import copy
from DATIS.DATIS import DATIS_test_input_selection,DATIS_redundancy_elimination
from keras.datasets import mnist


def get_faults(sample, mis_ind_test, Clustering_labels):
    i=0
    pos=0
    neg=0
    i=0
    cluster_lab=[]
    nn=-1
    for l in sample:
        if l in mis_ind_test:
            neg=neg+1 
            ind=list(mis_ind_test).index(l)
            if (Clustering_labels[ind]>-1):
                cluster_lab.append(Clustering_labels[ind])
            if (Clustering_labels[ind]==-1):
                cluster_lab.append(nn)
                nn=nn-1
        else:
            pos=pos+1

    faults_n=len(list(set(cluster_lab)))
    #All noisy mispredicted inputs are considered as one specific fault
        
    cluster_1noisy=copy.deepcopy(cluster_lab)
    for i in range(len(cluster_1noisy)):
        if cluster_1noisy[i] <=-1:
            cluster_1noisy[i]=-1
    faults_1noisy=len(list(set(cluster_1noisy)))
    return faults_n,faults_1noisy, neg


def calculate_rate(budget_ratio_list,test_support_output,x_test,rank_lst,ans,cluster_path):
   
    
    top_list =[]
    for ratio_ in budget_ratio_list:
        top_list.append(int(len(x_test)*ratio_))
    result_fault_rate= []
    clustering_labels = np.load(cluster_path+'/cluster1.npy')
    fault_sum_all = np.max(clustering_labels)+1+np.count_nonzero(clustering_labels == -1)
    mis_test_ind= np.load(cluster_path+'/mis_test_ind.npy')
    

    print('total test case:{len}')
   
    for i_, n in enumerate(top_list):
        if len(ans)!=0:
            n_indices =ans[i_]
            subset = np.stack([test_support_output[index] for index in ans[i_]])
        else :
            n_indices = rank_lst[:n] 
            subset = test_support_output[rank_lst[:n], :]

        
        n_fault,n_noisy,n_neg = get_faults(n_indices,mis_test_ind,clustering_labels)

        faults_rate = n_fault/min(n,fault_sum_all)
        
 
        print(f"The Fault Detection Rate of Top: {n} cases :{faults_rate}")
        result_fault_rate.append(faults_rate)
    return 

def calculate_sce(predictions, labels, n_bins=15):
    
    probability = predictions
    n_data = len(predictions)
    n_class = len(predictions[0])
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    confidences = np.max(predictions, axis=1)
    predictions = np.argmax(predictions, axis=1)
    accuracies = np.equal(predictions,labels)

    idx = np.arange(n_data)
        #make matrices of zeros
    pred_matrix = np.zeros([n_data,n_class])
    label_matrix = np.zeros([n_data,n_class])
        #self.acc_matrix = np.zeros([self.n_data,self.n_class])
    pred_matrix[idx,predictions] = 1
    label_matrix[idx,labels] = 1

    acc_matrix = np.equal(pred_matrix, label_matrix)

    sce = 0

    for i in range(n_class):
        bin_prop = np.zeros(n_bins)
        bin_acc = np.zeros(n_bins)
        bin_conf = np.zeros(n_bins)
        bin_score = np.zeros(n_bins)

        confidences = probability[:,i]
        accuracies = acc_matrix[:,i]

        for i, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
            # Calculated |confidence - accuracy| in each bin
            in_bin = np.greater(confidences,bin_lower.item()) * np.less_equal(confidences,bin_upper.item())
            bin_prop[i] = np.mean(in_bin)

            if bin_prop[i].item() > 0:
                bin_acc[i] = np.mean(accuracies[in_bin])
                bin_conf[i] = np.mean(confidences[in_bin])
                bin_score[i] = np.abs(bin_conf[i] - bin_acc[i])
        
        sce += np.dot(bin_prop,bin_score)

    return sce/n_class



def load_data():  
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_test = x_test.reshape(-1, 28, 28, 1)
        x_train = x_train.reshape(-1, 28, 28, 1)
      
        x_test = x_test.astype('float32')
        x_train = x_train.astype('float32')
        x_train /= 255
        x_test /= 255
        return (x_train, y_train), (x_test, y_test)
def load_data_corrupted():
    data_corrupted_file = "./corrupted_data/mnist/data_corrupted.npy"
    label_corrupted_file = "./corrupted_data/mnist/label_corrupted.npy"
    data_corrupted_file = "./corrupted_data/mnist/data_corrupted.npy"
    label_corrupted_file = "./corrupted_data/mnist/label_corrupted.npy"
    x_test_ood = np.load(data_corrupted_file)
    y_test_ood = np.load(label_corrupted_file)
    y_test_ood = y_test_ood.reshape(-1)
    x_test_ood = x_test_ood.reshape(-1, 28, 28, 1)
    x_test_ood = x_test_ood.astype('float32')
    x_test_ood /= 255
    return x_test_ood,y_test_ood
   
     


def demo(data_type):
   
    if data_type =='nominal':
        (x_train, y_train), (x_test, y_test) = load_data()
        cluster_path ='./cluster_data/LeNet5_mnist_nominal'
    elif data_type == 'corrupted':
        (x_train, y_train), (x_test, y_test)= load_data()
        x_test, y_test = load_data_corrupted()
        cluster_path ='./cluster_data/LeNet5_mnist_corrupted'

    model_path = "./model/model_mnist_LeNet5.hdf5"

    ori_model = load_model(model_path)
     
    
    new_model = Model(ori_model.input, outputs=ori_model.layers[-2].output)
    train_support_output = new_model.predict(x_train)
    train_support_output= np.squeeze(train_support_output)
    test_support_output=new_model.predict(x_test) 
    test_support_output= np.squeeze(test_support_output)
    softmax_test_prob = ori_model.predict(x_test)
       
    ece_softmax =calculate_sce(softmax_test_prob,y_test,15)
        
    rank_lst,my_test_prob =DATIS_test_input_selection(softmax_test_prob,train_support_output,y_train,test_support_output,y_test,10)
    
    my_ece =calculate_sce(my_test_prob,y_test,15)

    print("softmax_sce: {:.3f}, our_sce: {:.3f}".format(ece_softmax, my_ece))

    budget_ratio_list =[0.005,0.01,0.02,0.03,0.05,0.1]

    ans= DATIS_redundancy_elimination(budget_ratio_list,rank_lst,test_support_output,y_test)

    calculate_rate(budget_ratio_list,test_support_output,x_test,rank_lst,ans,cluster_path)

    
    

if __name__ == '__main__':
   
    demo('nominal')
    print("         =====================================           ")
    demo('corrupted')
    

   
