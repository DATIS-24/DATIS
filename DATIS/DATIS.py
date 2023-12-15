import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Normalizer


def DATIS_test_input_selection(softmax_prob,train_support_output,y_train,test_support_output,y_test,num_classes,k=100,T=0.1):
    

    normalizer = Normalizer(norm='l2')
    train_support_output = normalizer.transform(train_support_output)
    test_support_output = normalizer.transform(test_support_output)

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_support_output, y_train)

    metrics = []
    prob_test = np.zeros((len(test_support_output), num_classes))
    
    for i, z in tqdm(enumerate(test_support_output),total=len(test_support_output)):
        distance, indices = knn.kneighbors(z.reshape(1,-1), n_neighbors=k)
        support_points = train_support_output[indices.flatten()]
        support_labels = y_train[indices.flatten()]
        distance_sum = -np.sum((z - support_points)**2, axis=1)/T 
        denominator = np.sum(np.exp(distance_sum))
        
        for j in range(num_classes):
            num_rator= np.multiply(np.exp(distance_sum),(support_labels == j))
            numerator=np.sum(num_rator)
            prob_test[i][j] = numerator/denominator

    
    softmax_max_indices = np.argmax(softmax_prob, axis=1)
    
    max_indices = np.argmax(prob_test, axis=1)
    temp = prob_test.copy()
    for i in range(len(max_indices)):
        temp[i][max_indices[i]] = -1
    second_max_indices = np.argmax(temp, axis=1)
    metrics = []
    epsilon = 1e-15
    for i in range(len(max_indices)):

        if(max_indices[i]==softmax_max_indices[i]):
            a = prob_test[i][second_max_indices[i]]
            b = prob_test[i][softmax_max_indices[i]]
           
        else:
            a = prob_test[i][max_indices[i]]
            b = prob_test[i][softmax_max_indices[i]]

        metrics.append(a / (b+epsilon))             
        
    
    rank_lst = np.argsort(metrics)
    rank_lst = rank_lst[::-1]

    return rank_lst,prob_test


def DATIS_redundancy_elimination(budget_ratio_list,rank_list,test_support_output,y_pred):
    
    size = len(test_support_output)
    normalizer = Normalizer(norm='l2')
    test = normalizer.transform(test_support_output)
    ratio_list =[0.005,0.01,0.02,0.03,0.05,0.1]
    pool_list = [3,3,2,2,2,2]
    weight_list = [0.3,0.3,0.2,0.2,0.2,0.2]
    top_list =[]
    arg_index_list =[]
    for ratio_ in budget_ratio_list:
        top_list.append(int(size*ratio_))
        for i,ratio in enumerate(ratio_list):
            if ratio_==ratio:
                arg_index_list.append(i)

    ans = []
    for i_,k in tqdm(enumerate(top_list),total=len(top_list)):
        
        index =arg_index_list[i_]
        tmp_k = int(k*pool_list[index])
       
        selected_indices = rank_list[:tmp_k]
       
        tmp_set = test[selected_indices, :]
        tmp_label =y_pred[selected_indices]
        kn = k
        if kn>100:
            kn=100
        knn = KNeighborsClassifier(n_neighbors=kn)
        knn.fit(tmp_set,tmp_label)
       
        distances = np.zeros(tmp_k)  
        for i,z in enumerate(tmp_set) :
            distance, indices = knn.kneighbors(z.reshape(1,-1), n_neighbors=kn)
            distances[i] = np.mean(distance)

        gini_weights = np.arange(tmp_k, 0, -1)  
        rank_weights = np.argsort(-distances)
        distance_weights =np.zeros(tmp_k)
        weight_k = tmp_k
        for i in rank_weights:
            distance_weights[i]= weight_k
            weight_k-=1

        weights = (1-weight_list[index])*gini_weights+weight_list[index]*distance_weights
        sorted_indices = selected_indices[np.argsort(-weights)][:int(k)]
        ans.append(sorted_indices)

    return ans


