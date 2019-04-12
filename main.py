import numpy as np

import warnings
warnings.filterwarnings("ignore")

import sys,os
sys.path.append(os.getcwd())
from meta_data import DataSet, meta_data, model_select, cal_meta_data, cal_meta_data_sequence

if __name__ == "__main__":
    
    dataset_path = './newdata/'
    # datasetnames = np.load('datasetname.npy')
    # datasetnames = ['echocardiogram', 'heart', 'heart-hungarian', 'heart-statlog', 'house',
    #                     'house-votes', 'spect', 'statlog-heart', 'vertebral-column-2clases']
    # 'wdbc', 'clean1', 'ethn', , 'blood', 'breast-cancer-wisc'
    datasetnames = ['australian']
    # Different types of models, each type has many models with different parameters
    # modelnames = ['KNN', 'LR', 'RFC', 'RFR', 'DTC', 'DTR', 'SVM', 'GBC', 'ABC', 'ABR']
    modelnames = ['LR']

    # in the same dataset and the same ratio of initial_label_rate,the number of split.
    split_count = [90]
    # The number of unlabel data to select to generate the meta data.
    num_xjselect = 30

    diff_five_round = 20

    n_labelleds = np.arange(2, 102, 5)
    os.system("mkdir dataset_split_count90")
    # first choose a dataset
    for datasetname in datasetnames:
        os.system("mkdir dataset_split_count90/" + datasetname)
        os.system("mkdir dataset_split_count90/" + datasetname + "/split_count")

    # first choose a dataset
    for datasetname in datasetnames:
    
        dataset = DataSet(datasetname, dataset_path)
        X = dataset.X
        y = dataset.y
        distacne = dataset.get_distance()
        _, cluster_center_index = dataset.get_cluster_center()
        print(datasetname + ' DataSet currently being processed********************************************')
        # run multiple split on the same dataset
        # every time change the value of initial_label_rate
        for split_c in split_count:
            for n_labelled in n_labelleds:
                metadata = None
                # trains, tests, label_inds, unlabel_inds = dataset.split_data_by_nlabelled(n_labelled, test_ratio=0.6, split_count=split_count, saving_path='./n_labelled_split_info')
                trains, tests, label_inds, unlabel_inds = dataset.split_data_by_nlabelled_fulldataset(n_labelled, test_ratio=0.5, split_count=split_c)
                for t in range(split_c):
                    meta_data = cal_meta_data_sequence(X, y, distacne, cluster_center_index, modelnames,  
                        tests[t], label_inds[t], unlabel_inds[t], t, num_xjselect, diff_five_round)
                    if metadata is None:
                        metadata = meta_data
                    else:
                        metadata = np.vstack((metadata, meta_data))       

                # np.save('./bigmetadata/datasetname/split_count/'+str(n_labelled)+datasetname +str(split_count)+ '_big_metadata.npy', metadata) 
                np.save('./dataset_split_count90/'+datasetname+'/split_count/'+str(n_labelled)+datasetname +str(split_count)+ '_big_metadata'+str(num_xjselect)+'.npy', metadata)           
                           

            print(datasetname + ' is complete and saved successfully.')
            # np.save('./bigmetadata/'+datasetname + '_big_metadata.npy', metadata)

    print("All done!")
