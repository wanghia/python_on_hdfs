# !/bin/bash

hdfs dfs -get /user/wanghai/active/env.tar.gz
tar -xzf env.tar.gz
# # dataset
data_path='/user/wanghai/active'
hdfs dfs -get ${data_path}/newdata
# hdfs dfs -rm -r ${data_path}/dataset_query50
# # mkdir and save main.py
# hdfs dfs -put -f -p dataset_query50 ${data_path}
# hdfs dfs -put -f ./bazel-bin/run_hadoop/lgb.runfiles/__main__/run_hadoop/main.py ${data_path}/dataset_query50

# # make a note.txt
# touch note.txt
# echo "1231=1766remove group, depth=1,s=920,5folds, use 2 fold, 0.01" > note.txt
# hdfs dfs -put -f note.txt ${data_path}/result_${time}

# run and save
./miniconda3/bin/python main.py
hdfs dfs -put -f -p ./dataset_split_count90 ${data_path}
