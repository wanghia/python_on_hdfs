# !/bin/bash


PICO_DIR=/home/wanghai/local/picotools/picotools/pico_auto/bins/rb-0-3-0/output/bin

yarn jar ./hadoop-patch.jar com.tfp.hadoop.yarn.launcher.Client \
  --appname sc90 --jar ./hadoop-patch.jar \
  --shell_command "./run.sh " \
  --queue pico \
  --container_memory=10240 \
  --num_containers=1 \
  --container_vcores 30 \
  --shell_env HADOOP_USER_NAME=`whoami` \
  --shell_env WEBHDFS_USER=`whoami` \
  --file ${PICO_DIR}/yarn_wrapper \
  --file ./run.sh \
  --file ./main.py \
  --file ./meta_data.py
#--shell_env HADOOP_HOME=/usr/hdp/current/hadoop-client \
#--shell_env WEBHDFS_HDFS=/usr/hdp/current/hadoop-client/bin/hdfs \
