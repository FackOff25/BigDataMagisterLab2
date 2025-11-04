FROM ubuntu:latest
RUN groupadd hadoop
RUN useradd -g hadoop hduser

RUN mkdir /scripts

WORKDIR /scripts

RUN echo "hduser   ALL=(ALL:ALL) ALL" >> /etc/sudoers

COPY ./*.xml /opt/
COPY ./envs.txt /opt/
COPY ./spark-env.sh /opt/
COPY ./hadoop-2.9.2.tar.gz /opt/

RUN apt-get update
RUN apt-get install -y openjdk-8-jdk wget gcc make ssh openssh-client openssh-server zlib1g-dev libssl-dev curl
RUN apt-get install -y pip
RUN apt-get install -y unzip
RUN mkdir -p /home/hduser/.ssh/
RUN chown hduser:hadoop -R /home/hduser/

USER hduser
RUN ssh-keygen -t rsa -P "" -f /home/hduser/.ssh/id_rsa
RUN cp /home/hduser/.ssh/id_rsa.pub /home/hduser/.ssh/authorized_keys

USER root

# Hadoop
RUN cd /opt && tar -xzf ./hadoop-2.9.2.tar.gz && mkdir -p /usr/local/hadoop && mv hadoop-2.9.2 /usr/local/hadoop
RUN mkdir -p /usr/local/hadoop/hadoop_tmp/hdfs/namenode
RUN mkdir -p /usr/local/hadoop/hadoop_tmp/hdfs/datanode
RUN chown hduser:hadoop -R /usr/local/hadoop/

RUN cat /opt/envs.txt >> /home/hduser/.bashrc
RUN echo "export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-amd64" >> /usr/local/hadoop/hadoop-2.9.2/etc/hadoop/hadoop-env.sh
RUN head -n -2 /usr/local/hadoop/hadoop-2.9.2/etc/hadoop/core-site.xml > /usr/local/hadoop/hadoop-2.9.2/etc/hadoop/core-site.xml
RUN head -n -3 /usr/local/hadoop/hadoop-2.9.2/etc/hadoop/hdfs-site.xml > /usr/local/hadoop/hadoop-2.9.2/etc/hadoop/hdfs-site.xml
RUN head -n -5 /usr/local/hadoop/hadoop-2.9.2/etc/hadoop/yarn-site.xml > /usr/local/hadoop/hadoop-2.9.2/etc/hadoop/yarn-site.xml
RUN head -n -3 /usr/local/hadoop/hadoop-2.9.2/etc/hadoop/mapred-site.xml.template > /usr/local/hadoop/hadoop-2.9.2/etc/hadoop/mapred-site.xml
RUN echo $(cat /opt/core-site.xml) >> /usr/local/hadoop/hadoop-2.9.2/etc/hadoop/core-site.xml
RUN echo $(cat /opt/hdfs-site.xml) >> /usr/local/hadoop/hadoop-2.9.2/etc/hadoop/hdfs-site.xml
RUN echo $(cat /opt/yarn-site.xml) >> /usr/local/hadoop/hadoop-2.9.2/etc/hadoop/yarn-site.xml
RUN echo $(cat /opt/mapred-site.xml) >> /usr/local/hadoop/hadoop-2.9.2/etc/hadoop/mapred-site.xml

#Spark
RUN pip3 install pyspark==3.5.0 --break-system-packages
RUN pip3 install py4j --break-system-packages
