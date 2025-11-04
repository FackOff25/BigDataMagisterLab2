#!/bin/bash
set -e
if [ ! -f for_install/hadoop-2.9.2.tar.gz ]; then
    wget https://archive.apache.org/dist/hadoop/common/hadoop-2.9.2/hadoop-2.9.2.tar.gz -O for_install/hadoop-2.9.2.tar.gz
fi

sudo docker-compose down
sudo docker-compose up -d --build
sudo docker exec -i -t hadoop service ssh restart