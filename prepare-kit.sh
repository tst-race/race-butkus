#!/bin/bash

rm -rf kit/artifacts/*

for NODE_TYPE in linux-x86_64-client \
                     linux-x86_64-server \
                     linux-arm64-v8a-client \
                     linux-arm64-v8a-server;
do
    ARTIFACT_DIR=kit/artifacts/${NODE_TYPE}/PluginButkus
    mkdir -p ${ARTIFACT_DIR}
    cp -r source/* ${ARTIFACT_DIR}
done
