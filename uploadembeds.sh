#!/bin/sh

API_ENDPOINT="http://127.0.0.1:3000/embeddings"

jq -c '.[]' dataset.json | while read item; do
    item_as_array="[$item]"
    curl -X POST -H "Content-Type: application/json" -d "${item_as_array}" ${API_ENDPOINT}
    #echo ${item}
done