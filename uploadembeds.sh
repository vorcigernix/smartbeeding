#!/bin/sh

API_ENDPOINT="http://127.0.0.1:3000/embeddings"

# Replace \n with a placeholder, process with jq, and replace the placeholder with <br>
sed 's/\\n/PLACEHOLDER/g' dataset.json | jq -rc '.[]' | sed 's/PLACEHOLDER/<br>/g' | while read item; do
    item_as_array="[$item]"
    curl -X POST -H "Content-Type: application/json" -d "${item_as_array}" ${API_ENDPOINT}
    echo "${item_as_array}\n"
done