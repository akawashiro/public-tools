#! /bin/bash -eux

ollama_container_name=ollama_container_service
ollama_dir=/home/akira/.cache/ollama_container_service
mkdir -p ${ollama_dir}

# Because jendeley use this service, you must not change the port.
docker run \
    --publish 11434:11434 \
    --name ${ollama_container_name} \
    --rm \
    --detach \
    --gpus=all \
    -v ${ollama_dir}:/root/.ollama \
    ollama/ollama
sleep 2

# jendeley use llama3.2 model, so we need to pull it first.
curl http://localhost:11434/api/pull -d '{
  "name": "llama3.2"
}'
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2",
  "prompt": "What is 1 + 1?",
  "stream": false
}'

echo "ollama server is ready to use. Press Ctrl+C to stop the server."
docker attach ${ollama_container_name}
