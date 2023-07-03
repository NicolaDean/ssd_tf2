rm -rf train
rm -rf test
rm -rf valid
#curl -L https://public.roboflow.com/ds/VxmeW4uEgZ?key=cuvW694dyg > roboflow.zip; unzip roboflow.zip; rm roboflow.zip
wget https://public.roboflow.com/ds/VxmeW4uEgZ?key=cuvW694dyg -O roboflow.zip; unzip roboflow.zip; rm roboflow.zip