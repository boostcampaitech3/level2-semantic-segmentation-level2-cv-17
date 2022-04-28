apt-get -y install curl
apt-get -y install libcurl3
apt-get -y install libcurl4-openssl-dev

pip install fiftyone==0.15.0.2

# if you already install fiftyone 0.15.1 
# reference below codes
# 1. pip uninstall fiftyone fiftyone-brain fiftyone-db
# 2. delete .fiftyone 
# 3. pip install fiftyone==0.15.0.2