export DEBIAN_FRONTEND=noninteractive  

add-apt-repository -y ppa:deadsnakes/ppa 
apt install -y python3.12

if [ -f "/autograder/source/requirements.txt" ]; then
    apt install -y python3.12-distutils
    wget https://bootstrap.pypa.io/get-pip.py
    python3.12 get-pip.py
    pip3.12 install -r /autograder/source/requirements.txt
fi