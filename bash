
sudo apt-get update
sudo apt install docker.io
sudo docker run --rm -p 8787:8787 -e PASSWORD=ferro1982 rocker/tidyverse

mkdir repos

git clone https://github.com/gefero/ML_imputation