#!/bin/bash

echo '*** Install Jupyter lab extensions ***'
echo 'Install Node.js (preprocessing)'

pip install --upgrade jupyterlab
# Node.jsの導入(バージョンは適宜修正してください)
pip install curl
curl -sL https://deb.nodesource.com/setup_14.x |bash -
apt-get install -y --no-install-recommends nodejs

# Node.jsの最新版が上手く入らないので，以下のようにして対応
apt install -y nodejs npm
npm install n -g
n stable
apt purge -y nodejs npm

echo 'Install extensions'

# 変数名や型、内容を常に横に表示しておける. デバッグのお供になるかも?
jupyter labextension install @lckr/jupyterlab_variableinspector

# Tensorboard 連携
# TODO エラー吐くけど動くので放置
pip install jupyter-tensorboard
jupyter labextension install jupyterlab_tensorboard
jupyter serverextension enable --py jupyterlab_tensorboard

# plotly用jupyter連携用extension
jupyter labextension install jupyterlab-plotly@4.14.3

jupyter labextension install @hokyjack/jupyterlab-monokai-plus
jupyter labextension install @ryantam626/jupyterlab_code_formatter
jupyter labextension install @jupyterlab/toc
jupyter labextension install jupyterlab-vimrc
jupyter labextention install @axlair/jupyterlab_vim
jupyter serverextension enable --py jupyterlab_code_formatter

wget https://linux.kite.com/dls/linux/current
chmod 777 current
sed -i 's/"--no-launch"//g' current > /dev/null
./current --install ./kite-installer

# Jupyter 起動
# 起動したらそのコンテナは放っておけば良い
# 以下のコマンドを実行する (ポート番号(e.g. 8888)は各自でお願いいたします)
jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --notebook-dir='/root'
