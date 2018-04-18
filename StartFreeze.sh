#pyenv virtualenvwrapper
python -m tensorflow.python.tools.freeze_graph \
 --input_graph model.pb \
  --input_checkpoint ./CNN \
  --input_binary=true \
  --output_graph frozen_graph.pb\
  --output_node_names=y_CNN
