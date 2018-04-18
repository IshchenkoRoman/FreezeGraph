python -m tensorflow.python.tools.optimize_for_inference \
 --input=frozen_graph.pb \
  --input_binary=true \
  --frozen_graph=True \
  --output=Optimize_graph.pb\
  --output_names=y_CNN \
  --input_names=x keep_prob