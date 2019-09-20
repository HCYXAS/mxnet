import mxnet as mx

data = mx.sym.Variable('data')
fc1  = mx.sym.FullyConnected(data, name='fc1', num_hidden=128)
act1 = mx.sym.Activation(fc1, name='relu1', act_type="relu")
fc2  = mx.sym.FullyConnected(act1, name='fc2', num_hidden=10)
out  = mx.sym.SoftmaxOutput(fc2, name = 'softmax')
mod = mx.mod.Module(out)  # create a module by given a Symbol
mod.bind(data_shapes=nd_iter.provide_data,
          label_shapes=nd_iter.provide_label)
init = mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2)
mod.init_params(init)
for dictionary in module.get_params():
     for key in dictionary:
         print(key)
         print(dictionary[key].asnumpy())
