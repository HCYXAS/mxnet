# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# pylint: disable=missing-docstring
from __future__ import print_function

import logging

import mxnet as mx
from nce import nce_loss, NceAccuracy
from random_data import DataIterNce


def get_net(num_vocab):
    data = mx.sym.Variable('data')
    label = mx.sym.Variable('label')
    label_weight = mx.sym.Variable('label_weight')
    embed_weight = mx.sym.Variable('embed_weight')
    pred = mx.sym.FullyConnected(data=data, num_hidden=100)
    ret = nce_loss(
        data=pred,
        label=label,
        label_weight=label_weight,
        embed_weight=embed_weight,
        vocab_size=num_vocab,
        num_hidden=100)
    return ret


if __name__ == '__main__':
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    batch_size = 128
    vocab_size = 10000
    feature_size = 100
    num_label = 6

    data_train = DataIterNce(100000, batch_size, vocab_size, num_label, feature_size)
    data_test = DataIterNce(1000, batch_size, vocab_size, num_label, feature_size)

    network = get_net(vocab_size)
    model = mx.mod.Module(
        symbol=network,
        data_names=[x[0] for x in data_train.provide_data],
        label_names=[y[0] for y in data_train.provide_label],
        context=[mx.gpu()]
    )

    metric = NceAccuracy()
    model.fit(
        train_data=data_train,
        eval_data=data_test,
        num_epoch=20,
        optimizer='sgd',
        optimizer_params={'learning_rate': 0.03, 'momentum': 0.9, 'wd': 0.00001},
        initializer=mx.init.Xavier(factor_type='in', magnitude=2.34),
        eval_metric=metric,
        batch_end_callback=mx.callback.Speedometer(batch_size, 50)
    )
