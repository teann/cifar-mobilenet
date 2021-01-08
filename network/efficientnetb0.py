from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf 
import numpy as np 


CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        # EfficientNet actually uses an untruncated normal distribution for
        # initializing conv layers, but keras.initializers.VarianceScaling use
        # a truncated distribution.
        # We decided against a custom initializer for better serializability.
        'distribution': 'normal'
    }
}

class efficientnetb0(object):
	"""docstring for MobileNet"""
	def __init__(self,is_training=True,keep_prob=0.5,num_classes=10):
		super(efficientnetb0, self).__init__()
		self.num_classes = num_classes
		self.is_training = is_training
		self.conv_num = 0
		self.weight_decay = 4e-5 ## remove weight decay or tune hyperparameter
		self.regularizer = tf.contrib.layers.l2_regularizer(scale=self.weight_decay) ## might need to switch to glorot_uniform in mb block
		self.initializer = tf.contrib.layers.variance_scaling_initializer()
		self.keep_prob = keep_prob
		self.run_name = 'efficientnetb0'

	def swish(self, inputs):
		return tf.nn.swish(inputs)

	def weight_variable(self, shape):
		initial = tf.truncated_normal(shape, stddev=0.1)
		return tf.Variable(initial)



	def mb_conv_block(self, inputs, activation, expand_ratio, in_channel, out_channel, kernel_size=1, stride=1):

		##EXPANSION PHASE
		if expand_ratio == 1:
			return inputs
		else:
			filters = in_channel * expand_ratio
			inputs = tf.keras.layers.Conv2D(filters, 1, strides=1, padding='same', use_bias=False, name="expand_conv_"+str(self.conv_num))(inputs)
			inputs = tf.keras.layers.BatchNormalization(name='bn_expand_'+str(self.conv_num))(inputs,training=self.is_training)
			inputs = self.swish(inputs)

		## DEPTHWISE CONVOLUTION
		depthwise = tf.keras.layers.DepthwiseConv2D(kernel_size, strides=stride, padding='same', use_bias=False, name="depthwise_conv_"+str(self.conv_num))(inputs)
		inputs = tf.keras.layers.BatchNormalization(name='bn_dw_'+str(self.conv_num))(depthwise,training=self.is_training)
		inputs = self.swish(inputs)

		##SQUEEZE AND EXCITATION PHASE
		se_ratio = 0.25
		num_reduced_filters = max(1, int(in_channel * se_ratio))
		se_tensor = tf.keras.layers.GlobalAveragePooling2D(name="se_tensor_squeeze_" + str(self.conv_num))(inputs)
		target_shape = (1, 1, se_tensor.shape[1])
		se_tensor = tf.keras.layers.Reshape(target_shape, name="se_tensor_reshape_" + str(self.conv_num))(se_tensor)
		se_tensor = tf.keras.layers.Conv2D(num_reduced_filters, 1, padding='same', use_bias=True, name='conv_se_reduce_'+str(self.conv_num))(se_tensor)
		se_tensor = self.swish(se_tensor) 
		se_tensor = tf.keras.layers.Conv2D(filters, 1, padding='same', use_bias=True, name='conv_se_expand_'+str(self.conv_num))(se_tensor)
		se_tensor = tf.math.sigmoid(se_tensor)
		x = tf.math.multiply(se_tensor, inputs, name='se_excite_' + str((self.conv_num)))

		##Output shape
		x = tf.keras.layers.Conv2D(out_channel, 1, padding='same', use_bias=False, name='conv_project_'+str(self.conv_num))(x)
		x = tf.keras.layers.BatchNormalization(name='project_bn_'+str(self.conv_num))(x,training=self.is_training)

		self.conv_num+=1

		return x





	def _variable_with_weight_decay(self, name, shape,wd):
		var = tf.get_variable(name,shape,initializer=self.initializer,dtype=tf.float32)
		if wd is not None:
			weight_decay = tf.multiply(tf.nn.l2_loss(var),wd,name='weight_loss')
			tf.add_to_collection('losses',weight_decay)
		return var 


	def forward(self,inputs):
		out0 = tf.keras.layers.Conv2D(32, 3, strides=(1,1), padding = 'same', use_bias=False, name="stem_conv")(inputs)
		print('strides at 1')
		out0 = tf.keras.layers.BatchNormalization(name='stem_bn')(out0,training=self.is_training)
		out0 = self.swish(out0)
		out1 = self.mb_conv_block(out0, activation='swish', expand_ratio=1, in_channel=32, out_channel=16, kernel_size=3, stride=1)
		out2 = self.mb_conv_block(out1, activation='swish', expand_ratio=6, in_channel=16, out_channel=24, kernel_size=3, stride=2)
		out3 = self.mb_conv_block(out2, activation='swish', expand_ratio=6, in_channel=24, out_channel=40, kernel_size=5, stride=2)
		out4 = self.mb_conv_block(out3, activation='swish', expand_ratio=6, in_channel=40, out_channel=80, kernel_size=3, stride=2)
		out5 = self.mb_conv_block(out4, activation='swish', expand_ratio=6, in_channel=80, out_channel=112, kernel_size=5, stride=1)
		out6 = self.mb_conv_block(out5, activation='swish', expand_ratio=6, in_channel=112, out_channel=192, kernel_size=5, stride=2)
		out7 = self.mb_conv_block(out6, activation='swish', expand_ratio=6, in_channel=192, out_channel=320, kernel_size=3, stride=1)
		# out8 = self.conv2d(out7,out_channel=1280,kernel_size=1,stride=1)
		out8 = tf.keras.layers.GlobalAveragePooling2D(name="top_adapt")(out7)
		print('NO AVERAGE POOL')
		print('LAST FC LAYER DEACTIVATED')
		out9 = tf.layers.flatten(out8)
		out10 = tf.layers.dropout(out9,rate=self.keep_prob)
		predicts = tf.layers.dense(out10,units=self.num_classes,kernel_initializer=self.initializer,kernel_regularizer=self.regularizer,name='fc')
		softmax_out = tf.nn.softmax(predicts,name='output')
		gradient_to_binarized = tf.gradients(ys=softmax_out, xs=out1)
		return gradient_to_binarized, predicts,softmax_out

	def make_layer(self,inputs,repeat=5):
		for i in range(repeat):
			inputs = self.separable_conv2d(inputs,out_channel=512,kernel_size=[3,3,512,1],stride=1)
		return inputs

	def loss(self,predicts,labels):
		losses = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels,predicts))
		l2_reg = tf.losses.get_regularization_losses()
		lr_reg2 = tf.get_collection('losses')
		losses+=tf.add_n(l2_reg)
		losses+=tf.add_n(lr_reg2)
		return losses



if __name__=='__main__':
	with tf.device('/CPU:0'):
		net = efficientnetb0()
		data = np.random.randn(64,32,32,3)
		inputs = tf.placeholder(tf.float32,[64,32,32,3])
		grads, predicts,softmax_out = net.forward(inputs)
		config = tf.ConfigProto(allow_soft_placement=True)
		config.gpu_options.allow_growth=True
		init = tf.global_variables_initializer()
		sess = tf.Session(config=config)
		sess.run(init)
		output = sess.run(predicts,feed_dict={inputs:data})
		print(output.shape)
		sess.close()

