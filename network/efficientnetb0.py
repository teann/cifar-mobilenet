from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf 
import numpy as np 

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

	def swish(self, inputs):
		return tf.nn.swish(inputs)


	def conv2d(self,inputs,out_channel,kernel_size=1,stride=1,advance=True):
		inputs = tf.layers.conv2d(inputs,filters=out_channel,kernel_size=kernel_size,strides=stride,padding='same',
			kernel_initializer=self.initializer,kernel_regularizer=self.regularizer,name='conv_'+str(self.conv_num))
		inputs = tf.layers.batch_normalization(inputs,training=self.is_training,name='bn'+str(self.conv_num))
		if advance == True:
			self.conv_num+=1
			return self.swish(inputs)
		else:
			return self.swish(inputs)
		# return tf.nn.relu(inputs)

	def mb_conv_block(self, inputs, activation, expand_ratio, in_channel, out_channel, kernel_size=1, stride=1):
		kernel_size[2] = kernel_size[2]*expand_ratio
		filters = in_channel * expand_ratio
		##EXPANSION PHASE
		if expand_ratio != 1:
			inputs = self.conv2d(inputs,out_channel=filters,kernel_size=1,stride=1,advance=False)

			if activation == 'swish':
				inputs = self.swish(inputs)
			else:
				inputs = tf.math.sigmoid(inputs) ## might need to change tf.math to tf.keras.layers.actviation???
		else:
			inputs = inputs


		## DEPTHWISE CONVOLUTION
		scope = 'conv_'+str(self.conv_num)
		with tf.variable_scope(scope) as scope:
			kernel = self._variable_with_weight_decay('weight',shape=kernel_size,wd = self.weight_decay)
			depthwise = tf.nn.depthwise_conv2d(inputs,kernel,[1,stride,stride,1],padding='SAME')
			# biases = tf.get_variable('biases',depthwise.shape[3],initializer=tf.zeros_initializer)
			#inputs = tf.nn.bias_add(depthwise, biases)
			inputs = tf.layers.batch_normalization(inputs,training=self.is_training,name='bn_dw_'+str(self.conv_num))
			if activation == 'swish':
				inputs = self.swish(inputs)
			else:
				inputs = tf.math.sigmoid(inputs)


		##SQUEEZE AND EXCITATION PHASE
		se_ratio = 0.25

		num_reduced_filters = max(1, int(in_channel * se_ratio))
		# se_tensor = tf.layers.average_pooling2d(inputs,pool_size=1,strides=1)
		se_tensor_pre_shape = inputs.shape
		se_tensor = tf.keras.layers.GlobalAveragePooling2D(name="se_tensor_squeeze_" + str(self.conv_num))(inputs)
		target_shape = (1, 1, se_tensor.shape[1])
		se_tensor = tf.keras.layers.Reshape(target_shape, name="se_tensor_reshape_" + str(self.conv_num))(se_tensor)


		## Might need to reshape
		se_tensor = tf.layers.conv2d(se_tensor, filters=num_reduced_filters, kernel_size=1, strides=1, padding='same', kernel_initializer=self.initializer, kernel_regularizer=self.regularizer,name='conv_se_reduce_'+str(self.conv_num))
		if activation == 'swish':
			se_tensor = self.swish(se_tensor)
		else:
			se_tensor = tf.math.sigmoid(se_tensor)
		se_tensor = tf.layers.conv2d(se_tensor, filters=filters, kernel_size=1, strides=1, padding='same', kernel_initializer=self.initializer, kernel_regularizer=self.regularizer,name='conv_se_expand_'+str(self.conv_num))

		se_tensor = tf.math.sigmoid(se_tensor)

		inputs = tf.math.multiply(se_tensor, inputs, name='se_excite_' + str((self.conv_num))) ## might not be able to multiply layers

		##Output shape
		inputs = tf.layers.conv2d(inputs, filters=out_channel, kernel_size=1, strides=1, padding='same', kernel_initializer=self.initializer, kernel_regularizer=self.regularizer,name='project_conv_'+str(self.conv_num))
		inputs = tf.layers.batch_normalization(inputs,training=self.is_training,name='bn_project_'+str(self.conv_num))
		self.conv_num+=1

		return inputs





	def _variable_with_weight_decay(self, name, shape,wd):
		var = tf.get_variable(name,shape,initializer=self.initializer,dtype=tf.float32)
		if wd is not None:
			weight_decay = tf.multiply(tf.nn.l2_loss(var),wd,name='weight_loss')
			tf.add_to_collection('losses',weight_decay)
		return var 


	def forward(self,inputs):
		out = self.conv2d(inputs,out_channel=32,kernel_size=3,stride=2, advance=True)
		out = self.mb_conv_block(out, activation='swish', expand_ratio=1, in_channel=32, out_channel=16, kernel_size=[3,3,32,1], stride=1)
		out = self.mb_conv_block(out, activation='swish', expand_ratio=6, in_channel=16, out_channel=24, kernel_size=[3,3,16,1], stride=2)
		out = self.mb_conv_block(out, activation='swish', expand_ratio=6, in_channel=24, out_channel=40, kernel_size=[3,3,24,1], stride=2)
		out = self.mb_conv_block(out, activation='swish', expand_ratio=6, in_channel=40, out_channel=80, kernel_size=[3,3,40,1], stride=2)
		out = self.mb_conv_block(out, activation='swish', expand_ratio=6, in_channel=80, out_channel=112, kernel_size=[3,3,80,1], stride=1)
		out = self.mb_conv_block(out, activation='swish', expand_ratio=6, in_channel=112, out_channel=192, kernel_size=[3,3,112,1], stride=2)
		out = self.mb_conv_block(out, activation='swish', expand_ratio=6, in_channel=192, out_channel=320, kernel_size=[3,3,192,1], stride=1)
		out = self.conv2d(out,out_channel=1280,kernel_size=1,stride=1)
		out = tf.layers.average_pooling2d(out,pool_size=1,strides=1)
		out = tf.layers.flatten(out)
		out = tf.layers.dropout(out,rate=self.keep_prob)
		predicts = tf.layers.dense(out,units=self.num_classes,kernel_initializer=self.initializer,kernel_regularizer=self.regularizer,name='fc')
		softmax_out = tf.nn.softmax(predicts,name='output')
		return predicts,softmax_out

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
	with tf.device('/GPU:0'):
		net = efficientnetb0()
		data = np.random.randn(64,32,32,3)
		inputs = tf.placeholder(tf.float32,[64,32,32,3])
		predicts,softmax_out = net.forward(inputs)
		config = tf.ConfigProto(allow_soft_placement=True)
		config.gpu_options.allow_growth=True
		init = tf.global_variables_initializer()
		sess = tf.Session(config=config)
		sess.run(init)
		output = sess.run(predicts,feed_dict={inputs:data})
		print(output.shape)
		sess.close()

