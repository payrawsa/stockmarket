import tensorflow as tf

def get_model(X, Y, mode='train', **kwargs):
	"""
	returns tensorflow model network

	parameters:
		X : tensor, input to the network
		Y : tensor, labels for gradient calculation
		mode : string, one of ['tran', 'valid', 'test']
		kwargs : keyword args, eg loss_fn
		output_units:

	returns:
		dict : {loss, accuracy, accuracy update op, update op initializer} if mode in [train, test] else
				{logits and predictions}
	"""

	# TODO: make new model
	# ENCODER/DECODER code here
	layer = #

	logit = layer

    #TODO:
    #predictions from logits
	Y_ = tf.nn.sigmoid(logit,)
	Y_ = tf.cast(x = tf.greater(tf.cast(Y_, tf.float32), sig_cond), dtype=tf.int8,	name = 'predictions')


	label = tf.stop_gradient(Y)

	if mode.lower() in ['train', 'valid']:
		if kwargs['loss_fn'] == 'sigmoid_cross_entropy':
        	loss_fn = tf.losses.sigmoid_cross_entropy
		elif kwargs['loss_fn'] == 'softmax_cross_entropy':
        	loss_fn = tf.losses.softmax_cross_entropy
		else:
			loss_fn = tf.losses.mean_squared_error
		loss = tf.reduce_mean( loss_fn(labels=(label, logits=logit))
		accuracy, update_op = tf.metrics.accuracy( labels=label, predictions=predictions, name='accuracy')
		accuracy_op_init = tf.variables_initializer([accuracy, update_op], name='metrics_initializer')
		# Model Summaries
		tf.summary.scalar('Loss', loss)
		tf.summary.scalar('Accuracy', accuracy)
		return {'loss':loss,'accuracy':accuracy,'acc_update_op':update_op, 'acc_initializer':accuracy_op_init }
	else:
		return {'logits':logit, 'predictions':Y_}
