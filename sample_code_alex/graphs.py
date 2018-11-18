import tensorflow as tf

def get_graph(graph_type=None, summaries_dir=None, *args, **kwargs):

	# argument validation
	assert graph_type.lower() in ['train', 'valid', 'test'], 'Invalid graph type'
	assert summaries_dir !=None, 'summary directory not defined'
	if graph_type=='train':
		assert 'lr' in kwargs.keys(), 'training graph must have learing rate'

	graph = tf.Graph()
	with graph.as_default():
		with tf.variable_scope('{}_graph'.format(graph_type)):

			if type == 'train':
				#optimiser
				global_step = tf.Variable(0, name='global_step')
				#decaying learning rate
				if kwargs['decay_lr']:
					lr = tf.train.exponential_decay(kwargs['lr'], global_step=global_step, decay_rate=kwargs['decay_rate'], decay_steps=kwargs['decay_steps'], name='lr_decay' )
					tf.summary.scalar('Learning Rate', lr)
				else:
					lr = kwargs['lr']
				optimizer = tf.train.AdamOptimizer( learning_rate=lr, )
			else:
				global_step = None
				optimizer = None

			summary_writer = tf.summary.FileWriter( summaries_dir, graph )
			config_proto = tf.ConfigProto(allow_soft_placement=True)
			session = tf.Session( config= config_proto )
			if graph_type.lower()=='valid':
				max_to_keep=1
			else:
				max_to_keep=5
			return {'graph':graph,
							'session':session,
							'global_step': global_step,
							'optimizer':optimizer,
							'summaries':tf.summary.merge_all,
							'summary_writer':summary_writer,
							'saver':tf.train.Saver,
							'max_to_keep':max_to_keep
							 }
