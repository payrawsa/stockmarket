def make_predictions(**kwargs):
	# TODO: write testing loop
	# testing step
	with tf.device("/device:GPU:0"):
		with tf.name_scope('test_loop'):
			test_graph = get_graph( graph_type='test', summaries_dir=kwargs['te_summaries_dir'], *args, **kwargs )
			with test_graph['graph'].as_default():
				with tf.variable_scope('test_graph'):
                    #TODO: change get_batch
					te_iter = get_batch( file_path=kwargs['train_data_filepath'], batch_size=kwargs['train_batch_size'] )
					te_x, te_y = te_iter.get_next()
					test_model = get_model(X=te_x, Y=te_y, mode='test', output_units=kwargs['label_size'])
					all_predictions = []
					# merge all summaries to be saved during training
					te_merge_all_summaries = test_graph['summaries']
					te_all_summaries = te_merge_all_summaries()
					test_graph['saver'] = test_graph['saver']()
					test_graph['saver'].restore(sess=test_graph['session'], save_path= tf.train.latest_checkpoint( checkpoint_dir=checkpoint_dir))

					test_graph['session'].run(te_iter.initializer)
					while True:
						try:
							#validate model
							predictions =	test_graph['session'].run([test_model['prediction']])
							all_predictions.append( zip(te_img_ids,predictions) )
						except tf.errors.OutOfRangeError:
							with open(os.path.join(checkpoint_dir, 'predictions.csv') as fp:
								fp.write(all_predictions)
							break
