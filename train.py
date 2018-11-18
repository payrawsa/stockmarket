def train(epochs=1000, *args, **kwargs ):
	"""
	train_data_filepath, tr_summaries_dir, va_summaries_dir, learing_rate, image_dir, label_dir, train_batch_size, label_size, checkpoint_dir, tr_checkpoint_prefix, va_checkpoint_prefix break_patience, lr
	"""
	with tf.device("/device:GPU:0"):
		with tf.name_scope('training_loop'):
			# build graph and insert model, iterator
			train_graph = get_graph( graph_type='train', summaries_dir=kwargs['tr_summaries_dir'], *args, **kwargs )
			valid_graph = get_graph( graph_type='valid', summaries_dir=kwargs['va_summaries_dir'], *args, **kwargs )
			with train_graph['graph'].as_default():
				with tf.variable_scope('train_graph'):
                    #TODO: change get_batch ######
					tr_iter = get_batch( file_path=kwargs['train_data_filepath'], batch_size=kwargs['train_batch_size'] )
					tr_x, tr_y = tr_iter.get_next()
					train_model = get_model(X=tr_x, Y=tr_y, mode='train', output_units=kwargs['label_size'])
					train_op = train_graph['optimizer'].minimize(train_model['loss'], global_step=train_graph['global_step'])
					# merge all summaries to be saved during training
					tr_merge_all_summaries = train_graph['summaries']
					tr_all_summaries = tr_merge_all_summaries()
					train_graph['session'].run(tf.global_variables_initializer())
					train_graph['saver'] = train_graph['saver'](max_to_keep = train_graph['max_to_keep'])
					# save initial graph
					train_graph['saver'].save(save_path=kwargs['tr_checkpoint_prefix'], sess=train_graph['session'], global_step=train_graph['global_step'])

			with valid_graph['graph'].as_default():
				with tf.variable_scope('valid_graph'):
					# naive early stopping parameters
					prev_loss = best_loss = 100000000000.0
					cur_patience = 0
                    #TODO: change get_batch #######
                    va_iter = get_batch( file_path=kwargs['valid_data_filepath'], batch_size=kwargs['train_batch_size'] )
					va_x, va_y = va_iter.get_next()
					valid_model = get_model(X=va_x, Y=va_y, mode='valid', output_units=kwargs['label_size'])
					va_merge_all_summaries = valid_graph['summaries']
					va_all_summaries = va_merge_all_summaries()
					valid_graph['saver'] = valid_graph['saver'](max_to_keep = valid_graph['max_to_keep'])

			for epoc in range(epochs):
				# one complete pass of training data
				print("epoc: {}".format(epoc) )

				with train_graph['graph'].as_default():
					with tf.variable_scope('train_graph'):
						# reinitialize train_batch iterator
						train_graph['session'].run(tr_iter.initializer)
						train_graph['session'].run(train_model['acc_initializer'])
						while True:
							try:
								#train model
								loss, _, accuracy, _ , tr_summaries = train_graph['session'].run([train_model['loss'], train_model['acc_update_op'], train_model['accuracy'], train_op, tr_all_summaries])
								train_graph['summary_writer'].add_summary(tr_summaries, train_graph['global_step'].eval())
								train_graph['saver'].save(save_path=kwargs['tr_checkpoint_prefix'], sess=train_graph['session'], global_step=train_graph['global_step'].eval())
							except tf.errors.OutOfRangeError:
								break

				with valid_graph['graph'].as_default():
					with tf.variable_scope('valid_graph'):
						valid_graph['session'].run(va_iter.initializer)
						valid_graph['session'].run(valid_graph['acc_initializer'])
						valid_graph['saver'].restore(sess=valid_graph['session'], save_path= tf.train.latest_checkpoint(kwargs['checkpoint_dir']) )
						while True:
							try:
								#validate model
								val_loss, _, val_accuracy, val_summaries = valid_graph['session'].run([valid_model['loss'], valid_model['acc_update_op'], valid_model['accuracy'], va_all_summaries])
							except tf.errors.OutOfRangeError:
								valid_graph['summary_writer'].add_summary(val_summaries, epoc)
								print("global_step: {:6}, loss: {:13.6f}, accuracy ={:.6f}, val_loss: {:13.6f}, val_accuracy: {:.6f}".format(valid_graph['global_step'].eval(), loss, accuracy, val_loss, val_accuracy))
								break
						# save checkpoint of loss be better than previous one
						if float("{:.2f}".format(val_loss)) < float("{:.2f}".format(best_loss)):
							best_loss = val_loss
							cur_patience = 0
							valid_graph['saver'].save(save_path=kwargs['va_checkpoint_prefix'], sess=valid_graph['session'], global_step=valid_graph['global_step'].eval())
						else:
							cur_patience += 1

					if cur_patience == break_patience:
						print('\n############ Early stopping ############')
						train_graph['session'].close()
						valid_graph['session'].close()
						break
