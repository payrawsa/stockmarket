def test(**kwargs ):
	"""
	train_data_filepath, tr_summaries_dir, va_summaries_dir, learing_rate, image_dir, label_dir, train_batch_size, label_size, checkpoint_dir, tr_checkpoint_prefix, va_checkpoint_prefix break_patience, lr
	"""
	with tf.device("/device:GPU:0"):
        with tf.name_scope('testing_loop'):
            test_graph = get_graph( graph_type='test', summaries_dir=kwargs['te_summaries_dir'], *args, **kwargs )
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
					valid_graph['saver'] = valid_graph['saver']()

					valid_graph['session'].run(va_iter.initializer)
					valid_graph['session'].run(valid_graph['acc_initializer'])
					valid_graph['saver'].restore(sess=valid_graph['session'], save_path= tf.train.latest_checkpoint(kwargs['checkpoint_dir']) )
					while True:
						try:
							#validate model
							val_loss, _, val_accuracy, val_summaries = valid_graph['session'].run([valid_model['loss'], valid_model['acc_update_op'], valid_model['accuracy'], va_all_summaries])
						except tf.errors.OutOfRangeError:
							valid_graph['summary_writer'].add_summary(val_summaries, epoc)
							print("loss: {:13.6f}, accuracy ={:.6f}, val_loss: {:13.6f}, val_accuracy: {:.6f}".format(loss, accuracy, val_loss, val_accuracy))
							break
