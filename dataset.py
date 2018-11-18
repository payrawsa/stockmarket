import tensorflow as tf
def get_batch(file_path, batch_size=32, buffer_size=50000):
	'''
	Get iterator of dataset

	Parameters:
	file_path - directory path containing train, valid and test dataset files
	batch_size - batch size of dataset

	Returns:
	tf.data.Iterator : iterator of dataset
	'''
    def decode_line(line):
        #TODO:
        pass

    dataset = tf.data.TextLineDataset(filenames=file_path, )
	dataset = dataset.shuffle(buffer_size=buffer_size).map(lambda x: decode_line(x))
	dataset = dataset.batch(batch_size).prefetch(batch_size*2)
	return dataset.make_initializable_iterator()
