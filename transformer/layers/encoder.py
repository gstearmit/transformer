import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dropout
from transformer.layers.encoder_layer import EncoderLayer
from transformer.layers.generate_position import generate_positional_encoding


class Encoder(tf.keras.layers.Layer):
	def __init__(self, n, h, vocab_size, d_model, d_ff, activation, dropout_rate=0.1, eps=0.1):
		# Khởi tạo lớp Encoder với các tham số cần thiết
		super(Encoder, self).__init__()
		self.n = n  # Số lượng lớp encoder
		self.d_model = d_model  # Kích thước của vector đầu ra
		# Tạo danh sách các lớp EncoderLayer
		self.encoder_layers = [EncoderLayer(h, d_model, d_ff, activation, dropout_rate, eps) for _ in range(n)]
		# Lớp nhúng từ vựng
		self.word_embedding = Embedding(vocab_size, output_dim=d_model)
		# Lớp dropout để giảm overfitting
		self.dropout = Dropout(dropout_rate)

	def call(self, q, is_train, mask=None):
		"""
            Parameters
            ----------
            x: tensor
				input sentence
                shape: (..., q_length)
			is_train: bool
				is training or not
			mask: tensor
				masking
			Returns
            ----------
            encoder_out: tensor
                the new representation of sentence by attention between its words
				shape: (..., q_length, d_model)
        """
		q_length = q.shape[1]  # Lấy độ dài của câu đầu vào
		
		# Nhúng câu đầu vào thành vector
		# embedded_q shape: (..., q_length, d_model)
		# TODO: Normalize embedded_q
		embedded_q = self.word_embedding(q)

		# Điều chỉnh độ lớn của vector nhúng
		embedded_q *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

		# Tạo mã hóa vị trí cho câu
		# positional_encoding shape: (1, q_length, d_model)
		positional_encoding = generate_positional_encoding(q_length, self.d_model)

		# Thêm mã hóa vị trí vào vector nhúng và áp dụng dropout
		encoder_out = self.dropout(embedded_q + positional_encoding, training=is_train)

		# Duyệt qua từng lớp encoder và xử lý đầu ra
		for encoder_layer in self.encoder_layers:
			encoder_out = encoder_layer(encoder_out, is_train, mask)
		
		# Trả về đầu ra của encoder
		return encoder_out
