from transformer.layers.generate_mask import generate_mask
import tensorflow as tf

class Trainer:
    def __init__(self, model, optimizer, epochs, checkpoint_folder):
        # Khởi tạo các thuộc tính của lớp Trainer
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        # Tạo các metric để theo dõi loss và accuracy trong quá trình huấn luyện
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
        # Tạo checkpoint để lưu trạng thái của model và optimizer
        self.checkpoint = tf.train.Checkpoint(model = self.model, optimizer = self.optimizer)
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, checkpoint_folder, max_to_keep=3)

    def cal_acc(self, real, pred):
        # Tính toán độ chính xác
        accuracies = tf.equal(real, tf.argmax(pred, axis=2))

        # Tạo mask để loại bỏ padding tokens
        mask = tf.math.logical_not(real == 0)
        accuracies = tf.math.logical_and(mask, accuracies)

        # Chuyển đổi kiểu dữ liệu và tính trung bình
        accuracies = tf.cast(accuracies, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)

    def loss_function(self, real, pred):
        # Hàm tính loss
        cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        # Tạo mask để loại bỏ padding tokens
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss = cross_entropy(real, pred)

        # Áp dụng mask và tính trung bình loss
        mask = tf.cast(mask, dtype=loss.dtype)
        loss = loss * mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)

    def train_step(self, inp, tar):
        # Một bước huấn luyện
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]
        # Tạo các mask cần thiết
        encoder_padding_mask, decoder_look_ahead_mask ,decoder_padding_mask = generate_mask(inp, tar_inp)

        with tf.GradientTape() as tape:
            # Forward pass
            preds = self.model(inp, tar_inp, True, encoder_padding_mask, decoder_look_ahead_mask, decoder_padding_mask)
            # Tính loss
            d_loss = self.loss_function(tar_real, preds)

        # Tính gradient
        grads = tape.gradient(d_loss, self.model.trainable_variables)

        # Cập nhật trọng số
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        # Cập nhật metrics
        self.train_loss.update_state(d_loss)
        self.train_accuracy.update_state(self.cal_acc(tar_real, preds))

    def fit(self, data):
        print('=============Training Progress================')
        print('----------------Begin--------------------')
        # Khôi phục checkpoint nếu có
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print('Restored checkpoint manager !')

        for epoch in range(self.epochs):
            # Reset metrics cho mỗi epoch
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            
            for (batch, (inp, tar)) in enumerate(data):
                # Thực hiện một bước huấn luyện
                self.train_step(inp, tar)

                # In kết quả sau mỗi 50 batch
                if batch % 50 == 0:
                    print(f'Epoch {epoch + 1} Batch {batch} Loss {self.train_loss.result():.3f} Accuracy {self.train_accuracy.result():.3f}')

                # Lưu checkpoint sau mỗi 5 epoch
                if (epoch + 1) % 5 == 0:
                    saved_path = self.checkpoint_manager.save()
                    print('Checkpoint was saved at {}'.format(saved_path))
        print('----------------Done--------------------')

    def predict(self, encoder_input, decoder_input, is_train, max_length, end_token):
        print('=============Inference Progress================')
        print('----------------Begin--------------------')
        # Khôi phục checkpoint nếu có
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print('Restored checkpoint manager !')
        
        for i in range(max_length):
            # Tạo các mask cần thiết
            encoder_padding_mask, decoder_look_ahead_mask ,decoder_padding_mask = generate_mask(encoder_input, decoder_input)

            # Dự đoán
            preds = self.model(encoder_input, decoder_input, is_train, encoder_padding_mask, decoder_look_ahead_mask, decoder_padding_mask)

            # Lấy token cuối cùng được dự đoán
            preds = preds[:, -1:, :]  # (batch_size, 1, vocab_size)

            predicted_id = tf.argmax(preds, axis=-1)

            # Thêm token dự đoán vào đầu vào của decoder
            decoder_input = tf.concat([decoder_input, predicted_id], axis=-1)

            # Kết thúc nếu token dự đoán là end token
            if predicted_id == end_token:
                break

        return decoder_input