import tensorflow as tf

from models import ggnn, great_transformer, rnn, util
import spektral


class VarMisuseModel(tf.keras.layers.Layer):
    def __init__(self, config, vocab_dim):
        super(VarMisuseModel, self).__init__()
        self.config = config
        self.vocab_dim = vocab_dim

    def build(self, _):
        # These layers are always used; initialize with any given model's hidden_dim
        random_init = tf.random_normal_initializer(
            stddev= self.config["base"]["hidden_dim"] ** -0.5
        )
        self.embed = tf.Variable(
            random_init([self.vocab_dim, self.config["base"]["hidden_dim"]]),
            dtype=tf.float32,
        )
        self.pos_embed = tf.Variable(
            random_init([self.config["base"]["max_sequence_length"], self.config["base"]["hidden_dim"]]),
            dtype=tf.float32,
        )
        
        self.prediction = tf.keras.layers.Dense(2)
        
        # tf.keras.Sequential() # Pointers for the two labels
        # self.prediction.add(tf.keras.layers.Dense(1024))
        # self.prediction.add(tf.keras.layers.ReLU())
        # self.prediction.add(tf.keras.layers.Dense(512))
        # self.prediction.add(tf.keras.layers.ReLU())
        # self.prediction.add(tf.keras.layers.Dense(2))
        
        self.pooling = spektral.layers.GlobalAttentionPool(self.config["base"]["hidden_dim"])
        
        # Store for convenience
        self.pos_enc = tf.constant(
            util.positional_encoding(self.config["base"]["hidden_dim"], 5000)
        )

        # Next, parse the main 'model' from the config
        join_dicts = lambda d1, d2: {
            **d1,
            **d2,
        }  # Small util function to combine configs
        base_config = self.config["base"]
        desc = self.config["configuration"].split(" ")
        self.stack = []
        for kind in desc:
            if kind == "rnn":
                self.stack.append(
                    rnn.RNN(
                        join_dicts(self.config["rnn"], base_config),
                        shared_embedding=self.embed,
                    )
                )
            elif kind == "ggnn":
                self.stack.append(
                    ggnn.GGNN(
                        join_dicts(self.config["ggnn"], base_config),
                        shared_embedding=self.embed,
                    )
                )
            elif kind == "great":
                self.stack.append(
                    great_transformer.Transformer(
                        join_dicts(self.config["transformer"], base_config),
                        shared_embedding=self.embed,
                    )
                )
            elif (
                kind == "transformer"
            ):  # Same as above, but explicitly without bias_dim set -- defaults to regular Transformer.
                joint_config = join_dicts(self.config["transformer"], base_config)
                joint_config["num_edge_types"] = None
                self.stack.append(
                    great_transformer.Transformer(
                        joint_config, shared_embedding=self.embed
                    )
                )
            else:
                raise ValueError("Unknown model component provided:", kind)

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(None, None, None), dtype=tf.int32),
            tf.TensorSpec(shape=(None, None), dtype=tf.int32),
            tf.TensorSpec(shape=(None, 4), dtype=tf.int32),
            tf.TensorSpec(shape=(), dtype=tf.bool),
        ]
    )
    def call(self, tokens, token_mask, edges, training):
        # Embed subtokens and average into token-level embeddings, masking out invalid locations
        subtoken_embeddings = tf.nn.embedding_lookup(self.embed, tokens)
        subtoken_embeddings *= tf.expand_dims(
            tf.cast(tf.clip_by_value(tokens, 0, 1), dtype="float32"), -1
        )

        states = tf.reduce_mean(subtoken_embeddings, 2)
        # Track whether any position-aware model processes the states first. If not, add positional encoding to ensure that e.g. GREAT and GGNN
        # have sequential awareness. This is especially (but not solely) important because the default, non-buggy 'location' is the 0th token,
        # which is hard to predict for e.g. Transformers and GGNNs without either sequential awareness or a special marker at that location.
        if not self.stack or not isinstance(self.stack[0], rnn.RNN):
            # states += self.pos_enc[: tf.shape(states)[1]]
            states += self.pos_embed[: tf.shape(states)[1]]
            
        # Pass states through all the models (may be empty) in the parsed stack.
        for model in self.stack:
            if isinstance(model, rnn.RNN):  # RNNs simply use the states
                states = model(states, training=training)
            elif isinstance(model, ggnn.GGNN):  # For GGNNs, pass edges as-is
                states = model(states, edges, training=training)
            elif isinstance(
                model, great_transformer.Transformer
            ):  # For Transformers, reverse edge directions to match query-key direction and add attention mask.
                mask = tf.cast(token_mask, dtype="float32")
                mask = tf.expand_dims(tf.expand_dims(mask, 1), 1)
                attention_bias = tf.stack(
                    [edges[:, 0], edges[:, 1], edges[:, 3], edges[:, 2]], axis=1
                )
                states = model(
                    states, mask, attention_bias, training=training
                )  # Note that plain transformers will simply ignore the attention_bias.
            else:
                raise ValueError("Model not yet supported:", model)
        # max pool the sequences
        # states = tf.reduce_max(states, axis=1)
        states = self.pooling(states)

        # Finally, predict a simple 2-pointer outcome from the first super token, and return
        predictions = self.prediction(
            states
        )  # Convert to [batch, 1, seq-length] for convenience.
        return predictions

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(None), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
        ]
    )
    def get_loss(self, predictions, label):
        # The first token represent the bug location
        # loss_func = tf.keras.losses.Hinge()
        # loss_func = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        label_onehot = tf.one_hot(label, 2, dtype=tf.int64)
        
        label = tf.cast(label, dtype=tf.float32)
        loss = loss_func(label_onehot, predictions)
        # Calculate  the accuracy
        label = tf.cast(label, dtype=tf.int64)
        # predicted_labels = tf.squeeze(tf.cast(predictions > 0, dtype=tf.int64))
        predicted_labels = tf.argmax(predictions, axis=1)
        
        acc = tf.keras.metrics.binary_accuracy(label, predicted_labels)
        acc = tf.reduce_mean(acc)

        # Calculate the precision directly
        g_true = tf.math.count_nonzero(predicted_labels)
        precision = (
            (tf.math.count_nonzero(predicted_labels * label) / g_true)
            if g_true > 0
            else tf.cast(1.0, dtype=tf.float64)
        )

        # Calculate the recall directly
        p_true = tf.math.count_nonzero(label)
        recall = (
            (tf.math.count_nonzero(predicted_labels * label) / p_true)
            if p_true > 0
            else tf.cast(1.0, dtype=tf.float64)
        )

        # calculate f1
        sums = precision + recall
        f1 = (
            2 * (precision * recall) / sums
            if sums > 0
            else tf.cast(0.0, dtype=tf.float64)
        )

        return loss, (acc, precision, recall, f1)

    # Used to initialize the model's variables
    def run_dummy_input(self):
        self(
            tf.ones((3, 3, 10), dtype="int32"),
            tf.ones((3, 3), dtype="int32"),
            tf.ones((2, 4), dtype="int32"),
            True,
        )
