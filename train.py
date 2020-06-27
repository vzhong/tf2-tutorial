import os
import ujson as json
import revtok
import argparse
import tensorflow as tf
from rich import print
from data import SNLI
from tqdm.auto import tqdm
from collections import defaultdict


class Model(tf.keras.Model):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dropout = tf.keras.layers.Dropout(args.dropout)
        self.emb = tf.keras.layers.Embedding(args.nwords, args.demb)
        self.rnn = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(args.drnn, return_sequences=True))
        self.attn_scorer = tf.keras.layers.Dense(1)
        self.proj = tf.keras.layers.Dense(args.nlabels)

    def get_optimizer(self):
        return tf.keras.optimizers.Adam()

    def get_objective(self):
        return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def call(self, batch, training=True):
        emb1 = self.emb(batch['ids1'])
        rnn1 = self.rnn(self.dropout(emb1, training=training))
        score1 = self.attn_scorer(self.dropout(rnn1, training=training))
        mask1 = tf.expand_dims(batch['mask1'], axis=2)
        score1 = score1 - 1e20*(1-mask1)
        score1 = tf.nn.softmax(score1, axis=1)
        ctx1 = tf.reduce_sum(tf.multiply(rnn1, tf.broadcast_to(score1, shape=rnn1.shape)), axis=1)

        emb2 = self.emb(batch['ids2'])
        rnn2 = self.rnn(self.dropout(emb2, training=training))
        score2 = self.attn_scorer(self.dropout(rnn2, training=training))
        mask2 = tf.expand_dims(batch['mask2'], axis=2)
        score2 = score2 - 1e20*(1-mask2)
        score2 = tf.nn.softmax(score2, axis=1)
        ctx2 = tf.reduce_sum(tf.multiply(rnn2, tf.broadcast_to(score2, shape=rnn2.shape)), axis=1)
        diff = ctx1 - ctx2
        return self.proj(diff)

    def compute_loss(self, out, batch, objective):
        return objective(batch['label'], out)

    def extract_preds(self, out, batch):
        return tf.argmax(out, axis=1, output_type=tf.int32)

    def compute_metrics(self, preds, batch):
        eq = tf.equal(preds, batch['label'])
        return dict(acc=eq.numpy().mean().item())

    def early_stop(self, metrics, best):
        better = metrics['dev_acc'] > best.get('dev_acc', -1)
        if better:
            print('{} > {}'.format(metrics['dev_acc'], best.get('dev_acc', -1)))
        return better

    def output_dir(self):
        return self.args.dexp

    def get_checkpoint_manager(self, optimizer):
        ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, model=self)
        manager = tf.train.CheckpointManager(ckpt, self.output_dir(), max_to_keep=3)
        return ckpt, manager

    def run_train(self, train, dev, num_train_steps):
        train = train.shuffle(1000).padded_batch(self.args.batch)
        optimizer = self.get_optimizer()
        objective = self.get_objective()

        ckpt, manager = self.get_checkpoint_manager(optimizer=optimizer)
        ckpt.restore(manager.latest_checkpoint)
        if manager.latest_checkpoint:
            print("Restored from {}".format(manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")

        bar = tqdm(total=num_train_steps, desc='training steps')
        best = dict()

        for epoch in range(self.args.epoch):
            metrics = defaultdict(list)
            for batch_id, batch in enumerate(train):
                ckpt.step.assign_add(1)

                with tf.GradientTape() as tape:
                    out = self(batch, training=True)
                    loss = self.compute_loss(out, batch, objective)
                    grads = tape.gradient(loss, self.trainable_variables)
                    optimizer.apply_gradients(zip(grads, self.trainable_variables))

                    metrics['loss'].append(loss.numpy().mean())
                    preds = self.extract_preds(out, batch)
                    for k, v in self.compute_metrics(preds, batch).items():
                        metrics[k].append(v)

                bar.update(1)
                if int(ckpt.step) % self.args.eval_period == 0:
                    metrics = {'train_{}'.format(k): sum(v)/len(v) for k, v in metrics.items()}
                    metrics.update({'dev_{}'.format(k): v for k, v in self.run_evaluate(dev).items()})
                    print(metrics)
                    if self.early_stop(metrics, best):
                        best.update(metrics)
                        print('Found new best!')
                        save_path = manager.save()
                        print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
                    metrics = defaultdict(list)

    def run_evaluate(self, data: tf.data.Dataset):
        data = data.padded_batch(self.args.batch)
        metrics = defaultdict(list)
        for batch_id, batch in enumerate(data):
            out = self(batch, training=True)
            preds = self.extract_preds(out, batch)
            for k, v in self.compute_metrics(preds, batch).items():
                metrics[k].append(v)
        return {k: sum(v)/len(v) for k, v in metrics.items()}

    @classmethod
    def preprocess_data(cls, raw):
        names = sorted(list(raw.keys()))
        vocab = set()
        for name in names:
            split = raw[name]
            for ex in tqdm(split, desc='tokenizing {}'.format(name)):
                ex['words1'] = w = revtok.tokenize(ex['sent1'])
                vocab.update(w)
                ex['words2'] = w = revtok.tokenize(ex['sent2'])
                vocab.update(w)
        word2index = ['PAD'] + sorted(list(vocab))
        index2word = {w: i for i, w in enumerate(word2index)}

        for name in names:
            split = raw[name]
            for ex in tqdm(split, desc='numericalizing {}'.format(name)):
                ex['ids1'] = [index2word[w] for w in ex['words1']]
                ex['mask1'] = [1] * len(ex['ids1'])
                ex['ids2'] = [index2word[w] for w in ex['words2']]
                ex['mask2'] = [1] * len(ex['ids2'])
                del ex['sent1']
                del ex['sent2']
                del ex['words1']
                del ex['words2']

        return dict(splits=data, word2index=word2index, index2word=index2word)

    @classmethod
    def make_tf_data(cls, data):
        types = dict(ids1=tf.int32, mask1=tf.float32, ids2=tf.int32, mask2=tf.float32, label=tf.int32)
        shapes = dict(ids1=[None], ids2=[None], mask1=[None], mask2=[None], label=[])
        return tf.data.Dataset.from_generator(lambda: (ex for ex in data), types, shapes)


if __name__ == '__main__':
    print("TensorFlow version: {}".format(tf.__version__))
    print("Eager execution: {}".format(tf.executing_eagerly()))

    fdata = os.path.join('data', 'proc.json')
    if not os.path.isfile(fdata):
        data = SNLI.load()
        proc = Model.preprocess_data(data)
        with open(fdata, 'wt') as f:
            json.dump(proc, f)
    else:
        with open(fdata) as f:
            proc = json.load(f)

    for k, v in proc.items():
        print(k, len(v))

    args = argparse.Namespace(
        nwords=len(proc['word2index']),
        seed=42,
        batch=32,
        demb=100,
        drnn=200,
        dropout=0.3,
        dexp=os.path.join(os.getcwd(), 'exp'),
        nlabels=len(SNLI.labels),
        epoch=10,
        eval_period=1000,
    )
    tf.random.set_seed(args.seed)
    model = Model(args)
    train = Model.make_tf_data(proc['splits']['train'])
    dev = Model.make_tf_data(proc['splits']['dev'])
    num_train_steps = args.epoch * len(proc['splits']['train']) // args.batch
    model.run_train(train, dev, num_train_steps)
