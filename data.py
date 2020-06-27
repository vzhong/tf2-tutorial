import os
import ujson as json
import zipfile
import requests


class SNLI:

    url = 'https://nlp.stanford.edu/projects/snli/snli_1.0.zip'
    labels = ['entailment', 'neutral', 'contradiction']

    @classmethod
    def read_split(cls, lines):
        data = []
        for l in lines:
            r = json.loads(l)
            if r['gold_label'] == '-':
                continue
            data.append(dict(label=cls.labels.index(r['gold_label']), sent1=r['sentence1'], sent2=r['sentence2']))
        return data

    @classmethod
    def load(cls):
        root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
        if not os.path.isdir(root):
            os.makedirs(root)

        fjson = os.path.join(root, 'snli.json')

        if not os.path.isfile(fjson):
            fzip = os.path.join(root, 'snli.zip')
            if not os.path.isfile(fzip):
                # download
                with requests.get(cls.url, stream=True) as r:
                    r.raise_for_status()
                    with open(fzip, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)

            with zipfile.ZipFile(fzip) as fz:
                train = cls.read_split(fz.read('snli_1.0/snli_1.0_train.jsonl').decode().splitlines())
                dev = cls.read_split(fz.read('snli_1.0/snli_1.0_dev.jsonl').decode().splitlines())
                test = cls.read_split(fz.read('snli_1.0/snli_1.0_test.jsonl').decode().splitlines())

            with open(fjson, 'wt') as f:
                json.dump(dict(train=train, dev=dev, test=test), f)

        with open(fjson) as f:
            return json.load(f)


if __name__ == '__main__':
    SNLI.load()
