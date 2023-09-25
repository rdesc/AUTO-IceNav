import json
import os


class Storage:
    def __init__(self, output_dir, file):
        self.fp = None
        if output_dir:
            self.fp = os.path.join(output_dir, file)
            with open(self.fp, 'w') as f:
                f.write('')
        self.data = None
        self.history = []

    def put_scalar(self, name, value):
        if self.data is None:
            self.data = {}
        self.data[name] = value

    def put_scalars(self, **kwargs):
        for k, v in kwargs.items():
            self.put_scalar(k, v)

    def step(self):
        if self.data is not None:
            if self.fp:
                with open(self.fp, 'a') as f:
                    f.write(json.dumps(self.data))
                    f.write('\n')
            else:
                self.history.append(self.data)
            self.data = None

    def get_history(self):
        if not self.fp:
            return self.history
        with open(os.path.join(self.fp), 'r') as f:
            return [json.loads(item) for item in f.readlines()]
