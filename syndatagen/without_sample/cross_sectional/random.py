import pandas as pd
import numpy as np

import random
import string

class RandomGenerator_WithoutSampleCrossSectional:
    def __init__(self, config):
        self.num_cols = config['num_rows']
        self.fields = config['fields']

    def random_string_generate(self):
        return ''.join(random.choices(string.ascii_lowercase, k=10))
    
    def random_int_generate(self):
        return random.randint(0, 100)
    
    def random_float_generate(self):
        return random.uniform(0, 100)
    
    def random_date_generate(self):
        return pd.Timestamp(np.datetime64('now') - np.random.randint(0, 365, size=self.num_cols))
    
    def random_bool_generate(self):
        return random.choice([True, False])
    
    def get_categories(self, categories, count, prompt=None):
        if count > len(categories):
            # TODO: LLM based generation
            raise ValueError('count cannot be greater than number of categories')
        else:
            return random.choices(categories, k=count)


    def random_categorical_generate(self, categories):
        return random.choice(categories)

    def generate(self):
        data = {}
        for field in self.fields:
            print(field)
            if field['type'] == 'random_string':
                data[field['name']] = [self.random_string_generate() for _ in range(self.num_cols)]
            elif field['type'] == 'categorical_string':
                categories = self.get_categories(field['type_params']['categories'], field['type_params']['count'])
                data[field['name']] = [self.random_categorical_generate(categories) for _ in range(self.num_cols)]
        return pd.DataFrame(data)