from transformers import BertConfig, BertTokenizer, BertPreTrainedModel, BertModel
from transformers.data import DataProcessor
from torch.utils.data import TensorDataset

import os
import copy
import json
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import random
import logging

logger = logging.getLogger(__name__)

class BertForWordClassification(BertPreTrainedModel):

    def __init__(self, config):
        super(BertForWordClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                start_position=None, end_position=None, labels=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        output = outputs[0]
        output = self.dropout(output)
        span_output = torch.randn(output.shape[0],output.shape[-1]).to(output.device)
        for i in range(output.shape[0]):
            span_output[i] = output[i][start_position[i]:end_position[i]].mean(dim=0)
        logits = self.classifier(span_output)

        outputs = (logits,) + outputs[2:]

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs

MODEL_CLASSES = {
    'bert': (BertConfig, BertForWordClassification, BertTokenizer),
}


class InputExample(object):
    def __init__(self, guid, tokens, text_a, text_b=None, pos=[0,0], label=None):
        self.guid = guid
        self.tokens = tokens
        self.text_a = text_a
        self.text_b = text_b
        self.start_position = pos[0]
        self.end_position = pos[1]
        self.label = label

class InputFeatures(object):

    def __init__(self, input_ids, attention_mask, token_type_ids, start_position, end_position, label):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.start_position = start_position
        self.end_position = end_position
        self.label = label

class WordClassificationProcessor(DataProcessor):

    def get_example_from_tensor_dict(self, tensor_dict):
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['sentence'].numpy().decode('utf-8'),
                            None, str(tensor_dict['label'].numpy()))

    def _read_json(self, file):
        with open(file, encoding='utf-8') as f:
            return json.load(f)

    def get_train_examples(self, args):
        return self._create_examples(
            self._read_json(os.path.join(args.data_dir, args.train_file)), "train", args.do_mask)

    def get_dev_examples(self, args):
        return self._create_examples(
            self._read_json(os.path.join(args.data_dir, args.predict_file)), "dev", args.do_mask)

    def get_labels(self):
        # return ["Literal_Modifier", "Metonymic", "Non_Literal_Modifier", "Literal", "Mixed"]
        return [0, 1]

    def _create_examples(self, lines, set_type, do_mask):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            pos = line['pos']
            if do_mask:
                line['sentence'][pos[0]:pos[1]] = ['x']
                pos[1] = pos[0] + 1
            text_a = " ".join(line['sentence'])
            label = line['label']
            examples.append(
                InputExample(guid=guid, tokens=line['sentence'], text_a=text_a, text_b=None, pos=pos, label=label))
        return examples

def load_and_cache_examples(args, tokenizer, evaluate=False):

    processor = WordClassificationProcessor()
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        args.predict_file if evaluate else args.train_file,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        'masked' if args.do_mask else 'unmasked',
        str(args.max_seq_length)))
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        examples = processor.get_dev_examples(args) if evaluate else processor.get_train_examples(args)
        features = convert_examples_to_features(examples,
                                                tokenizer,
                                                label_list=label_list,
                                                max_length=args.max_seq_length,
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=0,
        )
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
    all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids,
                            all_start_positions, all_end_positions, all_labels)
    return dataset

def convert_examples_to_features(examples, tokenizer,
                                      max_length=512,
                                      label_list=None,
                                      pad_token=0,
                                      pad_token_segment_id=0,
                                      mask_padding_with_zero=True):

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 2000 == 0:
            logger.info("Writing example %d" % (ex_index))

        inputs = tokenizer.encode_plus(
            example.text_a,
            example.text_b,
            add_special_tokens=True,
            max_length=max_length
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        tf_tokens = tokenizer.convert_ids_to_tokens(input_ids)

        orig_to_tok_index = []
        all_tokens = ['[CLS]']
        for (i, token) in enumerate(example.tokens):
            orig_to_tok_index.append(len(all_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                all_tokens.append(sub_token)
        orig_to_tok_index.append(len(all_tokens))

        start_position = orig_to_tok_index[example.start_position]
        end_position = orig_to_tok_index[example.end_position]
        ori_target = ''.join(example.tokens[example.start_position:example.end_position]).lower()
        new_target = ''.join(tf_tokens[start_position:end_position]).replace('##','').lower()
        if len(tf_tokens) > max_length:
            logger.warning("Example length too long, remove example.",)
            continue
        if len(ori_target) != len(new_target):
            print (tf_tokens)
            print (example.tokens)
            logger.warning("Mapping original word: %s, to: %s, remove this example.",ori_target, new_target)
            continue

        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask), max_length)
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids), max_length)

        label = label_map[example.label] if example.label else None

        if ex_index < 2:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            if label:
                logger.info("label: %s (id = %d)" % (example.label, label))

        features.append(
                InputFeatures(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              start_position=example.start_position,
                              end_position=example.end_position,
                              label=label))


    return features

def convert_single_example_to_input(example, tokenizer, do_mask=True):
    # input example
    processor = WordClassificationProcessor()
    label_list = processor.get_labels()

    pos = example['pos']
    if do_mask:
       example['sentence'][pos[0]:pos[1]] = ['x']
       pos[1] = pos[0] + 1
    text_a = " ".join(example['sentence'])
    examples = [InputExample(guid=0, tokens=example['sentence'], text_a=text_a, text_b=None, pos=pos, label=None)]

    # convert example to feature
    features = convert_examples_to_features(examples,
                                            tokenizer,
                                            label_list=label_list,
                                            max_length=256,
                                            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                            pad_token_segment_id=0,
    )

    # convert to tensor
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
    all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)

    return {'input_ids': all_input_ids,
            'attention_mask': all_attention_mask,
            'token_type_ids': all_token_type_ids,
            'start_position': all_start_positions,
            'end_position': all_end_positions
            }


def data_augmentation(file_path):

    with open (file_path) as f:
        data = json.load(f)
    name_set = []
    for instance in data:
        name_set.append(instance['sentence'][instance['pos'][0]:instance['pos'][1]])
    aug_data = []
    for instance in data:
        st = instance['pos'][0]
        ed = instance['pos'][1]
        for _ in range(10):
            i = random.randint(0,len(name_set))
            aug_instance = copy.deepcopy(instance)
            new_target = name_set[i%len(name_set)]
            aug_instance['sentence'][st:ed] = new_target
            aug_instance['pos'][1] = st + len(new_target)
            aug_data.append(aug_instance)
    random.shuffle(aug_data)

    path,name = os.path.split(file_path)

    with open(os.path.join(path,'augmented',name),'w') as f:
        json.dump(aug_data, f)

def to_prewin_format(file_path):

    path, name = os.path.split(file_path)
    dataset = name.split('_')[0]
    split = 'train' if 'train' in name else 'test'

    with open(file_path) as f:
        data = json.load(f)

    metonymic = []
    literal = []
    for piece in data:
        sentence = piece['sentence']
        pos = piece['pos']
        pmw = ' '.join(sentence[pos[0]:pos[1]])
        line = f'{pmw}<SEP>{" ".join(sentence[:pos[0]])}<ENT>{pmw}<ENT>{" ".join(sentence[pos[1]:])}\n'
        if piece['label'] == 1:
            metonymic.append(line)
        else:
            literal.append(line)

    with open(os.path.join(path,'prewin', f'{dataset}_metonymic_{split}.txt'),'w') as f:
        for line in metonymic:
            f.write(line)
    with open(os.path.join(path,'prewin', f'{dataset}_literal_{split}.txt'),'w') as f:
        for line in literal:
            f.write(line)

