import torch
from datasets import load_from_disk
from transformers import BertTokenizer
from transformers import BertModel
import random

class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        dataset = load_from_disk(r'D:\h20240513\seamew\ChnSentiCorp')

        def f(data):
            return len(data['text']) > 40

        self.dataset = dataset.filter(f)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        text = self.dataset[i]['text']
        sentence1 = text[:20]
        sentence2 = text[20:40]
        label = 0

        if random.randint(0, 1) == 0:
            j = random.randint(0, len(self.dataset) - 1)
            sentence2 = self.dataset[j]['text'][20:40]
            label = 1

        return sentence1, sentence2, label

token = BertTokenizer.from_pretrained('bert-base-chinese')

def collate_fn(data):

    sents = [i[:2] for i in data]
    labels = [i[2] for i in data]



    data = token.batch_encode_plus(batch_text_or_text_pairs=sents,
                                   truncation=True,
                                   padding='max_length',
                                   max_length=45,
                                   return_tensors='pt',
                                   return_length=True,
                                   add_special_tokens=True)

    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    token_type_ids = data['token_type_ids']
    # 把一句话中第15个词作为mask
    labels = torch.LongTensor(labels)

    return input_ids, attention_mask, token_type_ids, labels

def test(model):
    model.eval()
    model = model.cuda()
    correct = 0
    total = 0

    loader_test = torch.utils.data.DataLoader(dataset=Dataset(),
                                              batch_size=32,
                                              collate_fn=collate_fn,
                                              shuffle=True,
                                              drop_last=True)

    for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(loader_test):
        input_ids, attention_mask, token_type_ids, labels = input_ids.cuda(), attention_mask.cuda(), token_type_ids.cuda(), labels.cuda()
        if i == 5:
            break

        print(i)

        with torch.no_grad():
            out = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids)

        out = out.argmax(dim=1)
        correct += (out == labels).sum().item()
        total += len(labels)



    print(correct / total)

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc = torch.nn.Linear(768, 2)



    def forward(self, input_ids, attention_mask, token_type_ids):
        pretrained = BertModel.from_pretrained('bert-base-chinese')
        pretrained = pretrained.cuda()
        with torch.no_grad():
            out = pretrained(input_ids=input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids)

        out = self.fc(out.last_hidden_state[:, 0])
        out = out.softmax(dim=1)
        return out

model = Model()
model = torch.load(r'D:\h20240513\bert\checkpoint\td_bert.pth')
test(model)