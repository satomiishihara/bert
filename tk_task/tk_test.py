import torch
from datasets import load_from_disk
from transformers import BertTokenizer
from transformers import BertModel

class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        dataset = load_from_disk(r'D:\h20240513\seamew\ChnSentiCorp')

        def f(data):
            return len(data['text']) > 30

        self.dataset = dataset.filter(f)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        text = self.dataset[i]['text']

        return text

token = BertTokenizer.from_pretrained('bert-base-chinese')

def collate_fn(data):


    data = token.batch_encode_plus(batch_text_or_text_pairs=data,
                                   truncation=True,
                                   padding='max_length',
                                   max_length=30,
                                   return_tensors='pt',
                                   return_length=True)

    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    token_type_ids = data['token_type_ids']
    # 把一句话中第15个词作为mask
    labels = input_ids[:, 15].reshape(-1).clone()
    input_ids[:, 15] = token.get_vocab()[token.mask_token]

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
        if i == 15:
            break

        print(i)

        with torch.no_grad():
            out = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids)

        out = out.argmax(dim=1)
        correct += (out == labels).sum().item()
        total += len(labels)

        print(token.decode(input_ids[0]))
        print(token.decode(labels[0]), token.decode(out[0]))

    print(correct / total)

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.decoder = torch.nn.Linear(768, token.vocab_size, bias=False)
        self.bias = torch.nn.Parameter(torch.zeros(token.vocab_size))
        self.decoder.bias = self.bias


    def forward(self, input_ids, attention_mask, token_type_ids):
        pretrained = BertModel.from_pretrained('bert-base-chinese')
        pretrained = pretrained.cuda()
        with torch.no_grad():
            out = pretrained(input_ids=input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids)
        out = self.decoder(out.last_hidden_state[:, 15])
        return out

model = Model()
model = torch.load(r'D:\h20240513\bert\checkpoint\tk_bert.pth')
test(model)