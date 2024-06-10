import torch
from datasets import load_from_disk
from transformers import BertTokenizer
from transformers import BertModel
from transformers import AdamW


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

dataset = Dataset()
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


loader = torch.utils.data.DataLoader(dataset=dataset,
                                     batch_size=32,
                                     collate_fn=collate_fn,
                                     shuffle=True,
                                     drop_last=True)
for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(loader):
    break

pretrained = BertModel.from_pretrained('bert-base-chinese')
pretrained = pretrained.cuda()


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.decoder = torch.nn.Linear(768, token.vocab_size, bias=False)
        self.bias = torch.nn.Parameter(torch.zeros(token.vocab_size))
        self.decoder.bias = self.bias


    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            out = pretrained(input_ids=input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids)
        out = self.decoder(out.last_hidden_state[:, 15])
        return out

model = Model()
model = model.cuda()
# print(model(input_ids=input_ids,
#       attention_mask=attention_mask,
#       token_type_ids=token_type_ids).shape)

optimizer = AdamW(model.parameters(), lr=5e-4)
criterion = torch.nn.CrossEntropyLoss()

model.train()
epochs = 5
for epoch in range(epochs):
    step_iter = 0
    loss_sum = 0
    acc_sum = 0
    for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(loader):
        step_iter += 1
        input_ids, attention_mask, token_type_ids, labels = input_ids.cuda(), attention_mask.cuda(), token_type_ids.cuda(), labels.cuda()
        out = model(input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids)

        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_sum += loss

        out = out.argmax(dim=1)
        accuracy = (out == labels).sum().item() / len(labels)
        acc_sum += accuracy
    print(epoch, loss_sum / step_iter, acc_sum / step_iter)



torch.save(model, r'D:\h20240513\bert\checkpoint\tk_bert.pth')