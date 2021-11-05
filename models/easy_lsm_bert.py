import torch
import torch.nn as nn
import transformers

pretrained_bert = transformers.BertModel.from_pretrained('bert-base-uncased', return_dict=False)


class LanguageModel(torch.nn.Module):
    def __init__(self):
        super(LanguageModel, self).__init__()
        self.bert_layer = pretrained_bert
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(768, 100)

    def forward(self, ids):
        _, output = self.bert_layer(ids)
        output = self.linear(self.dropout(output))
        return output


class ModelAudio(nn.Module):
    def __init__(self, feature_shape=128, num_layers_lstm=3, hidden_size=128, batch_size=16, bidirectional=True):
        super(ModelAudio, self).__init__()

        self.lstm = nn.LSTM(feature_shape, hidden_size=hidden_size, num_layers=num_layers_lstm,
                            bidirectional=bidirectional)
        self.dropout = nn.Dropout(0.2)
        self.activation = nn.ReLU()

        self.linear = nn.Linear(hidden_size * 2 * 150, 100)

    def forward(self, data):
        data = data.reshape(data.shape[0], data.shape[2], data.shape[1])
        batch_shape = data.shape[0]

        output, (h, c) = self.lstm(data)
        x = self.activation(output)
        x = x.view(batch_shape, -1)
        x = self.linear(x)
        return x


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

        self.language_model = LanguageModel()
        self.audio_model = ModelAudio()

        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(200, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text, audio):
        text = self.language_model(text)
        audio = self.audio_model(audio)

        output = torch.cat((text, audio), 1)
        output = torch.flatten(self.sigmoid(self.classifier(self.dropout(output))))
        return output
