import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
    
    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)
        return hidden, cell

 
class LSTMDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, output_regions, output_seq_len, num_layers=1, dropout=0.0):
        super().__init__()
        self.output_regions = output_regions
        self.output_seq_len = output_seq_len
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size=output_regions * output_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_size, output_regions * output_size)

    def forward(self, hidden, cell, target_output=None, teacher_forcing_ratio=0.5):
        batch_size = hidden.size(1)
        device = hidden.device

        input_step = torch.zeros(batch_size, 1, self.output_regions * self.output_size, device=device)
        outputs = []

        for t in range(self.output_seq_len):
            output, (hidden, cell) = self.lstm(input_step, (hidden, cell))
            output_projected = self.fc(output.squeeze(1))
            output_reshaped = output_projected.view(batch_size, self.output_regions, self.output_size)
            outputs.append(output_reshaped)

            use_teacher_forcing = (target_output is not None) and (torch.rand(1).item() < teacher_forcing_ratio)
            if use_teacher_forcing:
                input_step = target_output[:, :, t].reshape(batch_size, 1, -1)
            else:
                input_step = output_projected.unsqueeze(1)

        outputs = torch.stack(outputs, dim=2)
        return outputs.squeeze(-1) if self.output_size == 1 else outputs


class EncoderDecoderModel(nn.Module):
    def __init__(self, input_features, hidden_size, output_regions, output_seq_len, output_features=1, num_layers=1, dropout=0.0):
        super().__init__()
        self.encoder = LSTMEncoder(input_size=input_features, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        self.decoder = LSTMDecoder(hidden_size=hidden_size,
                                   output_size=output_features,
                                   output_regions=output_regions,
                                   output_seq_len=output_seq_len,
                                   num_layers=num_layers,
                                   dropout=dropout)

    def forward(self, x, target_output=None, teacher_forcing_ratio=0.5):
        x = x.permute(0, 2, 1)
        hidden, cell = self.encoder(x)
        output = self.decoder(hidden, cell, target_output, teacher_forcing_ratio)
        return output
