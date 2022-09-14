from asyncio import futures
import torch
import torch.nn as nn

# import dataset class
from torch.utils.data import Dataset
import argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--date", type=int, default=0)
parser.add_argument("--context", type=int, default=100)
parser.add_argument("--future", type=int, default=10)

args = parser.parse_args()


class WeatherData(Dataset):
    def __init__(self, future=0):
        self.data = pd.read_csv("Data/raw_data_sinaica.csv")
        # self.data = self.data.dropna()
        # rellenar los datos con promedios moviles o interpolaciones
        self.data = self.data.interpolate()
        self.data["Time"] = pd.to_datetime(self.data["Time"])
        self.data = self.data.sort_values(by=["Estación", "Time"])
        self.data = self.data.set_index("Time")
        self.Municipios = self.data["Estación"].unique()
        self.Mundicc = {}
        # hacer un diccionario para el nombre de los municipios
        for i in range(len(self.Municipios)):
            self.Mundicc[self.Municipios[i]] = i
        # Aplicar el diccionario al dataframe
        self.data["Estación"] = self.data["Estación"].map(self.Mundicc)
        for col in self.data.columns:
            self.data = self.data[self.data[col] >= 0]
        # numpy array
        # agarrar solo datos de la estacion 0
        self.data = self.data[self.data["Estación"] == 0]
        # quitar columna estacíon
        self.data = self.data.drop(columns=["Estación"])
        self.columns = self.data.columns
        # standarizar los datos
        self.mean = self.data.mean()
        self.std = self.data.std()
        self.data = (self.data - self.mean) / self.std
        # correjir los NaN
        self.data = self.data.fillna(0)
        self.data = self.data.to_numpy()
        # convertir los numeros menores a 0 en
        self.x = torch.tensor(self.data[:-1], dtype=torch.float)
        self.y = torch.tensor(self.data[1:], dtype=torch.float)
        self.set_future(future)

    def set_future(self, future):
        self.future = future

    def __len__(self):
        return len(self.data) - self.future - 1

    def __getitem__(self, idx):
        return self.x[idx].unsqueeze(0), self.y[idx].unsqueeze(0)

    def get_batch(self, batch_size, future=0):
        idx = np.random.randint(0, len(self.data) - future - 1, batch_size)
        return self.x[idx], self.y[idx + future]


Data = WeatherData(future=args.future)


def LinearCell(num_input, num_hidden, Dropout=0):
    Seq = nn.Sequential(
        nn.Linear(num_input, num_hidden), nn.LeakyReLU(0.8), nn.Dropout(Dropout)
    )
    return Seq


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, num_linear=1):
        """
        input_size: input size
        hidden_size: hidden size
        output_size: output size
        num_layers: number of layers
        num_linear: number of linear layers
        """
        super(Net, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers)
        LinearSeq = []
        for i in range(num_linear):
            LinearSeq.append(LinearCell(hidden_size, hidden_size, Dropout=0))
        self.LinearSeq = nn.Sequential(*LinearSeq)
        self.L1 = LinearCell(hidden_size, hidden_size, Dropout=0)
        self.L2 = LinearCell(hidden_size, hidden_size, Dropout=0)
        self.LOut = LinearCell(hidden_size, output_size)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.normal_(m.bias)

    def forward(self, x, future=0):
        """
        x: input [sequence_length,batch_size,input_size]
        sequence_length: sequence length
        batch_size: batch size
        input_size: input size == 15
        future: number of future predictions
        """
        # outputs
        outputs = []
        h_t = torch.zeros(self.num_layers, 1, self.hidden_size)
        c_t = torch.zeros(self.num_layers, 1, self.hidden_size)

        for input_t in x.split(1, dim=0):
            out, (h_t, c_t) = self.lstm1(input_t, (h_t, c_t))
            # print (out.shape)
            out = self.LinearSeq(out)
            l1 = self.L1(out)
            l2 = self.L2(l1)
            output = self.LOut(l2)
            outputs.append(output)

        for i in range(future):
            out, (h_t, c_t) = self.lstm1(output, (h_t, c_t))
            out = self.LinearSeq(out)
            l1 = self.L1(out)
            l2 = self.L2(l1)
            output = self.LOut(l2)
            outputs.append(output)
        outputs = torch.cat(outputs, dim=0)
        return outputs


# load model weights
input_size = 15
argument = {
    "input_size": input_size,
    "hidden_size": 18,
    "output_size": input_size,
    "num_layers": 3,
    "num_linear": 2,
}
model = Net(**argument)
model.load_state_dict(torch.load("Model/modelApodaca4.2.pt"))

print("Date: ", args.date)
print("Contex: ", args.context)
print("Future: ", args.future)
print("Predicting...")
date, context = args.date, args.context
future = args.future
model.eval()
with torch.no_grad():
    x, y = Data[date:context]
    x = x.transpose(0, 1)
    out = model(x, future=future)
print("Results saved on output.csv")
out = out.squeeze(1).numpy()
out = out * Data.std.to_numpy() + Data.mean.to_numpy()
# convert to dataframe
out = pd.DataFrame(out, columns=Data.columns)
# save to csv
out.to_csv("output.csv", index=False)
