from google.colab import drive
drive.mount('/content/drive')

!pip install pkbar

!pip install music21

import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class MojModel(nn.Module):
    def __init__(self, dim_len, out_len, hidden_units=256, drop_prob=0.2, num_layers=2):
        super(MojModel, self).__init__()
        self.dim_len = dim_len
        self.embeddings = nn.Embedding(out_len, dim_len)
        self.lstm_cell1 = nn.LSTM(input_size=dim_len, hidden_size=hidden_units, dropout=drop_prob, num_layers=num_layers, batch_first=True)
        self.fc_layer2 = nn.Linear(in_features=hidden_units*dim_len, out_features=out_len)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input):
        emb = self.embeddings(input)
        #ovo je potrebno raditi samo u slučaju treninga na više GPU-ova jer niz bude rascjepkan.
        emb = pack_padded_sequence(emb, np.full(input.size(0), self.dim_len), batch_first=True)
        self.lstm_cell1.flatten_parameters()
        h1, _ = self.lstm_cell1(emb)
        h1, _ = pad_packed_sequence(h1, batch_first=True)
        out = h1.contiguous().view(input.size(0), -1)
        lstm_to_out = self.fc_layer2(out)
        gen_feats = self.softmax(lstm_to_out)
        return gen_feats

import random

instrumenti = [instrument.AcousticGuitar(), instrument.AcousticBass(), instrument.AltoSaxophone(), instrument.BassDrum(), instrument.BassClarinet(), instrument.ChurchBells(), instrument.ElectricGuitar(),
              instrument.Mandolin(), instrument.Percussion(), instrument.Piano(), instrument.Trumpet()]

def random_instrument():
  random_index = random.randint(0, len(instrumenti) - 1)
  return instrumenti[random_index]

print(random_instrument())

import glob
import numpy as np
import os.path
import re
import pickle
from music21 import converter, instrument, note, chord

pickle_file = "/content/drive/MyDrive/DL/Lab09/data/cache_data.pkl"
numpy_file = "/content/drive/MyDrive/DL/Lab09/data/numpy_data.npy"

def load_from_pickle():
    print('Opening from pickle cache')
    with open(pickle_file, "rb") as f:
        data = pickle.load(f)
    with open(numpy_file, "rb") as f:
        network_input = np.load(f)
        network_output = np.load(f)
    return network_input, network_output, data[0], data[1], data[2]

def save_to_pickle(network_input, network_output, pitchnames, indices_note, note_indices):
    print('Saving to pickle')
    with open(pickle_file, "wb") as f:
        pickle.dump([pitchnames, indices_note, note_indices], f)
    with open(numpy_file, "wb") as f:
        np.save(f, network_input)
        np.save(f, network_output)

def prepare_data_from_midi_dir(dir, step=1, maxlen=50, vectorization=True, reload_fresh=False):
    if not reload_fresh and os.path.isfile(pickle_file):
        print('Loading from pickle file...')
        return load_from_pickle()

    print('Loading from midi files...')
    input_data = []
    next_note = []
    svi = []

    ran_instrument = random_instrument()

    for file in glob.glob(dir + "/*.mid"):
        print("Parsing %s" % file)

        midi = converter.parse(file)

        for part in midi.parts:
          part.insert(0, ran_instrument)

        for el in midi.recurse():
          el.activeSite.replace(el, ran_instrument)

        notes_to_parse = None

        try: # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            #s2.parts[0].insert(0, ran_instrument)
            notes_to_parse = s2.parts[0].recurse()
        except: # file has notes in a flat structure
            notes_to_parse = midi.flat.notes

        for element in midi.recurse():
            #element.activeSite.replace(element, ran_instrument)
            if isinstance(element, note.Note):
                svi.append(str(element.pitch))
                #svi.append(element.duration)
            elif isinstance(element, chord.Chord):
                svi.append('.'.join(str(n) for n in element.normalOrder))


    pitchnames = sorted(set(svi))

    note_indices = dict((n, i) for i, n in enumerate(pitchnames))
    indices_note = dict((i, n) for i, n in enumerate(pitchnames))

    print('total notes:', len(pitchnames))

    network_input = None
    network_output = None

    if vectorization: # radimo vektorizaciju
        print('Vectorization...')
        for i in range(0, len(svi) - maxlen, step):
            input_data.append(svi[i: i + maxlen])
            next_note.append(svi[i + maxlen])
            if i % 10000 == 0:
                print('Sequenced', i, 'of', len(svi))
        print('Number of sequences:', len(input_data))


        network_input = np.zeros((len(input_data), maxlen), dtype=np.int64)
        network_output = np.zeros((len(input_data), 1), dtype=np.int64)

        for i, (item, next_c) in enumerate(zip(input_data, next_note)):
            network_input[i] = np.fromstring(' '.join(map(lambda p: str(note_indices[p]), item)), dtype=np.int64, sep=' ')
            network_output[i] = note_indices[next_c]
            if i % 10000 == 0:
                print('Vectorized', i, 'of', len(input_data))

    save_to_pickle(network_input, network_output, pitchnames, indices_note, note_indices)

    return network_input, network_output, pitchnames, indices_note, note_indices

from torch.utils.data import Dataset

class TrainingDataset(Dataset):
    def __init__(self, network_input, network_output):
        self.network_input = network_input
        self.network_output = network_output

    def __len__(self):
        return len(self.network_input)

    def __getitem__(self, idx):
        return self.network_input[idx], self.network_output[idx]

import os

import pkbar
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

text_files_dir = "/content/drive/MyDrive/DL/Lab09/midis"

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

batch_size = 128
model_ver = 'Model_LSTM'
epochs = 100
maxlen = 50

cuda_device_number = 1

cuda = True if torch.cuda.is_available() else False
device = torch.device('cpu')

print('Torch version:', torch.__version__)

network_input, network_output, words, indices_char, char_indices = prepare_data_from_midi_dir(text_files_dir, maxlen=maxlen)

writer = SummaryWriter('/content/drive/MyDrive/DL/Lab09/runs/' + model_ver)

loss_fn = torch.nn.CrossEntropyLoss()

model = MojModel(maxlen, len(words))

if cuda:
    device = torch.device('cuda')
    model.cuda()
    if len(list(range(cuda_device_number))) > 1:
        model = nn.DataParallel(model, device_ids=list(range(cuda_device_number))).to(device)

    loss_fn = loss_fn.cuda()

optimizer = torch.optim.Adadelta(model.parameters())

trainig_dataset = TrainingDataset(network_input, network_output)
dataloader = DataLoader(trainig_dataset, batch_size=batch_size, num_workers=16, shuffle=True, drop_last=True, pin_memory=True)
train_per_epoch = int(len(network_input) / batch_size)

for epoch in range(epochs):
    print('\nEpoch: %d/%d' % (epoch + 1, epochs))
    kbar = pkbar.Kbar(target=train_per_epoch, width=20)

    for i, (input, output) in enumerate(dataloader):
        optimizer.zero_grad()
        inp = input.type(torch.LongTensor).to(device)
        out = output.type(torch.LongTensor).to(device)
        output = model(inp)
        loss = loss_fn(output, out.contiguous().view(-1))
        loss.backward()

        optimizer.step()
        kbar.update(i, values=[("loss", loss)])

        writer.add_scalar('training loss', loss.item(), (epoch * train_per_epoch) + i)

    path_str = os.path.join('/content/drive/MyDrive/DL/Lab09/models/', model_ver + '_lab_' + str(epoch + 1) + '.pth')
    torch.save(model.state_dict(), path_str)

import os
import random
import torch
from torch import nn
from music21 import instrument, note, stream, chord

midi_files_dir = "/content/drive/MyDrive/DL/Lab09/midis"
model_to_load = "/content/drive/MyDrive/DL/Lab09/models/Model_LSTM_lab_100.pth"

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

batch_size = 128
epochs = 100
maxlen = 50

cuda_device_number = 1

def sample(preds, temperature=1):
    preds = preds / temperature
    exp_preds = torch.exp(preds)
    preds = exp_preds / torch.sum(exp_preds)
    probas = torch.multinomial(preds, 1)
    return probas


def create_midi(prediction_output):
    print('creating midi')
    offset = 0
    output_notes = []

    ran_instrument = random_instrument()

    for pattern in prediction_output:
        # pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                #new_note.storedInstrument = ran_instrument
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # pattern is a note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            #new_note.storedInstrument = ran_instrument
            output_notes.append(new_note)

        offset += 0.5

    midi_stream = stream.Stream(output_notes)

    midi_stream.write("midi", fp="/content/drive/MyDrive/DL/Lab09/output/song31.mid")
    print('midi created')


cuda = True if torch.cuda.is_available() else False
device = torch.device('cpu')

print('Torch version:', torch.__version__)

network_input, network_output, words, indices_char, char_indices = prepare_data_from_midi_dir(midi_files_dir, maxlen=maxlen)

model = MojModel(maxlen, len(words))

if cuda:
    device = torch.device('cuda')
    model.cuda()
    if len(list(range(cuda_device_number))) > 1:
        model = nn.DataParallel(model, device_ids=list(range(cuda_device_number))).to(device)

loaded = torch.load(model_to_load)
model.load_state_dict(loaded)

initial_seed = random.randint(0, len(network_input))

rolling_values = torch.from_numpy(network_input[initial_seed]).cuda().unsqueeze(0)
# ovo je primjer gdje je batch size 1. U istom vremenu možemo generirati zapravo puno više primjera.

izlaz = []

with torch.no_grad():
    for i in range(500):
        output = model(rolling_values)
        output = sample(output)
        next_char = indices_char[output.item()]
        rolling_values = torch.cat((rolling_values, output), dim=-1)[0,1:].unsqueeze(0)

        izlaz.append(indices_char[output.item()])

create_midi(izlaz)

s = converter.parse('/content/drive/MyDrive/DL/Lab09/midis/Cymatics - Alive - 150 BPM F Min.mid')

for p in s.parts:
  p.insert(0, instrument.Baritone())

s.write('midi', '/content/drive/MyDrive/DL/Lab09/output/custom_01_THEME.mid')