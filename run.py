from Src.Model import RNN
from Src.DataPrep import DataPrep

import time
import math
import argparse
from sys import argv

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import torch
import torch.nn as nn


def time_since(s):
    now = time.time()
    s = now - s
    m = math.floor(s / 60)
    s -= m * 60
    return f'|{m:3}:{s:02.0f}|'

def fit(model, criterion, category_tensor, line_tensor):
    hidden = model.init_hidden()
    model.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = model(line_tensor[i], hidden)
    
    loss = criterion(output, category_tensor) #criterion
    loss.backward()

    for p in model.parameters():
        p.data.add_(p.grad.data, alpha=-lr)

    return output, loss.item()

def train(model, criterion, n_iters, print_every=5000):
    model.train()
    current_loss = 0
    global all_losses
    all_losses = []

    start = time.time()

    for iter in range(1, n_iters + 1):
        category, line, category_tensor, line_tensor = data.random_training_sample() 
        output, loss = fit(model, criterion, category_tensor, line_tensor)
        current_loss += loss

        if iter % print_every == 0:
            __, guess = data.cat_from_output(output, n_predictions=1)[0] 
            correct = '✓' if guess == category else f'✗ {category:10}'
            print(f'{iter:6} {iter / n_iters * 100:6.2f}% {time_since(start)} {loss:7.4f} {line:10} {guess:10} {correct}')

        if iter % 1000 == 0:
            all_losses.append(current_loss/ 1000)
            current_loss = 0
    model.eval()

def predict(model, line, top_n=3, prnt=True):
    model.eval()
    if prnt:
        print(f'> {line}↴')
    line_tensor = data.line2tensor(line)
    with torch.no_grad():
        hidden = model.init_hidden()

        for i in range(line_tensor.size()[0]):
            output, hidden = model(line_tensor[i], hidden)
        
        topv, topi = output.topk(top_n, 1, True)

        for i in range(top_n):
            value = topv[0, i].item()
            category_index = topi[0, i].item()
            if prnt:
                print(f'{value:4.2f} {data.all_categories[category_index]}')
    model.train()
    return data.all_categories[topi[0, 0].item()] 

def plot_loss():
    fig, ax = plt.subplots()
    ax.plot(all_losses)
    plt.show()


def plot_confusion(model, n=10000):
    confusion = torch.zeros(data.n_categories, data.n_categories) 

    for i in range(n):
        category, line, __, __ = data.random_training_sample()
        output = predict(model, line, top_n=1, prnt=False)
        category_i = data.all_categories.index(category)
        output_i = data.all_categories.index(output)
        confusion[category_i, output_i] += 1
    for i in range(data.n_categories):
        confusion[i] /= confusion[i].sum()

    fig, ax = plt.subplots()

    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)

    ax.set_xticklabels([''] + data.all_categories, rotation=90)
    ax.set_yticklabels([''] + data.all_categories)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    
ap = argparse.ArgumentParser()
ap.add_argument('-t', '--train', action='store_true')
ap.add_argument('-p', '--plot', action='store_true')
ap.add_argument('-c', '--classify')
options = ap.parse_args()
if __name__ == '__main__':
    data = DataPrep()
    rnn = RNN(data.n_letters, 128, data.n_categories)

    if options.train:
        criterion = nn.NLLLoss()

        lr = 0.005
        train(rnn, criterion, n_iters=100000)
        plot_loss()
        torch.save(rnn.state_dict(), 'rnn.pth')
    else:
        rnn.load_state_dict(torch.load('rnn.pth'))

    if options.plot:
        plot_confusion(rnn)
    
    if options.classify:
        predict(rnn, argv[-1], top_n=3, prnt=True)
