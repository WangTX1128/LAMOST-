import torch.nn as nn
import torch

# First checking if GPU is available


class BiLSTM(nn.Module):


    def __init__(self, input_size, output_size, hidden_dim, n_layers, bidirectional=True, drop_prob=0):
        """
        Initialize the model by setting up the layers.
        """
        super(BiLSTM, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers,
                            dropout=drop_prob, batch_first=True,
                            bidirectional=bidirectional)

        # linear layers
        if bidirectional:
            self.fc = nn.Linear(hidden_dim * 2, output_size)
        else:
            self.fc = nn.Linear(hidden_dim, output_size)


    def forward(self, x):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        batch_size = x.size(0)
        out,_ = self.lstm(x) #out = batch, seq_len, num_directions * hidden_size
        out1 = out[:, -1, :16] #最后一层正向传播的最后一个timestep
        out2 = out[:, 0, 16:]  #最后一层反向传播最后一个timestep
        out = torch.cat((out1,out2), dim=1)
        out = self.fc(out)

        return out

    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        number = 1
        if self.bidirectional:
            number = 2

        if torch.cuda.is_available():
            hidden = (weight.new(self.n_layers * number, batch_size, self.hidden_dim).zero_().cuda(),
                      weight.new(self.n_layers * number, batch_size, self.hidden_dim).zero_().cuda()
                      )
        else:
            hidden = (weight.new(self.n_layers * number, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers * number, batch_size, self.hidden_dim).zero_()
                      )

        return hidden

if __name__ == '__main__':

    input = torch.rand(64, 937, 4)
    model = BiLSTM(4, 4, 16, 5)#input_size, output_size, hidden_dim, n_layers
    out,h = model(input)
    print(out.shape)
    #print(h.shape)
    #print(out[:,0,:].shape)
    #print(out[:,-1,:].shape)
    #print(out[:,-1,:16] == h[8]) #h[0]第一层前向传播最后一个timestep ,h[1]第一层后向传播最后一个timestep
    #print(out[:,0,16:] == h[9])  #h[2]第二层前向                   ,h[3]第二层后向
                                #h[4]第三层前向                   ,h[5]第三层后向
                                #h[6]第四层前向                   ,h[7]第四层后向
                                #h[8]第五层前向                   ,h[9]第五层后向传播最后一个timestep

