---
layout: post
title:  "Using Neural Networks to Help Manage Invested Capital"
date:   2023-03-24 01:08:51 +0000
categories: jekyll update
---
I've been working on a capstone project for my Master's degree along with some team members.  As the school year wraps up, I figured this would be a good time to summarize the progress that has been made up to this point.  In this post, I'll be focusing on providing a bit of background for the project and highlighting some of the models that I built for the project.  There are a few more steps before the project is done, which I'll mention in this post as well.  

### Background
First, a little bit of background about the topic itself.  You may know something about pension funds or you might be a pensioner yourself.  Pension funds essentially take money, in lump sum payments or over time, and distribute it to retirees who've met the requirements to receive a distribution.  The pension funds don't just hold onto the money they have in cash.  More recently, they have been committing some portion of it to [private equity or venture capital firms](https://www.investopedia.com/articles/investing-strategy/090916/how-do-pension-funds-work.asp) who invest the money on their behalf and distribute profits, just like the pension funds to pensioners.  The same thing occurs at sovereign wealth funds and other institutional investment funds. 

Because these institutional investors are committing tons of money, the PE/VC firms don't expect them to pay the full amount up front.  From time to time, they will send a [capital call or drawdown](https://carta.com/blog/capital-call/) which gives their limited partners (the investors in the fund) notice of a potential deal and instructions to wire some amount of their share so the general partner (usually the PE/VC firm) can make the transaction.  The opposite of a capital call is a capital distribution.  

The pension fund is now posed with a problem: What to do with the committed capital?  If they just hold the total amount of their commitment in cash, then they'll be missing out on returns and suppressing their IRR.  If they invest too much of their committed capital and can't come up with the cash when a capital call comes, then they'll be subject to some penalty specified by the fund terms.  A potentially lucrative deal could even fall through.  

This is where our project fills in a gap.  Our group built and tested a suite of models to try and predict the amounts of capital calls based on historical data.  I'll be focusing on those that I created.    

### Data and Cleaning

The main investment fund data was sourced from [Idaho's Pension fund](https://www.persi.idaho.gov/investments/).  In addition, I used a perturbed sine wave to ensure that the models were working as intended.  

First, here's a look at the data for one of the funds that the Idaho Pension fund keeps money with: 
<img src="/assets/img/capstone_post/Figure1.png">
The Idaho data had the fields graphed above, for 138 unique funds.  Some of these funds had data spanning across 91 quarters but others did not.  In the instance above, there were 43 quarters worth of data on Chisholm Partners IV.  The data was originally organized into excel files released quarterly, and our team aggregated everything into one source.  In total, we had 4332 quarters worth of data across the 138 funds.  

It's apparent that the numbers that we're dealing with are massive (except for the rates), so prior to feeding anything into my models, I performed a [MinMax scaling](https://towardsdatascience.com/everything-you-need-to-know-about-min-max-normalization-in-python-b79592732b79#:~:text=Variables%20that%20are%20measured%20at,used%20prior%20to%20model%20fitting.) to get everything between [0,1].  I decided this was the most appropriate scaling since it was the simplest for this data.  Nothing was normally distributed, so Z-scaling felt forced, and unit length scaling would've been too complicated.  This was one of the handful of regularization techniques. 

To ensure that the models were working as expected, I used a perturbed sine wave as a first test.  

### The Models 
All of the models I coded were in PyTorch, and for the Idaho data, they were trained on an MSI running Ubuntu 22.04 with a GeForce RTX 3060. I created two different models, one simple and one more complex.  

#### Inputs 
Inputs to the models were fed in sequences of 4, fund by fund.  The predictors for the Idaho data was lagged capital calls. Essentially, the model was fed the previous 4 quarters worth of capital calls and asked to predict the value in the next quarter. The sequences were length 4 so the model could pick up the seasonality within the data over a year (4 quarters).  Since the MLP doesn't have a recurrent structure, the sequence was flattened and then fed into the model.   In addition, padding was added so that if the batch number loaded from the dataset was less than the window size of 4 then repeated values were added as padding.  For example, for batch i = 3 for the Idaho data, the models were given 

{% highlight python %}
[tensor([[[0.0455],
         [0.0455],
         [0.0675],
         [0.0879]]]), tensor([0.0895])]
{% endhighlight %}
where 0.0455, the first value of Paid In Capital, is repeated twice to pad the input sequence and 0.0895 is the target value. 


For the perturbed sine wave test, the models were provided with the x values and previous (lagged) values of the perturbed sine wave.  I used the following customized data sets to feed the models.  I utilized the `SequenceData` structure found at [this resource](https://www.crosstab.io/articles/time-series-pytorch-lstm/#data) for the sine and Idaho data set. The `IdahoData` was constructed so there would be no overlap of data across funds.

{% highlight python %}

class SequenceData(Dataset):
    """
    Class for easily loading sequential data
    """

    def __init__(self, y_col: str, x_col: List[str], window: int, data_path: str = None, df_arg: pd.DataFrame = None):
        """
        params
        ------
        y_col:: target column(s)
        x_col:: predictors
        window:: how long each subsequence is
        data_path:: where data lives

        """
        if data_path:
            df = pd.read_csv(data_path)
        else:
            df = df_arg

        self.features = x_col
        self.target = y_col
        self.window = window
        self.y = torch.tensor(df[y_col].values).float()
        self.X = torch.tensor(df[x_col].values).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i): 
        if i >= self.window - 1:
            i_start = i - self.window + 1
            x = self.X[i_start:(i + 1), :]
        else:
            padding = self.X[0].repeat(self.window - i - 1, 1)
            x = self.X[0:(i + 1), :]
            x = torch.cat((padding, x), 0)

        return x, self.y[i]

class IdahoData(Dataset):
    """
    Class for loading the Idaho Dataset
    """
    
    def __init__(self, y_col: str, x_col: str, window: int, data_path: str):
        """
        Same as Sequence Data
        """
        self.y_col = y_col
        self.x_col = x_col
        self.window = window
        self.X = []
        self.y = []
        df = pd.read_csv(data_path)

        fund_names = df["Investment Name"].unique()

        for fund in fund_names:
            fund_spec_data = df[df["Investment Name"] == fund].copy()
            name = f"shifted_{y_col}"
            shifted_y = fund_spec_data[y_col].shift(1)
            fund_spec_data[name] = shifted_y
            fund_sequence = SequenceData(y_col, x_col + [name], window, data_path = None, df_arg = fund_spec_data[1:])
            for j in range(len(fund_sequence)):
                X, y = fund_sequence.__getitem__(j)
                self.X.append(X)
                self.y.append(y)
            
    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

{% endhighlight %}

#### Simple MLP
The simplest NN model I could built for this problem was just an MLP.  I tried to write the code so that I could add dropouts, bias, layers, etc. at will, which will help further down the line with 5/10 fold cross validations to find the best hyperparameters.

{% highlight python %}
class GeneralNN(nn.Module):
    """
    General, modular MLP

    adds relu after every hidden layer

    params
    --------
    input_len:: size of the input, i.e. number of nodes in input layer
    output_len:: size of output
    hidden_dim:: gives number of hidden layers (len(hidden_dim)) and the dimensions in each
    dropout:: adds dropout after every hidden layer if its nonzero 
    bias:: add bias term or not
    """
    def __init__(self, input_len: int, output_len: int, hidden_dim: List[int], dropout: float = 0, bias = True):
        super().__init__()

        self.input_len = input_len
        self.output_len = output_len
        if hidden_dim: # check if there's anything in hidden_dim
            layers = [nn.Linear(input_len, hidden_dim[0], bias = bias)]

            for i, hidden in enumerate(hidden_dim[:-1]):
                layer = [nn.Linear(hidden, hidden_dim[i+1], bias = bias), nn.ReLU()]
                if dropout:
                    layer.append(nn.Dropout(dropout))
                layers += layer

            layers += [nn.Linear(hidden_dim[-1], output_len, bias = bias)]
        else:
            layers = [nn.Linear(input_len, output_len, bias = bias)]

        self.main = nn.Sequential(*layers)

    def forward(self, X):
        output = self.main(X)
        return output

{% endhighlight %}

The initial architecture I used was just 1 fully connected hidden layer with 32 nodes, initialized with 

{% highlight python %}
input_len = len(torch.flatten(X))
output_len = 1
hidden_dim = [32, 32, 32]
bias = True
dropout = 0.2

model = GeneralNN(input_len, output_len, hidden_dim, dropout, bias).to(device)
{% endhighlight %}

#### LSTM RNN
I decided to also use a stacked, LSTM RNN.  We did not have a large amount of data and this architecture can capture "seasonal" patterns demonstrated by its use in NLP applications.  This implementation of the model just doesn't have an embedding layer since I'm not training on text.  The following code is how I implemented it 

{% highlight python %}
class LSTM(nn.Module):

    def __init__(self, num_hidden, num_features, out_features, num_layers, dropout, batch_first = True):
        super().__init__()
        self.num_sensors = num_features
        self.num_hidden = num_hidden
        self.out_features = out_features
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size = num_features, hidden_size = num_hidden, batch_first = batch_first, num_layers = num_layers, dropout = dropout)
        self.linear = nn.Linear(in_features = self.num_hidden, out_features = out_features)

    def forward(self, X):
        batch_size = X.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.num_hidden).requires_grad_().to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.num_hidden).requires_grad_().to(device)

        out, (hn,cn) = self.lstm(X, (h0, c0))
        output = self.linear(hn[0]).flatten()

        return output

{% endhighlight %}
As you can see, I take the the last hidden state of the LSTM model and run that through the linear output layer to get to the number of time points predicted given by `out_features`.  I chose to have 32 features in the hidden state and a dropout rate of 0.2, similar to the simple MLP.  

{% highlight python %}
num_features = len(x_cols) + 1
out_features = 1

num_layers = 2
num_hidden = 32
dropout = 0.2

model1 = LSTM(num_hidden, num_features, out_features, num_layers, dropout)
{% endhighlight %}
To read more about this implementation, I recommend taking a look at the docs [here](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html).  

#### Optimizers and Loss
I used the following optimizer and loss function for both models.  `model1` refers to the LSTM model. 
{% highlight python %}
lr = 10**(-4)
optimizer = torch.optim.Adam(params = model1.parameters(), lr = lr)
loss = nn.MSELoss()
{% endhighlight %}
I chose `torch.optim.Adam()` as the optimizer due to its well documented advantages, specifically for non-stationary data. Since this problem is a regression problem, I decided that mean square error was the most appropriate. I came to a learning rate of 10**(-4) after tinkering around with the models on the perturbed sine wave.  I'm planning implementing a cross validation to rigorously determine the best rate, as mentioned earlier.  

#### Training and Testing

I used the same training and testing loops for both models.  Here's the code for both across one full pass (epoch) through the training/testing sets.

{% highlight python %}
def train(training_data, model, loss, optimizer):
    # training for one epoch
    model.train()
    total = 0
    actuals = []
    pred = []
    for i, batch in enumerate(training_data):
        X,y = batch
        # X = X.permute(1,0,2) do this if batch_first = False
        X,y = X.to(device), y.to(device)
        output = model(X)
        ls = loss(output, y)

        optimizer.zero_grad()
        ls.backward()
        optimizer.step()
        total += ls.item()  

        actuals.append(y)
        pred.append(output)

    mean_loss = total/len(training_data)
    print(f"Average training loss is {mean_loss}")

    return actuals, pred, mean_loss

def test(testing_data, model, loss):
    # testing for one epoch
    model.eval()
    actuals, pred = [], []
    total = 0

    with torch.no_grad():
        for i, batch in enumerate(testing_data):
            X,y = batch
            # X = X.permute(1,0,2) do this if batch_first = False
            X,y = X.to(device), y.to(device)
            output = model(X)
            ls = loss(output, y)
            actuals.append(y[0].cpu().detach().numpy())
            pred.append(output[0].cpu().detach().numpy())

            total += ls.item()

    mean_loss = total/len(testing_data)
    print(f"Average testing loss is {mean_loss}")
    return actuals, pred, mean_loss
{% endhighlight %}

I also implemented early stopping to prevent overfitting.  When average validation error increased, I stopped training and saved the model binary.  

For both the perturbed sine wave and Idaho data, I used a 80/20 training testing split and each model was trained on the same training and testing data.  

### Results

#### Simple MLP
The following graphs show the performance of the MLP on the perturbed sine wave in the validation set. 
<img src="/assets/img/capstone_post/sine/preds_vs_vals_mlp.png">
It's apparent that the model converges quickly, in only 4 epochs.  In addition, the early stopping procedure seems to be working since the model is picking up on the true sine wave relationship, not the random noise injected into it.  Here's a look at some graphs that plot the epochs vs. loss.  
<img src="/assets/img/capstone_post/sine/epoch_vs_loss_mlp.png">
Based on the performance, the model architecture was a decent candidate to train on the Idaho data.  When I applied this model on the Idaho data, it took only 4 epochs to converge.  Here's a look at the model performance on the validation set.
<img src="/assets/img/capstone_post/idaho/preds_vs_vals_mlp.png">
The best validation error achieved was 0.0005, and again, the model converges relatively quickly before it starts overfitting.  Note that the best error rate may be a bit misleading since I performed a MinMax scaling on the original data.  It would be more accurate to think of this in terms of percentages.  Essentially, the model is off by about 0.5% relative to the size of the capital call.  Since the capital calls are sometimes in the tens of millions, at worst, the model can predict the value of the capital call &plusmn; $100,000.  Here is a look at the epochs vs. loss plot
<img src="/assets/img/capstone_post/idaho/epoch_vs_loss_mlp.png">

#### LSTM RNN
On the other hand, the LSTM RNN model took many epochs to train, but achieved better accuracy.  
<img src="/assets/img/capstone_post/sine/preds_vs_vals_lstm.png">
The graph above shows the model's results after the first 5 epochs. It took only 12 epochs to converge which is about 3 times as long as the MLP. However, there performance was slighly better, as the predictions nearly overlay the true sine wave. This suggests that the recurrent architecture captures the periodic trends within the data which boded well for the Idaho data.  

When applied to the Idaho data, the same trend emerged: longer training better accuracy.  
<img src="/assets/img/capstone_post/idaho/preds_vs_vals_lstm.png">
Again, it took much longer to train this model compared to the MLP with the modeling converging on epoch 46.  However, the accuracy is about twice as good with the an error of 0.2% compared to the MLP model. 

### Conclusions
Overall, the models are performing well in the framework that was constructed for them.  When training on the the capital demands for other funds, they are able to predict the sequence of capital demands for other funds.  This might suggest a few things:
- There are market conditions that each fund is responding to and they have similar investment strategies that cause them to generally require capital at similar times
- Some funds may act as "leaders."  For example, KKR may be especially active in a quarter which compels other funds to make more transactions as well.  These types of "leader" funds could have been in the training set, and hence the model is basing its predictions from those funds. 

If the first assumption is true, then incorporating other market variables into the set of predictors may increase model performance.  To pursue this route, we would ideally have data at a higher granularity than the quarter level and much more data over time.  On the other hand, if there are truly some "leaders" amongs the funds, then a model that takes advantage of leading indicators may be a good way to forecast demand.  Traditionally, these types of models are used in demand forecasting and supply chain operations, but with our current data, it may be possible to test this hypothesis.  

Given our constraints on data, the second hypthesis deserves further exploration.  I also plan on using cross validation to optimize the hyperparameters for the original models.  The hyperparameters that I used were chosen with a bit of tinkering and a more robust procedure is necessary to ensure that everything is optimal.

Thanks for reading this article and feel free to reach out at the contact information in the footer to discuss more!