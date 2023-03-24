---
layout: post
title:  "Using Neural Networks to Manage Invested Capital"
date:   2023-03-24 01:08:51 +0000
categories: jekyll update
---
Since September 2022, I've been working on a capstone project for my Master's degree along with some team members.  As the school year wraps up, I figured this would be a good time to summarize the progress that has been made up to this point.  In this post, I'll be focusing on providing a bit of background for the project and highlighting some of the models that I built for the project. 

### Background
First, a little bit of background about the topic itself.  You may know something about pension funds or you might be a pensioner yourself.  Pension funds essentially take money, in lump sum payments or over time, and distribute it to retirees who've met the requirements to receive a distribution.  The pension funds don't just hold onto the money they have in cash.  More recently, they have been committing some portion of it to [private equity or venture capital firms](https://www.investopedia.com/articles/investing-strategy/090916/how-do-pension-funds-work.asp) who invest the money on their behalf and distribute profits, just like the pension funds to pensioners.  The same thing occurs at sovereign wealth funds and other institutional investment funds. 

Because these institutional investors are committing tons of money, the PE/VC firms don't expect them to pay the full amount up front.  From time to time, they will send a [capital call or drawdown](https://carta.com/blog/capital-call/) which gives their limited partners (the investors in the fund) notice of a potential deal and instructions to wire some amount of their share so the general partner (usually the PE/VC firm) can make the transaction.  The opposite of a capital call is a capital distribution.  

The pension fund is now posed with a problem: What to do with the committed capital?  If they just hold the total amount of their commitment in cash, then they'll be missing out on returns.  If they invest too much of their committed capital and can't come up with the cash when a capital call comes, then they'll be subject to some penalty specified by the fund terms.  A potentially lucrative deal could even fall through.  

This is where our project fills in the gap.  Our group built and tested a whole suite of models to try and predict the amounts of capital calls based on historical data.  Originally, we spent our time trying to find correlations between market data, like S&P500 levels, and capital calls.  Long story short, that didn't pan out and we were forced to pivot to a different idea.  We still managed to get some decent models, and I'll be focusing on those that I coded and created.    

### Models and Data

The main investment fund data was sourced from [Idaho's Pension fund](https://www.persi.idaho.gov/investments/).  In addition, I used a perturbed sine wave to ensure that the models were working as intended.  All of the models I coded were in PyTorch, and for the Idaho data, they were trained on an MSI running Ubuntu 22.04 with a GeForce RTX 3060.  

First, here's a look at the data for one of the funds that the Idaho Pension fund keeps money with: 
<img src="/assets/img/Figure1.png">
The Idaho data had the fields graphed above, for 138 unique funds.  Many of these funds had data spanning across 91 quarters but others did not.  
