# ============================================================
# DIINA v19.1
# Stable Long Distance Dependency Benchmark
# Transformer vs Mamba vs DIINA
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim

import random
import time
import pandas as pd
import matplotlib.pyplot as plt


print("Torch:", torch.__version__)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)


torch.manual_seed(42)
random.seed(42)


# ============================================================
# Vocabulary
# ============================================================

TOKENS = [
    "A","B","C","D",
    "x","y","z","n",
    "QUERY"
]


stoi = {
    t:i for i,t in enumerate(TOKENS)
}


labels = {
    "A":0,
    "B":1,
    "C":2,
    "D":3
}



# ============================================================
# Dataset
# ============================================================


def create_dataset(distance, samples):

    dataset=[]

    for _ in range(samples):

        key=random.choice(
            list(labels.keys())
        )


        noise=[
            random.choice(
                ["x","y","z","n"]
            )
            for _ in range(distance)
        ]


        seq=[key]+noise+["QUERY"]


        dataset.append(
            (
                seq,
                labels[key]
            )
        )


    return dataset



def encode(seq):

    return torch.tensor(
        [stoi[x] for x in seq],
        dtype=torch.long
    )



# ============================================================
# Transformer
# ============================================================


class TransformerModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.emb=nn.Embedding(
            len(stoi),
            32
        )


        layer=nn.TransformerEncoderLayer(
            d_model=32,
            nhead=4,
            batch_first=True
        )


        self.encoder=nn.TransformerEncoder(
            layer,
            num_layers=2
        )


        self.fc=nn.Linear(
            32,
            4
        )


    def forward(self,x):

        h=self.emb(x)

        h=self.encoder(h)

        return self.fc(
            h[:,-1,:]
        )



# ============================================================
# Mamba baseline
# ============================================================


class MambaBaseline(nn.Module):

    def __init__(self):

        super().__init__()

        self.emb=nn.Embedding(
            len(stoi),
            32
        )


        self.rnn=nn.GRU(
            32,
            32,
            batch_first=True
        )


        self.fc=nn.Linear(
            32,
            4
        )


    def forward(self,x):

        h,_=self.rnn(
            self.emb(x)
        )

        return self.fc(
            h[:,-1,:]
        )



# ============================================================
# DIINA
# ============================================================


class DIINA(nn.Module):

    def __init__(self):

        super().__init__()


        self.emb=nn.Embedding(
            len(stoi),
            32
        )


        self.rnn=nn.GRU(
            32,
            32,
            batch_first=True
        )


        self.inhibit=nn.Sequential(
            nn.Linear(32,32),
            nn.Sigmoid()
        )


        self.fc=nn.Linear(
            32,
            4
        )


    def forward(self,x):

        h,_=self.rnn(
            self.emb(x)
        )


        gate=self.inhibit(h)


        # dynamic inhibition

        h=h*(1-gate)


        return self.fc(
            h[:,-1,:]
        )



# ============================================================
# Training
# ============================================================


def train(model,data,epochs=20):

    model.to(device)

    optimizer=optim.Adam(
        model.parameters(),
        lr=0.001
    )

    loss_fn=nn.CrossEntropyLoss()


    start=time.time()


    for epoch in range(epochs):

        random.shuffle(data)


        for seq,target in data:


            x=encode(seq).unsqueeze(0).to(device)

            y=torch.tensor(
                [target]
            ).to(device)



            out=model(x)


            loss=loss_fn(
                out,
                y
            )


            optimizer.zero_grad()

            loss.backward()

            optimizer.step()



    return time.time()-start




# ============================================================
# Evaluation
# ============================================================


def evaluate(model,data):

    model.eval()

    correct=0


    with torch.no_grad():

        for seq,target in data:

            x=encode(seq).unsqueeze(0).to(device)


            pred=torch.argmax(
                model(x)
            ).item()


            if pred==target:
                correct+=1


    return correct/len(data)




# ============================================================
# Experiment
# ============================================================


distances=[
    50,
    100,
    200
]


models={

    "Transformer":TransformerModel,

    "Mamba":MambaBaseline,

    "DIINA":DIINA

}


results=[]



for d in distances:


    print("\n====================")
    print("Distance:",d)
    print("====================")


    train_data=create_dataset(
        d,
        300
    )


    test_data=create_dataset(
        d,
        100
    )



    for name,model_class in models.items():


        model=model_class()


        t=train(
            model,
            train_data
        )


        acc=evaluate(
            model,
            test_data
        )


        print(
            name,
            "Accuracy:",
            round(acc,4),
            "Time:",
            round(t,2)
        )


        results.append({

            "Model":name,

            "Distance":d,

            "Accuracy":acc,

            "Time":t

        })



# ============================================================
# Save
# ============================================================


df=pd.DataFrame(results)


print("\n========== FINAL RESULTS ==========")

print(df)



df.to_csv(
    "DIINA_v19_1_LongDistance.csv",
    index=False
)



plt.figure(
    figsize=(8,5)
)


for model in df.Model.unique():

    temp=df[
        df.Model==model
    ]

    plt.plot(
        temp.Distance,
        temp.Accuracy,
        marker="o",
        label=model
    )


plt.xlabel(
    "Dependency Distance"
)

plt.ylabel(
    "Accuracy"
)

plt.title(
    "DIINA v19.1 Long Distance Dependency"
)


plt.legend()


plt.savefig(
    "DIINA_v19_1_Result.png",
    dpi=300
)



print("""
========================

FILES GENERATED:

DIINA_v19_1_LongDistance.csv

DIINA_v19_1_Result.png

========================
""")