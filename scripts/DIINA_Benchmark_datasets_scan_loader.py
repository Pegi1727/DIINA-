import torch
from torch.utils.data import Dataset, DataLoader


class SCANDataset(Dataset):
    """
    SCAN-like sequence transformation dataset

    Input:
        command sequence

    Output:
        action sequence
    """

    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets


    def __len__(self):
        return len(self.inputs)


    def __getitem__(self, index):

        return (
            torch.tensor(
                self.inputs[index],
                dtype=torch.long
            ),

            torch.tensor(
                self.targets[index],
                dtype=torch.long
            )
        )



def collate_fn(batch):

    inputs, targets = zip(*batch)

    inputs = torch.nn.utils.rnn.pad_sequence(
        inputs,
        batch_first=True,
        padding_value=0
    )

    targets = torch.nn.utils.rnn.pad_sequence(
        targets,
        batch_first=True,
        padding_value=0
    )

    return inputs, targets



def create_scan_loader(
        inputs,
        targets,
        batch_size=16,
        shuffle=True
):

    dataset = SCANDataset(
        inputs,
        targets
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )

    return loader



def load_dummy_scan():

    """
    Small reproducible SCAN example
    """

    inputs = [
        [1,2,3],
        [2,3,4],
        [5,6,7],
        [1,4,6]
    ]


    targets = [
        [3,4,5],
        [4,5,6],
        [6,7,8],
        [2,5,7]
    ]


    return inputs, targets