from Evol_Instruct import logger

class EvolDataset:
    def __init__(self, data):
        self.data = data 

    
    def __getitem__(self, idx) -> dict:
        return self.data[idx]

    def __len__(self) -> int:
        return len(self.data)

class EvolDataloader:
    def __init__(self, dataset: EvolDataset, batch_size: int = 1, shuffle: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(self.dataset)))
        if self.shuffle:
            self._shuffle_indices()
        self.current_idx = 0
    
    def __iter__(self):
        self.current_idx = 0  # Reset the index each time we start iterating
        if self.shuffle:
            self._shuffle_indices()  # Shuffle the data for each new iteration
        return self
    
    def __next__(self):
        if self.current_idx >= len(self.dataset):
            raise StopIteration
        
        batch_indices = self.indices[self.current_idx:self.current_idx + self.batch_size]
        batch_data = [self.dataset[i] for i in batch_indices]
        self.current_idx += self.batch_size
        return batch_data
    
    def _shuffle_indices(self):
        import random
        random.shuffle(self.indices)
    
    def __len__(self):
        # Return the number of batches
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
